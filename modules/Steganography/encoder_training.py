"""Train a deer-in-truck steganography encoder against a surrogate ensemble.

Ports the training loop from encoder_decoder.ipynb into a reusable module, so the FLIP
experiment matrix (architecture x ensemble size) can be driven from a script.

Each experiment:
  * loads an ensemble of surrogates (one architecture, seeds 0..ensemble-1),
  * selects the top-K target deer by worst-case surrogate margin,
  * trains a DeerBlendEncoder (optionally with a PoisonDiscriminator in the loop),
  * saves a checkpoint compatible with datasets.SteganographicPoisonerv2 and a JSON summary
    (surrogate deer rate, in-loop and fresh discriminator accuracy, PSNR, Linf).

CLI:
  python -m modules.Steganography.encoder_training --arch resnet18 --ensemble 5 --k 1 \
      --eps-pixel 0.02352941 --sigma 0.3 --lam-adv 10 --epochs 40 --out noise_generator/encoders
"""

import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from modules.Steganography.deer_blend import (
    DeerBlendEncoder, tv_loss, masked_l2_loss, saturation_loss,
    PoisonDiscriminator, discriminator_loss, generator_adv_loss)
from modules.Steganography.surrogates import build_surrogate, extract_features

NORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)])


# --------------------------------------------------------------------------- data

class _ClassImages(Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.images[i], 0


def _class_images(data_root, train, label):
    ds = datasets.CIFAR10(data_root, train=train, download=False, transform=NORM)
    idx = [i for i, y in enumerate(ds.targets) if y == label]
    return torch.stack([ds[i][0] for i in idx]), idx


# --------------------------------------------------------------------------- straight-through

def quantize_st(x):
    x01 = (x + 1) / 2
    xq = torch.round(x01 * 255) / 255
    return (x01 + (xq - x01).detach()) * 2 - 1


def diff_augment(x, max_shift=3):
    B = x.size(0)
    flip = torch.rand(B, device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    pad = F.pad(x, [max_shift] * 4, mode="reflect")
    dx, dy = np.random.randint(0, 2 * max_shift + 1, size=2)
    return pad[:, :, dy:dy + 32, dx:dx + 32]


# --------------------------------------------------------------------------- losses / eval

def surrogate_losses(surrogates, deer_feats, x_p, deer_idx, target_label):
    """surrogates: list of (model, arch). deer_feats: list of [K, feat_dim] per surrogate."""
    feat_loss = ce_loss = 0.0
    fooled = []
    y = torch.full((x_p.size(0),), target_label, device=x_p.device, dtype=torch.long)
    for (model, arch), f_bank in zip(surrogates, deer_feats):
        f = extract_features(model, x_p, arch)
        logits = model.linear(f)
        f_deer = f_bank[deer_idx]                                     # [B, feat_dim]
        feat_loss = feat_loss + (((f - f_deer) ** 2).sum(1) / (f_deer ** 2).sum(1)).mean()
        ce_loss = ce_loss + F.cross_entropy(logits, y)
        fooled.append((logits.argmax(1) == target_label).float().mean().item())
    K = len(surrogates)
    return feat_loss / K, ce_loss / K, fooled


@torch.no_grad()
def evaluate(encoder, loader, surrogates, deer_bank, target_label, device, disc=None):
    encoder.eval()
    hits = torch.zeros(len(surrogates)); n = 0; linf = 0.0; mse = 0.0; d_acc = 0.0
    for x, _ in loader:
        x = x.to(device)
        deers = deer_bank[torch.randint(0, deer_bank.size(0), (x.size(0),), device=device)]
        x_p = quantize_st(encoder(x, deers))
        for k, (model, _) in enumerate(surrogates):
            hits[k] += (model(x_p).argmax(1) == target_label).sum().item()
        if disc is not None:
            d_acc += discriminator_loss(disc, x, x_p)[1].item() * x.size(0)
        linf = max(linf, (x_p - x).abs().max().item() / 2)
        mse += (((x_p - x) / 2) ** 2).mean(dim=(1, 2, 3)).sum().item()
        n += x.size(0)
    psnr = 10 * np.log10(1.0 / (mse / n))
    return (hits / n).tolist(), linf, psnr, (d_acc / n if disc is not None else None)


# --------------------------------------------------------------------------- training

def train_encoder(encoder, loader, test_loader, surrogates, deer_bank, deer_feats, target_label,
                  device, epochs=40, lr=1e-3, lam_feat=1.0, lam_ce=5.0, lam_l2=1.0, lam_tv=0.0,
                  lam_sat=0.0, disc=None, lam_adv=0.0, adv_start_epoch=10, eval_every=10, verbose=True):
    opt = torch.optim.Adam(encoder.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(loader))
    opt_d = torch.optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.999)) if disc is not None else None
    K = deer_bank.size(0)
    history = []
    for epoch in range(epochs):
        encoder.train()
        if disc is not None:
            disc.train()
        use_adv = disc is not None and epoch >= adv_start_epoch
        stats = np.zeros(6); n = 0; d_acc = float("nan")
        for x, _ in loader:
            x = x.to(device)
            deer_idx = torch.randint(0, K, (x.size(0),), device=device)
            delta, z = encoder.perturbation(x, deer_bank[deer_idx], return_raw=True)
            x_p = quantize_st(torch.clamp(x + delta, -1, 1))

            if disc is not None:
                d_loss, d_acc_t = discriminator_loss(disc, x, x_p.detach())
                opt_d.zero_grad(set_to_none=True); d_loss.backward(); opt_d.step()
                d_acc = d_acc_t.item()

            feat_loss, ce_loss, fooled = surrogate_losses(surrogates, deer_feats, diff_augment(x_p),
                                                          deer_idx, target_label)
            l2 = masked_l2_loss(delta, encoder.eps, x)
            tv = tv_loss(delta, encoder.eps)
            sat = saturation_loss(z)
            adv = generator_adv_loss(disc, x_p) if use_adv else torch.zeros((), device=device)
            loss = (lam_feat * feat_loss + lam_ce * ce_loss + lam_l2 * l2 + lam_tv * tv
                    + lam_sat * sat + lam_adv * adv)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()

            stats += np.array([loss.item(), feat_loss.item(), ce_loss.item(),
                               np.mean(fooled), adv.item() if use_adv else 0.0,
                               d_acc if disc is not None else 0.0]); n += 1
        stats /= n
        msg = (f"epoch {epoch+1}/{epochs} | loss {stats[0]:.3f} | feat {stats[1]:.3f} | "
               f"CE {stats[2]:.3f} | train p(target) {stats[3]:.3f}")
        if disc is not None:
            msg += f" | adv {stats[4]:.3f} | D acc {stats[5]:.3f}"
        if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
            rates, linf, psnr, d_test = evaluate(encoder, test_loader, surrogates, deer_bank,
                                                 target_label, device, disc)
            msg += f" || TEST rate {[round(r,3) for r in rates]} | Linf {linf*255:.1f}/255 | PSNR {psnr:.1f}"
            if d_test is not None:
                msg += f" | D acc {d_test:.3f}"
            history.append(dict(epoch=epoch + 1, rates=rates, linf=linf, psnr=psnr, d_acc=d_test))
        if verbose:
            print(msg, flush=True)
    return history


def fit_fresh_detector(encoder, loader, test_loader, surrogates, deer_bank, target_label,
                       device, epochs=8, lr=2e-4, verbose=True):
    encoder.eval()
    det = PoisonDiscriminator().to(device)
    opt = torch.optim.Adam(det.parameters(), lr=lr, betas=(0.5, 0.999))
    acc = None
    for epoch in range(epochs):
        det.train()
        for x, _ in loader:
            x = x.to(device)
            with torch.no_grad():
                deers = deer_bank[torch.randint(0, deer_bank.size(0), (x.size(0),), device=device)]
                x_p = quantize_st(encoder(x, deers))
            loss, _ = discriminator_loss(det, x, x_p)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        _, _, _, acc = evaluate(encoder, test_loader, surrogates, deer_bank, target_label, device, det)
        if verbose:
            print(f"fresh detector epoch {epoch+1}/{epochs}: test clean-vs-poisoned acc {acc:.3f}", flush=True)
    return acc


# --------------------------------------------------------------------------- experiment

def select_top_k_deer(surrogates, deer_imgs, deer_pos, target_label, k, device):
    margins = []
    with torch.no_grad():
        for model, _ in surrogates:
            logits = torch.cat([model(deer_imgs[b:b + 512].to(device)).cpu()
                                for b in range(0, len(deer_imgs), 512)])
            other = logits.clone(); other[:, target_label] = float("-inf")
            margins.append(logits[:, target_label] - other.max(1).values)
    margin = torch.stack(margins).min(0).values
    top = margin.argsort(descending=True)[:k]
    return deer_imgs[top].contiguous().to(device), [deer_pos[j] for j in top.tolist()]


def run_experiment(arch, ensemble, out_dir, device, k=1, eps_pixel=6 / 255, sigma=0.3,
                   lam_adv=10.0, adv_start_epoch=10, tanh_temp=1.0, epochs=40,
                   source_label=9, target_label=4, data_root="data/data_cifar10",
                   surrogate_dir="noise_generator/surrogates", fresh_det_epochs=8, seed=0):
    """Train one (arch, ensemble) encoder; save checkpoint + summary. Skips if checkpoint exists."""
    os.makedirs(out_dir, exist_ok=True)
    name = f"{arch}_ens{ensemble}_k{k}"
    ckpt_path = os.path.join(out_dir, f"{name}.pth")
    summary_path = os.path.join(out_dir, f"{name}.json")
    surrogate_paths = [os.path.join(surrogate_dir, f"{arch}_seed{s}.pth") for s in range(ensemble)]

    if os.path.exists(ckpt_path) and os.path.exists(summary_path):
        print(f"[{name}] checkpoint exists, skipping")
        return json.load(open(summary_path))

    missing = [p for p in surrogate_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"missing surrogate checkpoints: {missing}")

    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)

    # surrogates
    surrogates = []
    for p in surrogate_paths:
        m = build_surrogate(arch).to(device)
        m.load_state_dict(torch.load(p, map_location=device)); m.eval()
        for q in m.parameters():
            q.requires_grad_(False)
        surrogates.append((m, arch))

    # data: trucks (train/test) + deer bank
    trucks_train, _ = _class_images(data_root, True, source_label)
    trucks_test, _ = _class_images(data_root, False, source_label)
    deer_imgs, deer_pos = _class_images(data_root, True, target_label)
    train_loader = DataLoader(_ClassImages(trucks_train), batch_size=64, shuffle=True, num_workers=4, drop_last=True)
    test_loader = DataLoader(_ClassImages(trucks_test), batch_size=64, shuffle=False, num_workers=4)

    deer_bank, deer_idx = select_top_k_deer(surrogates, deer_imgs, deer_pos, target_label, k, device)
    with torch.no_grad():
        deer_feats = [extract_features(m, deer_bank, a) for m, a in surrogates]

    # encoder + discriminator
    eps = 2 * eps_pixel
    encoder = DeerBlendEncoder(eps=eps, blur_sigma=sigma, tanh_temp=tanh_temp, use_mask=False).to(device)
    disc = PoisonDiscriminator().to(device)

    print(f"[{name}] training: {ensemble}x {arch}, k={k}, eps={round(eps_pixel*255)}/255, "
          f"sigma={sigma}, lam_adv={lam_adv}")
    t0 = time.time()
    history = train_encoder(encoder, train_loader, test_loader, surrogates, deer_bank, deer_feats,
                            target_label, device, epochs=epochs, disc=disc, lam_adv=lam_adv,
                            adv_start_epoch=adv_start_epoch)
    fresh_acc = fit_fresh_detector(encoder, train_loader, test_loader, surrogates, deer_bank,
                                   target_label, device, epochs=fresh_det_epochs)
    train_min = (time.time() - t0) / 60

    torch.save({"state_dict": encoder.state_dict(), "eps": eps, "mask_tau": encoder.mask_tau,
                "mask_floor": encoder.mask_floor, "blur_sigma": sigma, "tanh_temp": tanh_temp,
                "use_mask": False, "deer_idx": deer_idx, "deer_bank": deer_bank.cpu()}, ckpt_path)

    last = history[-1]
    summary = dict(name=name, arch=arch, ensemble=ensemble, k=k, eps_pixel=eps_pixel, sigma=sigma,
                   lam_adv=lam_adv, epochs=epochs, surrogate_paths=surrogate_paths,
                   deer_idx=deer_idx, surrogate_deer_rate=last["rates"],
                   mean_surrogate_deer_rate=float(np.mean(last["rates"])),
                   train_discriminator_acc=last["d_acc"], fresh_discriminator_acc=fresh_acc,
                   psnr=last["psnr"], linf_255=last["linf"] * 255, train_minutes=round(train_min, 1),
                   checkpoint=ckpt_path)
    json.dump(summary, open(summary_path, "w"), indent=2)
    print(f"[{name}] done in {train_min:.1f} min | deer rate {summary['mean_surrogate_deer_rate']:.3f} "
          f"| fresh D {fresh_acc:.3f} | PSNR {last['psnr']:.1f}")
    return summary


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", required=True, choices=["resnet18", "resnet32", "efficientnet"])
    p.add_argument("--ensemble", type=int, required=True)
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--eps-pixel", type=float, default=6 / 255)
    p.add_argument("--sigma", type=float, default=0.3)
    p.add_argument("--lam-adv", type=float, default=10.0)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--out", default="noise_generator/encoders")
    p.add_argument("--data-root", default="data/data_cifar10")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_experiment(args.arch, args.ensemble, args.out, device, k=args.k, eps_pixel=args.eps_pixel,
                   sigma=args.sigma, lam_adv=args.lam_adv, epochs=args.epochs, data_root=args.data_root)
