"""Label-hiding steganographic trigger (the "first take" of encoder_decoder.ipynb), revisited.

Encoder: modules.Steganography.encoder.TriggerEncoder. Its input is the image concatenated with a
10-channel label map (the channel of the class to hide is all ones, the others all zeros), and it
outputs image + noise, with noise = noise_scale * tanh(...). Here noise_scale = eps, so the trigger's
magnitude is bounded by |delta| <= eps (6/255 by default) through the tanh output, with no MSE term.

Decoder: modules.Steganography.decoder.TriggerDecoder, a small CNN classifier that must recover the
hidden label from the poisoned image alone.

Training: on the whole CIFAR-10 training set, each image gets a hidden label drawn uniformly at random
(as in the first take), and the loss is

    lam_dec * CE(decoder(x_p), y_hidden)
  [ + lam_sur * CE(surrogate(augment(x_p)), y_hidden) ]     if a surrogate is given (adversarial loss)
  [ + lam_adv * non-saturating GAN loss of a PoisonDiscriminator ]   if use_disc (from adv_start_epoch)

x_p goes through 8-bit straight-through quantisation, as the poisoner outputs 8-bit images.

Checkpoint: {"state_dict", "decoder_state_dict", "noise_scale", "num_classes", "config", "history"},
loaded by datasets.SteganographicPoisonerv3 (flag "steganography_v3") to hide the target class.
"""

import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets

from modules.Steganography.decoder import TriggerDecoder
from modules.Steganography.deer_blend import PoisonDiscriminator, discriminator_loss, generator_adv_loss
from modules.Steganography.encoder import TriggerEncoder
from modules.Steganography.encoder_training import diff_augment, quantize_st
from modules.Steganography.surrogates import build_surrogate


def build_encoder(eps):
    enc = TriggerEncoder(image_channels=3, num_classes=10)
    enc.noise_scale = eps                      # |delta| <= eps, enforced by the tanh output
    return enc


def load_cifar_tensor(data_root, train, device):
    """Whole CIFAR-10 split as a [N, 3, 32, 32] tensor in [-1, 1] on `device`, plus labels."""
    ds = datasets.CIFAR10(data_root, train=train, download=False)
    x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).float().div(255).mul(2).sub(1)
    return x.to(device), torch.tensor(ds.targets, device=device)


def load_surrogate(arch, path, device):
    m = build_surrogate(arch).to(device)
    m.load_state_dict(torch.load(path, map_location=device)); m.eval()
    for q in m.parameters():
        q.requires_grad_(False)
    return m


@torch.no_grad()
def evaluate(encoder, decoder, x_test, y_test, device, surrogate=None, target=4, source=9, bs=500, seed=0):
    """Decoder accuracy on random hidden labels (whole test set), decoder accuracy and surrogate
    'deer' rate on trucks hiding deer, PSNR / Linf / saturation of the noise."""
    encoder.eval(); decoder.eval()
    g = torch.Generator(device="cpu").manual_seed(seed)
    hit = n = 0; mse = 0.0; linf = 0.0; sat = 0.0
    for i in range(0, len(x_test), bs):
        x = x_test[i:i + bs]
        y = torch.randint(0, 10, (len(x),), generator=g).to(device)
        x_p = quantize_st(encoder(x, y))
        hit += (decoder(x_p).argmax(1) == y).sum().item(); n += len(x)
        d = x_p - x
        mse += ((d / 2) ** 2).mean((1, 2, 3)).sum().item(); linf = max(linf, d.abs().max().item() / 2)
        sat += ((d.abs() >= 0.9 * encoder.noise_scale).float().mean((1, 2, 3))).sum().item()
    trucks = x_test[y_test == source]
    yt = torch.full((len(trucks),), target, device=device)
    xt = quantize_st(encoder(trucks, yt))
    out = dict(dec_acc=hit / n, psnr=10 * np.log10(1 / (mse / n)), linf_255=linf * 255, sat_frac=sat / n,
               dec_acc_truck_deer=(decoder(xt).argmax(1) == target).float().mean().item())
    if surrogate is not None:
        out["sur_deer_rate_trucks"] = (surrogate(xt).argmax(1) == target).float().mean().item()
    return out


def train_label_stego(name, out_dir, device, surrogate_arch=None, surrogate_path=None, use_disc=False,
                      eps_pixel=6 / 255, epochs=20, batch_size=64, lr=1e-3, lam_dec=1.0, lam_sur=5.0,
                      lam_adv=10.0, adv_start_epoch=5, data_root="data/data_cifar10", seed=0,
                      overwrite=False, verbose=True):
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, f"{name}.pth")
    summary_path = os.path.join(out_dir, f"{name}.json")
    if not overwrite and os.path.exists(ckpt_path) and os.path.exists(summary_path):
        print(f"[{name}] exists, skipping", flush=True)
        return json.load(open(summary_path))

    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)
    x_train, _ = load_cifar_tensor(data_root, True, device)
    x_test, y_test = load_cifar_tensor(data_root, False, device)

    eps = 2 * eps_pixel                                   # images live in [-1, 1]
    encoder, decoder = build_encoder(eps).to(device), TriggerDecoder(3, 10).to(device)
    surrogate = load_surrogate(surrogate_arch, surrogate_path, device) if surrogate_arch else None
    disc = PoisonDiscriminator().to(device) if use_disc else None

    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    steps = epochs * (len(x_train) // batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    opt_d = torch.optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.999)) if use_disc else None

    terms = [f"{lam_dec}*CE(decoder)"]
    if surrogate is not None:
        terms.append(f"{lam_sur}*CE({surrogate_arch} surrogate -> hidden label)")
    if use_disc:
        terms.append(f"{lam_adv}*GAN(discriminator, from epoch {adv_start_epoch})")
    print(f"[{name}] whole CIFAR-10, random hidden labels, |delta| <= {eps_pixel*255:.0f}/255 via tanh, "
          f"loss = {' + '.join(terms)}", flush=True)

    history = []; t0 = time.time()
    for epoch in range(epochs):
        encoder.train(); decoder.train()
        if disc is not None:
            disc.train()
        use_adv = use_disc and epoch >= adv_start_epoch
        perm = torch.randperm(len(x_train), device=device)
        st = np.zeros(6); nb = 0; d_acc = float("nan")
        for i in range(0, len(x_train) - batch_size + 1, batch_size):
            x = x_train[perm[i:i + batch_size]]
            y = torch.randint(0, 10, (len(x),), device=device)
            x_p = quantize_st(encoder(x, y))

            if disc is not None:
                d_loss, d_acc_t = discriminator_loss(disc, x, x_p.detach())
                opt_d.zero_grad(set_to_none=True); d_loss.backward(); opt_d.step()
                d_acc = d_acc_t.item()

            logits = decoder(x_p)
            dec_loss = F.cross_entropy(logits, y)
            loss = lam_dec * dec_loss
            sur_loss = torch.zeros((), device=device); sur_hit = 0.0
            if surrogate is not None:
                s_logits = surrogate(diff_augment(x_p))
                sur_loss = F.cross_entropy(s_logits, y)
                sur_hit = (s_logits.argmax(1) == y).float().mean().item()
                loss = loss + lam_sur * sur_loss
            adv = generator_adv_loss(disc, x_p) if use_adv else torch.zeros((), device=device)
            loss = loss + (lam_adv * adv if use_adv else 0.0)

            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            st += np.array([loss.item(), dec_loss.item(), (logits.argmax(1) == y).float().mean().item(),
                            sur_loss.item(), sur_hit, adv.item()]); nb += 1
        st /= nb
        ev = evaluate(encoder, decoder, x_test, y_test, device, surrogate)
        rec = dict(epoch=epoch + 1, loss=st[0], dec_loss=st[1], train_dec_acc=st[2], sur_loss=st[3],
                   train_sur_hit=st[4], adv=st[5], d_acc=d_acc if use_disc else None, **ev)
        history.append(rec)
        if verbose:
            msg = (f"[{name}] epoch {epoch+1}/{epochs} | loss {st[0]:.3f} | dec CE {st[1]:.3f} acc {st[2]:.3f}"
                   + (f" | sur CE {st[3]:.3f} hit {st[4]:.3f}" if surrogate is not None else "")
                   + (f" | adv {st[5]:.3f} D acc {d_acc:.3f}" if use_disc else "")
                   + f" || TEST dec acc {ev['dec_acc']:.3f} | truck->deer dec {ev['dec_acc_truck_deer']:.3f}"
                   + (f" sur {ev['sur_deer_rate_trucks']:.3f}" if surrogate is not None else "")
                   + f" | PSNR {ev['psnr']:.1f} | Linf {ev['linf_255']:.1f}/255 | sat {ev['sat_frac']:.3f}")
            print(msg, flush=True)

    config = dict(name=name, surrogate_arch=surrogate_arch, surrogate_path=surrogate_path, use_disc=use_disc,
                  eps_pixel=eps_pixel, epochs=epochs, batch_size=batch_size, lr=lr, lam_dec=lam_dec,
                  lam_sur=lam_sur if surrogate is not None else 0.0, lam_adv=lam_adv if use_disc else 0.0,
                  adv_start_epoch=adv_start_epoch, seed=seed)
    torch.save({"state_dict": encoder.state_dict(), "decoder_state_dict": decoder.state_dict(),
                "noise_scale": eps, "num_classes": 10, "config": config, "history": history}, ckpt_path)
    summary = dict(config, **history[-1], train_minutes=round((time.time() - t0) / 60, 1),
                   checkpoint=ckpt_path, history=history)
    json.dump(summary, open(summary_path, "w"), indent=2)
    print(f"[{name}] done in {summary['train_minutes']} min", flush=True)
    return summary
