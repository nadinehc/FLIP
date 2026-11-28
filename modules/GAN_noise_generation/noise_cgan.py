"""Image-conditional noise GAN (truck -> deer), bounded by |delta|_inf <= eps through tanh.

Generator  G: UNetCIFARGenerator (conditioned on the clean truck x, dropout off), used as a residual:
               x_p = Q(clamp(x + eps * G(x), -1, 1))      (G ends with tanh, so |delta| <= eps)
           Q is 8-bit straight-through quantisation: the FLIP poisoner outputs 8-bit images.
Discriminator D: says whether an image has been modified (clean -> 1, modified -> 0).
           "unconditional" (default): Steganography.deer_blend.PoisonDiscriminator, sees one image.
           "conditional": GAN_noise_generation.discriminator.CIFARTruckDiscriminatorFixed, sees the
               (clean, candidate) pair as in pix2pix. Patch logits (4x4), averaged in the BCE.
Losses     L_D = BCE(D(clean), 1) + BCE(D(x_p), 0)
           L_G = lam_disc * BCE(D(x_p), 1)                     (non-saturating GAN loss)
               + lam_sur  * CE(surrogate(augment(x_p)), deer)  (only if a surrogate is given)

Everything lives in [-1, 1] (Normalize(0.5, 0.5)), like the surrogates in noise_generator/surrogates.
Training data: the 5,000 CIFAR-10 training trucks; evaluation: the 1,000 test trucks.

Checkpoint <out>/<name>.pth: {"state_dict", "eps", "config", "history"} (eps in [-1, 1] units), plus
<name>.json (config + per-epoch history) and <name>_cache.npz (8-bit poisoned versions of every
CIFAR-10 truck, keyed by the SHA-1 of the clean pixels, read by datasets.NoiseCGANPoisoner).
"""

import hashlib
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets

from modules.GAN_noise_generation.discriminator import CIFARTruckDiscriminatorFixed
from modules.GAN_noise_generation.generator import UNetCIFARGenerator
from modules.Steganography.deer_blend import PoisonDiscriminator
from modules.Steganography.encoder_training import diff_augment, quantize_st
from modules.Steganography.surrogates import build_surrogate

SOURCE, TARGET = 9, 4                      # truck -> deer
SURR_DIR = "noise_generator/surrogates"
ARCHS = ["resnet18", "resnet32", "efficientnet"]


class NoiseCGAN(torch.nn.Module):
    """x -> clamp(x + eps * UNet(x)); UNet ends with tanh, so |delta| <= eps (images in [-1, 1])."""

    def __init__(self, eps):
        super().__init__()
        self.eps = eps
        self.unet = UNetCIFARGenerator(dropout_p=0.0)

    def forward(self, x):
        return torch.clamp(x + self.eps * self.unet(x), -1.0, 1.0)


def load_generator(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    g = NoiseCGAN(ck["eps"])
    g.load_state_dict(ck["state_dict"])
    return g.to(device).eval()


# --------------------------------------------------------------------------- data / models

def cifar_uint8(data_root, train):
    ds = datasets.CIFAR10(data_root, train=train, download=False)
    return ds.data, np.array(ds.targets)                     # [N, 32, 32, 3] uint8, [N]


def to_unit(x_uint8, device):
    return torch.from_numpy(x_uint8).permute(0, 3, 1, 2).float().div(255).mul(2).sub(1).to(device)


def load_surrogate(arch, seed, device):
    m = build_surrogate(arch).to(device)
    m.load_state_dict(torch.load(os.path.join(SURR_DIR, f"{arch}_seed{seed}.pth"), map_location=device))
    m.eval()
    for q in m.parameters():
        q.requires_grad_(False)
    return m


class Disc:
    """Uniform interface over the two discriminators: logits(clean, candidate) -> [B, ...]."""

    def __init__(self, kind, device):
        self.kind = kind
        self.net = (PoisonDiscriminator() if kind == "unconditional" else CIFARTruckDiscriminatorFixed()).to(device)

    def logits(self, clean, cand):
        return self.net(cand) if self.kind == "unconditional" else self.net(clean, cand)

    def d_loss(self, clean, poisoned):
        lc, lp = self.logits(clean, clean), self.logits(clean, poisoned)
        loss = F.binary_cross_entropy_with_logits(lc, torch.ones_like(lc)) + \
               F.binary_cross_entropy_with_logits(lp, torch.zeros_like(lp))
        acc = 0.5 * ((lc > 0).float().mean() + (lp < 0).float().mean())
        return loss, acc

    def g_loss(self, clean, poisoned):
        lp = self.logits(clean, poisoned)
        return F.binary_cross_entropy_with_logits(lp, torch.ones_like(lp))


# --------------------------------------------------------------------------- evaluation

@torch.no_grad()
def poison_batched(gen, x, bs=500):
    return torch.cat([quantize_st(gen(x[i:i + bs])) for i in range(0, len(x), bs)])


@torch.no_grad()
def deer_rate(model, x, bs=500):
    return float(torch.cat([model(x[i:i + bs]).argmax(1) for i in range(0, len(x), bs)]).eq(TARGET).float().mean())


@torch.no_grad()
def evaluate(gen, disc, x_test, eval_models):
    gen.eval(); disc.net.eval()
    xp = poison_batched(gen, x_test)
    d = (xp - x_test) / 2                                    # in [0, 1] pixel units
    dl = da = 0.0
    for i in range(0, len(x_test), 500):
        l, a = disc.d_loss(x_test[i:i + 500], xp[i:i + 500])
        dl += l.item() * len(xp[i:i + 500]); da += a.item() * len(xp[i:i + 500])
    out = dict(test_d_loss=dl / len(xp), test_d_acc=da / len(xp),
               linf_255=round(d.abs().max().item() * 255, 3),
               rms_255=d.pow(2).mean().sqrt().item() * 255,
               psnr=float(10 * np.log10(1 / d.pow(2).mean((1, 2, 3)).mean().item())))
    for k, m in eval_models.items():
        out[f"deer_{k}"] = deer_rate(m, xp)
    return out


# --------------------------------------------------------------------------- training

def train_noise_cgan(name, out_dir, device, surrogate_arch=None, disc_kind="unconditional",
                     eps_pixel=6 / 255, epochs=50, batch_size=64, lr_g=2e-4, lr_d=2e-4,
                     lam_disc=2.0, lam_sur=1.0, data_root="data/data_cifar10", seed=0,
                     overwrite=False):
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path, summary_path = (os.path.join(out_dir, f"{name}.{e}") for e in ("pth", "json"))
    if not overwrite and os.path.exists(ckpt_path) and os.path.exists(summary_path):
        print(f"[{name}] exists, skipping", flush=True)
        return json.load(open(summary_path))

    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)
    tr_x, tr_y = cifar_uint8(data_root, True)
    te_x, te_y = cifar_uint8(data_root, False)
    x_train, x_test = to_unit(tr_x[tr_y == SOURCE], device), to_unit(te_x[te_y == SOURCE], device)

    eps = 2 * eps_pixel                                      # [-1, 1] units
    gen = NoiseCGAN(eps).to(device)
    disc = Disc(disc_kind, device)
    surrogate = load_surrogate(surrogate_arch, 0, device) if surrogate_arch else None
    # Attacked surrogate, a held-out seed of the same arch, and held-out ResNet-32 (the FLIP expert arch).
    eval_models = {}
    if surrogate_arch:
        eval_models["attacked"] = surrogate
        eval_models[f"heldout_{surrogate_arch}"] = load_surrogate(surrogate_arch, 1, device)
    if surrogate_arch != "resnet32":
        eval_models["heldout_resnet32"] = load_surrogate("resnet32", 1, device)

    opt_g = torch.optim.Adam(gen.parameters(), lr=lr_g, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.net.parameters(), lr=lr_d, betas=(0.5, 0.999))
    target = torch.full((batch_size,), TARGET, device=device, dtype=torch.long)

    terms = [f"{lam_disc}*GAN({disc_kind} D)"] + ([f"{lam_sur}*CE({surrogate_arch} -> deer)"] if surrogate else [])
    print(f"[{name}] {len(x_train)} train trucks, |delta| <= {eps_pixel * 255:.0f}/255 via tanh, "
          f"L_G = {' + '.join(terms)}, {epochs} epochs", flush=True)

    history, t0 = [], time.time()
    for epoch in range(epochs):
        gen.train(); disc.net.train()
        perm = torch.randperm(len(x_train), device=device)
        st = np.zeros(6); nb = 0
        for i in range(0, len(x_train) - batch_size + 1, batch_size):
            x = x_train[perm[i:i + batch_size]]
            xp = quantize_st(gen(x))

            d_loss, d_acc = disc.d_loss(x, xp.detach())
            opt_d.zero_grad(set_to_none=True); d_loss.backward(); opt_d.step()

            g_adv = disc.g_loss(x, xp)
            loss = lam_disc * g_adv
            sur_loss = torch.zeros((), device=device); fooled = 0.0
            if surrogate is not None:
                s_logits = surrogate(diff_augment(xp))
                sur_loss = F.cross_entropy(s_logits, target)
                fooled = s_logits.argmax(1).eq(TARGET).float().mean().item()
                loss = loss + lam_sur * sur_loss
            opt_g.zero_grad(set_to_none=True); loss.backward(); opt_g.step()

            st += [d_loss.item(), d_acc.item(), g_adv.item(), sur_loss.item(), fooled, loss.item()]; nb += 1
        st /= nb
        ev = evaluate(gen, disc, x_test, eval_models)
        rec = dict(epoch=epoch + 1, d_loss=st[0], d_acc=st[1], g_adv_loss=st[2],
                   sur_loss=st[3] if surrogate else None, train_deer=st[4] if surrogate else None,
                   g_loss=st[5], **ev)
        history.append(rec)
        print(f"[{name}] ep {epoch + 1:2d}/{epochs} | D loss {st[0]:.3f} acc {st[1]:.3f} | G adv {st[2]:.3f}"
              + (f" | sur CE {st[3]:.3f} deer {st[4]:.3f}" if surrogate else "")
              + f" || test D acc {ev['test_d_acc']:.3f} | "
              + " ".join(f"{k[5:]} {v:.3f}" for k, v in ev.items() if k.startswith("deer_"))
              + f" | Linf {ev['linf_255']:.0f}/255 RMS {ev['rms_255']:.2f} PSNR {ev['psnr']:.1f}", flush=True)

    config = dict(name=name, surrogate_arch=surrogate_arch, disc=disc_kind, eps_pixel=eps_pixel,
                  epochs=epochs, batch_size=batch_size, lr_g=lr_g, lr_d=lr_d, lam_disc=lam_disc,
                  lam_sur=lam_sur if surrogate else 0.0, seed=seed)
    gen.eval()
    torch.save({"state_dict": gen.state_dict(), "eps": eps, "config": config, "history": history}, ckpt_path)
    write_cache(gen, np.concatenate([tr_x[tr_y == SOURCE], te_x[te_y == SOURCE]]),
                os.path.join(out_dir, f"{name}_cache.npz"), device)
    summary = dict(config, **history[-1], train_minutes=round((time.time() - t0) / 60, 1),
                   checkpoint=ckpt_path, history=history)
    json.dump(summary, open(summary_path, "w"), indent=2)
    print(f"[{name}] done in {summary['train_minutes']} min", flush=True)
    return summary


def image_key(x_uint8_hwc):
    return hashlib.sha1(np.ascontiguousarray(x_uint8_hwc, dtype=np.uint8).tobytes()).hexdigest()


@torch.no_grad()
def poison_uint8(gen, x_uint8, device="cpu", bs=500):
    """[N, 32, 32, 3] uint8 -> poisoned uint8, with |delta| <= round(eps * 255 / 2) exactly."""
    out = []
    lim = int(round(gen.eps / 2 * 255))
    for i in range(0, len(x_uint8), bs):
        xb = x_uint8[i:i + bs]
        delta = gen.eps * gen.unet(to_unit(xb, device)) / 2 * 255      # pixel units
        delta = torch.round(delta).clamp(-lim, lim).permute(0, 2, 3, 1).cpu().numpy().astype(np.int16)
        out.append(np.clip(xb.astype(np.int16) + delta, 0, 255).astype(np.uint8))
    return np.concatenate(out)


def write_cache(gen, x_uint8, path, device):
    gen = gen.to(device).eval()
    xp = poison_uint8(gen, x_uint8, device)
    np.savez_compressed(path, keys=np.array([image_key(x) for x in x_uint8]), images=xp)
