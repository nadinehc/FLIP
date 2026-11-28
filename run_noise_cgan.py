#!/usr/bin/env python
"""Train the four truck -> deer noise cGANs, then plot their losses across epochs.

    cgan_efficientnet   L_G = 2 * GAN(D) + 1 * CE(EfficientNet-B0 surrogate -> deer)
    cgan_resnet18       L_G = 2 * GAN(D) + 1 * CE(ResNet-18 surrogate -> deer)
    cgan_resnet32       L_G = 2 * GAN(D) + 1 * CE(ResNet-32 surrogate -> deer)
    cgan_nosurrogate    L_G = 2 * GAN(D)

All: U-Net generator conditioned on the truck, |delta| <= 6/255 via tanh, 50 epochs, surrogates
noise_generator/surrogates/<arch>_seed0.pth. See modules/GAN_noise_generation/noise_cgan.py.

Outputs: noise_generator/cgan/<name>.{pth,json} + <name>_cache.npz,
         experiments/cgan_noise/{loss_summary.png,pdf, summary.csv}.

    python run_noise_cgan.py                       # train missing ones + plot
    python run_noise_cgan.py --steps plot
    python run_noise_cgan.py --only cgan_resnet18 --overwrite
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

OUT_DIR = "noise_generator/cgan"
FIG_DIR = Path("experiments/cgan_noise")
VARIANTS = {                                # name -> surrogate arch
    "cgan_efficientnet": "efficientnet",
    "cgan_resnet18": "resnet18",
    "cgan_resnet32": "resnet32",
    "cgan_nosurrogate": None,
}
LABEL = {"cgan_efficientnet": "EfficientNet-B0", "cgan_resnet18": "ResNet-18",
         "cgan_resnet32": "ResNet-32", "cgan_nosurrogate": "No surrogate"}
COLOR = {"cgan_efficientnet": "#2a78b5", "cgan_resnet18": "#d1603d",
         "cgan_resnet32": "#3d9a5b", "cgan_nosurrogate": "#7a7a7a"}


def plot(names, out_dir=OUT_DIR, fig_dir=FIG_DIR, suffix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    H = {n: json.load(open(os.path.join(out_dir, f"{n}.json")))["history"]
         for n in names if os.path.exists(os.path.join(out_dir, f"{n}.json"))}
    panels = [("d_loss", "Discriminator loss  $L_D$", None),
              ("g_adv_loss", "Generator adversarial loss  BCE(D(x'), clean)", None),
              ("sur_loss", "Surrogate loss  CE(s(x'), deer)", None),
              ("deer_attacked", "Truck→deer rate, attacked surrogate (test trucks)", (0, 1.02)),
              ("test_d_acc", "Discriminator accuracy, clean vs modified (test)", (0.4, 1.02)),
              ("rms_255", "Noise RMS (pixel units /255, bound 6)", None)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2))
    for ax, (key, title, ylim) in zip(axes.flat, panels):
        for n, h in H.items():
            ys = [r.get(key) for r in h]
            if all(y is None for y in ys):
                continue
            ax.plot([r["epoch"] for r in h], ys, label=LABEL.get(n, n), color=COLOR.get(n), lw=1.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("epoch")
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(alpha=.2); ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(fontsize=9, frameon=False)
    fig.suptitle("Noise cGANs (truck → deer, ε = 6/255) · $L_G$ = 2·GAN + 1·surrogate CE", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .96))
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"loss_summary{suffix}.{ext}", dpi=200, bbox_inches="tight")
    print(f"saved {fig_dir}/loss_summary{suffix}.png", flush=True)

    keys = ["epoch", "d_loss", "d_acc", "g_adv_loss", "sur_loss", "train_deer", "test_d_acc",
            "deer_attacked", "deer_heldout_resnet18", "deer_heldout_resnet32", "deer_heldout_efficientnet",
            "linf_255", "rms_255", "psnr"]
    with open(fig_dir / f"summary{suffix}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["name"] + keys)
        for n, h in H.items():
            w.writerow([n] + [h[-1].get(k, "") for k in keys])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", nargs="+", choices=["train", "plot"], default=["train", "plot"])
    p.add_argument("--only", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--disc", choices=["unconditional", "conditional"], default="unconditional")
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    if "train" in a.steps:
        import torch
        from modules.GAN_noise_generation.noise_cgan import train_noise_cgan
        for n in a.only:
            train_noise_cgan(n, a.out_dir, torch.device("cuda"), surrogate_arch=VARIANTS[n],
                             disc_kind=a.disc, epochs=a.epochs, overwrite=a.overwrite)
    if "plot" in a.steps:
        plot(list(VARIANTS), out_dir=a.out_dir,
             suffix="" if a.out_dir == OUT_DIR else "_" + Path(a.out_dir).name)


if __name__ == "__main__":
    main()
