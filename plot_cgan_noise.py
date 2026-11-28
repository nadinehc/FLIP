#!/usr/bin/env python
"""Compare the noise learned by the four noise cGANs on the same held-out CIFAR-10 test trucks.

Figure (experiments/cgan_noise/noise_comparison.{png,pdf}):
  rows 1..N   one test truck each: clean | poisoned (x4 strategies) | signed noise (x4 strategies)
  last row    mean |delta| over all 1,000 test trucks, per strategy
Noise display: fixed scale, gray = 0, saturated at +-6/255 (the L_inf bound), same for every strategy.
Noise is the exact 8-bit difference the FLIP poisoner applies.

    python plot_cgan_noise.py [--n 4] [--ids 0 5 9 13]
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from modules.GAN_noise_generation.noise_cgan import SOURCE, cifar_uint8, load_generator, poison_uint8
from run_noise_cgan import LABEL, OUT_DIR, VARIANTS

EPS_PX = 6


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=4, help="number of example trucks")
    p.add_argument("--ids", type=int, nargs="+", default=None, help="indices among the test trucks")
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    te_x, te_y = cifar_uint8("data/data_cifar10", False)
    trucks = te_x[te_y == SOURCE]
    ids = a.ids or list(range(a.n))

    noise, poisoned = {}, {}
    for n in VARIANTS:
        g = load_generator(os.path.join(OUT_DIR, f"{n}.pth"), device)
        xp = poison_uint8(g, trucks, device)
        poisoned[n] = xp
        noise[n] = xp.astype(np.int16) - trucks.astype(np.int16)        # [N, 32, 32, 3], pixel units
        d = noise[n]
        print(f"{LABEL[n]:16s} Linf {np.abs(d).max():d}/255 | RMS {np.sqrt((d.astype(float) ** 2).mean()):.2f}"
              f" | mean|d| {np.abs(d).mean():.2f} | saturated (|d|=6) {np.mean(np.abs(d) == EPS_PX):.3f}")

    names = list(VARIANTS)
    nc = 1 + 2 * len(names)
    fig, axes = plt.subplots(len(ids) + 1, nc, figsize=(1.55 * nc, 1.7 * (len(ids) + 1)), squeeze=False)
    show_noise = lambda d: np.clip(d / EPS_PX * 127.5 + 127.5, 0, 255).astype(np.uint8)
    for r, i in enumerate(ids):
        axes[r, 0].imshow(trucks[i])
        for j, n in enumerate(names):
            axes[r, 1 + j].imshow(poisoned[n][i])
            axes[r, 1 + len(names) + j].imshow(show_noise(noise[n][i]))
    last = axes[-1]
    last[0].axis("off")
    for j, n in enumerate(names):
        last[1 + j].axis("off")
        m = np.abs(noise[n]).mean((0, 3))                                 # [32, 32]
        im = last[1 + len(names) + j].imshow(m, cmap="magma", vmin=0, vmax=EPS_PX)
    last[1].text(0, .5, "mean |δ| over\n1,000 test trucks →", transform=last[1].transAxes,
                 fontsize=9, va="center")

    axes[0, 0].set_title("clean", fontsize=9)
    for j, n in enumerate(names):
        axes[0, 1 + j].set_title(f"{LABEL[n]}\npoisoned", fontsize=8)
        axes[0, 1 + len(names) + j].set_title(f"{LABEL[n]}\nnoise δ", fontsize=8)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    cax = fig.add_axes([0.92, 0.04, 0.01, 0.14])
    fig.colorbar(im, cax=cax).set_label("|δ| (/255)", fontsize=8)
    fig.suptitle("Noise learned by each cGAN strategy · held-out test trucks · "
                 "noise shown at fixed scale (gray = 0, saturated at ±6/255)", fontsize=10, fontweight="bold")
    fig.subplots_adjust(left=.01, right=.9, top=.9, bottom=.02, wspace=.06, hspace=.08)
    out = Path("experiments/cgan_noise"); out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"noise_comparison.{ext}", dpi=200, bbox_inches="tight")
    print(f"saved {out}/noise_comparison.png")


if __name__ == "__main__":
    main()
