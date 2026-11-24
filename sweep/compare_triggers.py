#!/usr/bin/env python
"""
One figure per truck image, comparing triggers on the SAME image.

Layout of each figure (rows = seeds, columns = clean image, then for each
epsilon the triggered image and the amplified perturbation):

    clean | eps4 triggered | eps4 perturbation | eps8 triggered | eps8 perturbation

The images are chosen exactly as make_figures.py does (RandomState(pick-seed)
over the truck candidates), so every generator sees the same images.
A missing generator leaves its row blank and prints a skip line.

Usage (repo root, on a machine with torch and the dataset):
    python sweep/compare_triggers.py --seeds 0 1 2 3 4 5 --epsilons 4 8 \
        --n 3 --out-dir out/figures/compare
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'modules'))

import torch  # noqa: E402
from modules.base_utils.datasets import load_dataset  # noqa: E402
from modules.base_utils.image_noise import (  # noqa: E402
    flip_candidate_mask, load_generator)

PREFIX = 'adaptfix'
CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


def triggered(gen, clean):
    """Returns (triggered uint8 image, amplified perturbation uint8 image)."""
    x = torch.from_numpy(clean.astype(np.float32) / 255.0)
    x = x.permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        delta = gen.delta_pixel(x, quantize=True)
        dirty = torch.clamp(x + delta, 0, 1)
    dirty = np.uint8(np.round(
        dirty.squeeze(0).permute(1, 2, 0).numpy() * 255))
    diff = dirty.astype(np.int16) - clean.astype(np.int16)
    amp = np.clip(128 + diff * (127 / max(np.abs(diff).max(), 1)),
                  0, 255).astype(np.uint8)
    return dirty, amp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', required=True)
    ap.add_argument('--epsilons', type=int, nargs='+', default=[4, 8])
    ap.add_argument('--n', type=int, default=3,
                    help="How many truck images (one figure each).")
    ap.add_argument('--pick-seed', type=int, default=0,
                    help="Seed for choosing the images (as in make_figures).")
    ap.add_argument('--source-label', type=int, default=9)
    ap.add_argument('--target-label', type=int, default=4)
    ap.add_argument('--out-dir', default='out/figures/compare')
    args = ap.parse_args()

    data = load_dataset('cifar', train=True)
    labels = np.array([y for _, y in data])
    candidates = np.where(flip_candidate_mask(
        labels, args.source_label, args.target_label))[0]
    rng = np.random.RandomState(args.pick_seed)
    chosen = rng.choice(candidates, size=args.n, replace=False)

    # generators[(seed, eps)] = loaded generator, or None if missing
    gens = {}
    for s in args.seeds:
        for e in args.epsilons:
            p = ROOT / 'experiments' / \
                f'sweep_{PREFIX}_s{s}_trigger_vit_eps{e}' / 'generator.pth'
            if p.exists():
                g = load_generator(str(p), device='cpu')
                g.eval()
                gens[(s, e)] = g
            else:
                print(f"skip seed {s} eps {e}: {p} not found")
                gens[(s, e)] = None

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    n_cols = 1 + 2 * len(args.epsilons)

    for idx in chosen:
        image, y = data[int(idx)]
        clean = np.asarray(image, dtype=np.uint8)
        fig, axes = plt.subplots(len(args.seeds), n_cols,
                                 figsize=(2.3 * n_cols, 2.4 * len(args.seeds)),
                                 squeeze=False, facecolor='#fcfcfb')
        for r, s in enumerate(args.seeds):
            for c in range(n_cols):
                axes[r][c].set_xticks([])
                axes[r][c].set_yticks([])
                for sp in axes[r][c].spines.values():
                    sp.set_visible(False)
            axes[r][0].imshow(clean)
            axes[r][0].set_ylabel(f"seed {s}", fontsize=10)
            for k, e in enumerate(args.epsilons):
                g = gens[(s, e)]
                tc, pc = 1 + 2 * k, 2 + 2 * k
                if g is None:
                    for c in (tc, pc):
                        axes[r][c].text(0.5, 0.5, 'missing', ha='center',
                                        va='center', fontsize=9,
                                        transform=axes[r][c].transAxes)
                    continue
                dirty, amp = triggered(g, clean)
                axes[r][tc].imshow(dirty)
                axes[r][pc].imshow(amp)
                if r == 0:
                    axes[r][tc].set_title(f"ε={e}/255 triggered", fontsize=9)
                    axes[r][pc].set_title(f"ε={e}/255 perturbation (amp.)",
                                          fontsize=9)
        axes[0][0].set_title("clean", fontsize=9)
        fig.suptitle(f"Training image {int(idx)}: {CLASSES[int(y)]} "
                     f"(class {int(y)}). Target: deer (class 4).", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out = out_dir / f"compare_img{int(idx)}.png"
        fig.savefig(out, dpi=130, facecolor='#fcfcfb')
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == '__main__':
    main()
