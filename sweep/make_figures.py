#!/usr/bin/env python
"""
Renders the visual check on a trigger generator: for random source-class
images, the original, the triggered image, and the perturbation itself.

The point is to be able to say whether the trigger is visible. So the figure is
built to make that judgement honest rather than flattering:

  - The original and triggered columns are shown at the same scale, with no
    enhancement. If you can tell them apart, the trigger is visible.
  - The perturbation is shown twice: once at true scale (almost always a flat
    grey rectangle, which is the point) and once normalized to its own range,
    where the structure is visible. The amplification factor is printed, so an
    amplified panel can never be mistaken for what the victim receives.
  - Everything is computed from the *quantized, clamped* perturbation, i.e.
    exactly what gets written into the uint8 image a victim would train on.

Usage:
    python sweep/make_figures.py --generator out/.../generator.pth
    python sweep/make_figures.py --tag trig              # every trigger in a sweep
    python sweep/make_figures.py --generator G --n 8 --out figures/
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from modules.base_utils.datasets import load_dataset  # noqa: E402
from modules.base_utils.image_noise import (  # noqa: E402
    flip_candidate_mask, load_generator)


def collect_panels(generator, dataset, indices, device='cpu'):
    '''Builds the per-image panel data from the deployed perturbation.'''
    panels = []
    for i in indices:
        image, label = dataset[int(i)]
        clean = np.asarray(image, dtype=np.uint8)

        x_pix = torch.from_numpy(clean.astype(np.float32) / 255.0)
        x_pix = x_pix.permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            # quantize=True: the perturbation as it survives to disk.
            delta = generator.delta_pixel(x_pix, quantize=True)
            dirty = torch.clamp(x_pix + delta, 0.0, 1.0)

        dirty = np.uint8(np.round(
            dirty.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255))

        # Integer arithmetic: float32 subtraction of two n/255 values lands
        # off the 8-bit grid and would misreport the realized bound.
        diff = dirty.astype(np.int16) - clean.astype(np.int16)

        panels.append({
            'index': int(i),
            'label': int(label),
            'clean': clean,
            'dirty': dirty,
            'diff': diff,
            'linf_255': int(np.abs(diff).max()),
            'l2_255': float(np.sqrt((diff.astype(np.float64) ** 2).sum())),
            'changed_frac': float((diff != 0).mean()),
        })
    return panels


def render(panels, out_path, epsilon_255, title):
    '''Writes the figure. Returns the path, or None if matplotlib is absent.'''
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    n = len(panels)
    fig, axes = plt.subplots(n, 4, figsize=(9, 2.3 * n), squeeze=False)

    for r, p in enumerate(panels):
        diff = p['diff']
        amp = max(int(np.abs(diff).max()), 1)

        # True-scale perturbation, drawn on the same 0-255 footing as the
        # images: mid-grey plus the signed difference.
        true_scale = np.uint8(np.clip(128 + diff, 0, 255))
        # Self-normalized, so the structure is readable.
        amplified = np.uint8(np.clip(128 + diff * (127.0 / amp), 0, 255))

        for c, (img, label) in enumerate([
            (p['clean'], f"original (class {p['label']})"),
            (p['dirty'], "triggered"),
            (true_scale, "perturbation, true scale"),
            (amplified, f"perturbation, x{127.0 / amp:.0f}"),
        ]):
            ax = axes[r][c]
            ax.imshow(img, interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(label, fontsize=9)
        axes[r][0].set_ylabel(
            f"Linf {p['linf_255']}/255\n{100 * p['changed_frac']:.0f}% px",
            fontsize=7)

    fig.suptitle(f"{title}  --  epsilon = {epsilon_255:.0f}/255", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def write_pngs(panels, out_dir):
    '''Fallback and supplement: the raw panels as individual PNGs.'''
    try:
        from PIL import Image
    except ImportError:
        return []

    written = []
    for p in panels:
        amp = max(int(np.abs(p['diff']).max()), 1)
        for name, arr in [
            ('clean', p['clean']),
            ('triggered', p['dirty']),
            ('delta_true_scale', np.uint8(np.clip(128 + p['diff'], 0, 255))),
            ('delta_amplified',
             np.uint8(np.clip(128 + p['diff'] * (127.0 / amp), 0, 255))),
        ]:
            path = out_dir / f"img{p['index']}_{name}.png"
            Image.fromarray(arr).save(path)
            written.append(path)
    return written


def mean_pairwise_cosine(panels):
    '''Cosine similarity between the shown perturbations, averaged over pairs.'''
    if len(panels) < 2:
        return None
    flat = np.stack([p['diff'].astype(np.float64).ravel() for p in panels])
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    unit = flat / np.maximum(norms, 1e-12)
    gram = unit @ unit.T
    n = len(panels)
    return float((gram.sum() - np.trace(gram)) / (n * (n - 1)))


def summarize(panels, epsilon_255):
    linf = [p['linf_255'] for p in panels]
    return {
        'pairwise_cosine': mean_pairwise_cosine(panels),
        'epsilon_255': epsilon_255,
        'n_images': len(panels),
        'realized_linf_255_max': int(max(linf)),
        'realized_linf_255_mean': float(np.mean(linf)),
        'bound_respected': bool(max(linf) <= epsilon_255 + 1e-9),
        'l2_255_mean': float(np.mean([p['l2_255'] for p in panels])),
        'pixels_changed_mean': float(
            np.mean([p['changed_frac'] for p in panels])),
        'per_image': [
            {k: p[k] for k in
             ['index', 'label', 'linf_255', 'l2_255', 'changed_frac']}
            for p in panels
        ],
    }


def run_one(generator_pth, out_dir, n, seed, dataset_flag, source_label,
            target_label, device='cpu'):
    generator = load_generator(generator_pth, device=device)
    generator.eval()
    epsilon_255 = generator.epsilon * 255

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_dataset(dataset_flag, train=True)
    labels = np.array([y for _, y in data])
    candidates = np.where(
        flip_candidate_mask(labels, source_label, target_label))[0]
    if len(candidates) == 0:
        raise ValueError(
            f"No candidate images for source_label={source_label}.")

    rng = np.random.RandomState(seed)
    chosen = rng.choice(candidates, size=min(n, len(candidates)),
                        replace=False)

    panels = collect_panels(generator, data, chosen, device=device)
    stats = summarize(panels, epsilon_255)

    arch = json.load(open(
        str(generator_pth).rsplit('.', 1)[0] + '.json')).get('arch', 'conv')
    title = f"{Path(generator_pth).parent.name}  ({arch})"

    fig = render(panels, out_dir / 'trigger_examples.png', epsilon_255, title)
    pngs = write_pngs(panels, out_dir)

    with open(out_dir / 'trigger_examples.json', 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"  arch={arch}  epsilon={epsilon_255:.0f}/255  "
          f"realized Linf max={stats['realized_linf_255_max']}/255  "
          f"bound_respected={stats['bound_respected']}")
    print(f"  mean pixels changed: "
          f"{100 * stats['pixels_changed_mean']:.0f}%")
    if stats['pairwise_cosine'] is not None:
        print(f"  cross-image cosine of the perturbation: "
              f"{stats['pairwise_cosine']:.3f}  (1 = identical on every "
              f"image)")
    if fig:
        print(f"  wrote {fig}")
    else:
        print("  matplotlib not available; wrote individual PNGs instead")
    if pngs:
        print(f"  wrote {len(pngs)} individual PNGs in {out_dir}")
    print(f"  wrote {out_dir / 'trigger_examples.json'}")
    return stats


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Visual check on a trigger generator.")
    p.add_argument('--generator', help="Path to a generator.pth.")
    p.add_argument('--tag', help="Render every trigger of this sweep tag.")
    p.add_argument('--out', default=None,
                   help="Output directory. Defaults next to the generator, "
                        "or out/sweep/<tag>/figures/<trigger>/.")
    p.add_argument('--n', type=int, default=6, help="Images to show.")
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--dataset', default='cifar')
    p.add_argument('--source-label', type=int, default=9)
    p.add_argument('--target-label', type=int, default=4)
    p.add_argument('--device', default='cpu',
                   help="cpu is fine; this is a handful of forward passes.")
    args = p.parse_args(argv)

    if not args.generator and not args.tag:
        p.error("pass --generator or --tag")

    targets = []
    if args.generator:
        out = args.out or str(Path(args.generator).parent / 'figures')
        targets.append((args.generator, out))
    else:
        pattern = f"sweep_{args.tag}_trigger_*"
        found = sorted((REPO_ROOT / 'experiments').glob(pattern))
        if not found:
            print(f"no trigger directories matching "
                  f"experiments/{pattern}", file=sys.stderr)
            return 1
        for d in found:
            gen = d / 'generator.pth'
            if not gen.exists():
                print(f"skip {d.name}: no generator.pth")
                continue
            name = d.name[len(f"sweep_{args.tag}_trigger_"):]
            out = (args.out or
                   str(REPO_ROOT / 'out' / 'sweep' / args.tag / 'figures'))
            targets.append((str(gen), os.path.join(out, name)))

    if not targets:
        print("nothing to render", file=sys.stderr)
        return 1

    failures = 0
    for gen, out in targets:
        print(f"{gen}")
        try:
            run_one(gen, out, args.n, args.seed, args.dataset,
                    args.source_label, args.target_label, args.device)
        except Exception as exc:                       # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            failures += 1
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
