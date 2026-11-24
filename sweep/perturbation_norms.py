#!/usr/bin/env python
"""
L2 and L1 norms of the deployed perturbation, for the three training trucks
used in the recap, for every adaptfix seed (ViT generator, epsilon 4/255).

Reuses collect_panels() from make_figures.py, so the numbers match the
trigger_examples.json files (for seed 0, truck 3878 should give L2 = 115.48).

Usage (from the repo root, on the machine with torch):
    python sweep/perturbation_norms.py
    python sweep/perturbation_norms.py --seeds 0 1 2 3 4 5 --tag adaptfix

Writes out/sweep/perturbation_norms_trucks.csv.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'sweep'))

import make_figures as mf  # noqa: E402

TRUCKS = [3878, 38569, 48431]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5])
    p.add_argument('--tag', default='adaptfix')
    p.add_argument('--epsilon-dir', default='vit_eps4',
                   help="Trigger folder suffix, e.g. vit_eps4 or vit_eps8.")
    args = p.parse_args(argv)

    data = mf.load_dataset('cifar', train=True)
    rows = []
    for seed in args.seeds:
        name = f'sweep_{args.tag}_s{seed}_trigger_{args.epsilon_dir}'
        gen_path = REPO_ROOT / 'experiments' / name / 'generator.pth'
        if not gen_path.exists():
            print(f"seed {seed}: missing {gen_path}", file=sys.stderr)
            continue

        generator = mf.load_generator(str(gen_path), device='cpu')
        generator.eval()
        panels = mf.collect_panels(generator, data, TRUCKS, device='cpu')

        for panel in panels:
            # diff is the integer difference in 0-255 units, shape (32, 32, 3).
            l1 = float(np.abs(panel['diff'].astype(np.float64)).sum())
            rows.append({
                'seed': seed,
                'index': panel['index'],
                'l2_255': round(panel['l2_255'], 3),
                'l1_255': round(l1, 1),
            })
        print(f"seed {seed}: done")

    if not rows:
        print("no generators found", file=sys.stderr)
        return 1

    out_csv = REPO_ROOT / 'out' / 'sweep' / 'perturbation_norms_trucks.csv'
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['seed', 'index', 'l2_255', 'l1_255'])
        writer.writeheader()
        writer.writerows(rows)

    # Printed table: one row per seed, one column per truck (L2).
    print("\nL2 norm (/255 units)")
    print("seed  " + "  ".join(f"{t:>9}" for t in TRUCKS) + "     mean")
    for seed in args.seeds:
        vals = [r['l2_255'] for r in rows if r['seed'] == seed]
        if not vals:
            continue
        print(f"{seed:>4}  " + "  ".join(f"{r['l2_255']:>9.2f}"
              for r in rows if r['seed'] == seed)
              + f"  {np.mean(vals):>7.2f}")
    print(f"\nWrote {out_csv}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
