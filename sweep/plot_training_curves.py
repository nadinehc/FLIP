#!/usr/bin/env python
"""
Training curves per seed (one colour per seed) with a mean +/- 1 std band,
for 1xs, vit_eps4 and vit_eps8. Everything is read from the step logs
(tqdm postfix), so nothing needs the GPU.

  victim_curves.png    clean test acc (CTA) and poisoned test acc (PTA) per epoch
  expert_curves.png    same for the experts (expert 0 of each seed)
  attack_curves.png    label-optimisation loss (g_loss), binned over its epochs
  trigger_curves.png   ViT trigger training: backdoor loss and cross-image cosine
  curves.csv           every plotted value (the table view of the figures)

Usage (repo root):
    python sweep/plot_training_curves.py --tags adapt_s0 adapt_s1 adapt_s2
"""
import argparse
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent / 'out' / 'sweep'
TAGS = []
TRIGGERS = ['1xs', 'vit_eps4', 'vit_eps8']

# Categorical slots 1-3 of the reference palette (validated all-pairs).
SEED_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
SEED_MARKERS = ['o', 's', '^', 'D', 'v', 'P']   # second encoding: some slots are < 3:1
UNFIXED = set()   # tags run with the seeding bug; drawn dashed (set in main)
SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'

PB = re.compile(r"(\d+)/(\d+) \[.*?acc=([\d.]+), acc0=([\d.]+), "
                r"acc1=([\d.]+), loss=([\d.e+-]+)")
GL = re.compile(r"(\d+)/(\d+) \[.*?g_loss=([\d.]+)")
TR = re.compile(r"epoch (\d+):.*?loss_bd=([\d.]+).*?delta_cos=([\d.]+)")


def read(path):
    return Path(path).read_text().replace('\r', '\n')


def epoch_metrics(path, n_expected):
    """One (acc, CTA, PTA, loss) per epoch: the postfix only changes at epoch
    boundaries, so each distinct tuple is one epoch."""
    rows, last = [], None
    for m in PB.finditer(read(path)):
        key = m.groups()[2:]
        if key != last:
            rows.append([float(x) for x in m.groups()[2:]])
            last = key
    if len(rows) != n_expected:
        raise SystemExit(f"{path}: found {len(rows)} epochs, expected "
                         f"{n_expected}")
    return np.array(rows)          # columns: acc, cta, pta, loss


def attack_loss(path, bins=150):
    last = {}
    total = 1
    for m in GL.finditer(read(path)):
        last[int(m.group(1))] = float(m.group(3))   # last refresh wins
        total = int(m.group(2))
    n = np.array(sorted(last))
    y = np.array([last[k] for k in n])
    edges = np.linspace(0, total, bins + 1)
    idx = np.clip(np.digitize(n, edges) - 1, 0, bins - 1)
    out = np.array([y[idx == b].mean() if (idx == b).any() else np.nan
                    for b in range(bins)])
    return out


def trigger_curves(path):
    rows = [(int(a), float(b), float(c)) for a, b, c in TR.findall(read(path))]
    return np.array([[r[1], r[2]] for r in rows])  # loss_bd, delta_cos


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=3)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def panel(ax, x, ys, labels, ylim=None):
    """ys: list of 1-D arrays (one per seed) on the same x."""
    Y = np.vstack(ys)
    mean, std = Y.mean(0), Y.std(0, ddof=1)
    ax.fill_between(x, mean - std, mean + std, color=INK, alpha=0.12,
                    linewidth=0, zorder=1)
    for i, y in enumerate(ys):
        ax.plot(x, y, color=SEED_COLORS[i], lw=1.4, zorder=2 + i,
                linestyle='--' if TAGS[i] in UNFIXED else '-',
                marker=SEED_MARKERS[i], markersize=3.8,
                markevery=max(len(x) // 8, 1), markeredgecolor=SURFACE,
                markeredgewidth=0.6, label=labels[i])
    ax.plot(x, mean, color=INK, lw=2.0, zorder=6, label='mean ± 1 std')
    if ylim:
        ax.set_ylim(*ylim)
    style(ax)


def grid_figure(rows_spec, data, x_by_kind, labels, title, path, xlabel,
                csv_rows, kind):
    """rows_spec: [(metric_name, column, ylabel, ylim)]; data[trig][seed]->arr"""
    fig, axes = plt.subplots(len(rows_spec), 3, figsize=(11.5, 3.0 *
                             len(rows_spec) + 0.9), sharex=True,
                             facecolor=SURFACE, squeeze=False)
    for r, (mname, col, ylabel, ylim) in enumerate(rows_spec):
        for c, trig in enumerate(TRIGGERS):
            ax = axes[r][c]
            ys = [data[trig][s][:, col] for s in range(len(labels))]
            x = x_by_kind(ys[0])
            panel(ax, x, ys, labels, ylim)
            if r == 0:
                ax.set_title(trig, color=INK, fontsize=11, loc='left')
            if c == 0:
                ax.set_ylabel(ylabel, color=INK2, fontsize=10)
            if r == len(rows_spec) - 1:
                ax.set_xlabel(xlabel, color=INK2, fontsize=10)
            for s, y in enumerate(ys):
                for xi, yi in zip(x, y):
                    csv_rows.append([kind, trig, labels[s], mname, xi, yi])
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=4, frameon=False, fontsize=10,
               labelcolor=INK2, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha='left', y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tags', nargs='+', required=True)
    ap.add_argument('--unfixed', nargs='*', default=[],
                    help="Tags run with the seeding bug, drawn dashed")
    ap.add_argument('--out-dir', default='training_curves',
                    help="Folder under out/sweep/ for the figures and CSV")
    ap.add_argument('--budget', type=int, default=1500)
    ap.add_argument('--experts-epochs', type=int, default=20)
    ap.add_argument('--user-epochs', type=int, default=60)
    args = ap.parse_args()
    tags = args.tags
    labels = [f"seed {t.rsplit('_s', 1)[-1]}" + (" (unfixed)" if t in args.unfixed else "") for t in tags]
    if len(tags) > len(SEED_COLORS):
        raise SystemExit("at most 6 seeds (6 categorical slots)")
    TAGS[:] = tags
    UNFIXED.update(args.unfixed)
    out = ROOT / args.out_dir
    out.mkdir(exist_ok=True)
    csv_rows = []

    # victim
    victim = {t: [epoch_metrics(ROOT / tag / 'logs' /
                                f'user_{t}_m{args.budget}.log',
                                args.user_epochs) for tag in tags]
              for t in TRIGGERS}
    grid_figure([('CTA', 1, 'clean test accuracy, %', (40, 90)),
                 ('PTA', 2, 'poisoned test accuracy (PTA), %', (0, 105))],
                victim, lambda y: np.arange(1, len(y) + 1), labels,
                'Victim training (clean images, flipped labels)',
                out / 'victim_curves.png', 'epoch', csv_rows, 'victim')

    # experts (expert 0 of each seed; the four were identical in these runs)
    experts = {t: [epoch_metrics(ROOT / tag / 'logs' / f'expert_{t}_0.log',
                                 args.experts_epochs) for tag in tags]
               for t in TRIGGERS}
    grid_figure([('clean acc', 1, 'clean test accuracy, %', (40, 95)),
                 ('trigger acc', 2, 'expert accuracy on triggered '
                  'test images, %', (0, 105))],
                experts, lambda y: np.arange(1, len(y) + 1), labels,
                'Expert training (expert 0 of each seed)',
                out / 'expert_curves.png', 'epoch', csv_rows, 'expert')

    # attack
    attack = {t: [attack_loss(ROOT / tag / 'logs' / f'attack_{t}.log')[:, None]
                  for tag in tags] for t in TRIGGERS}
    nb = attack[TRIGGERS[0]][0].shape[0]
    grid_figure([('g_loss', 0, 'label-optimisation loss', (0, 1.5))],
                attack, lambda y: (np.arange(nb) + 0.5) / nb * 15, labels,
                'Attack (label optimisation), mean loss per 0.1 epoch',
                out / 'attack_curves.png', 'attack epoch', csv_rows, 'attack')

    # trigger training (no trigger is trained for 1xs)
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.2), sharex=True,
                             facecolor=SURFACE)
    for c, trig in enumerate(['vit_eps4', 'vit_eps8']):
        tc = [trigger_curves(ROOT / tag / 'logs' / f'trigger_{trig}.log')
              for tag in tags]
        for r, (col, name, yl, ylim) in enumerate(
                [(0, 'backdoor loss', 'backdoor loss', None),
                 (1, 'cosine', 'cross-image cosine', (0, 1))]):
            ax = axes[r][c]
            ys = [t[:, col] for t in tc]
            x = np.arange(1, len(ys[0]) + 1)
            panel(ax, x, ys, labels, ylim)
            if r == 0:
                ax.set_title(trig, color=INK, fontsize=11, loc='left')
            if c == 0:
                ax.set_ylabel(yl, color=INK2, fontsize=9.5)
            if r == 1:
                ax.set_xlabel('generator epoch (after warmup)', color=INK2,
                              fontsize=10)
            for s, y in enumerate(ys):
                for xi, yi in zip(x, y):
                    csv_rows.append(['trigger', trig, labels[s], name, xi, yi])
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=4, frameon=False, fontsize=10,
               labelcolor=INK2, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle('Trigger training (ViT generator; 1xs has none)', color=INK,
                 fontsize=12, x=0.01, ha='left', y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    fig.savefig(out / 'trigger_curves.png', dpi=150, facecolor=SURFACE,
                bbox_inches='tight')
    plt.close(fig)

    with open(out / 'curves.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['stage', 'trigger', 'seed', 'metric', 'x', 'value'])
        w.writerows(csv_rows)
    print(f"wrote {len(csv_rows)} points and 4 figures to {out}")


if __name__ == '__main__':
    main()
