#!/usr/bin/env python
"""
Visual summary of the multi-seed results.

  summary_metrics.png   one panel per metric. x = trigger (1xs, eps4, eps8).
                        A thin coloured line per seed, the mean as a black
                        line, and 95% t-confidence intervals as error bars.
  summary_paired.png    paired differences against 1xs (same seed). One panel
                        per defense/metric, mean and 95% CI per trigger, with
                        the per-seed differences as points.

Usage (repo root):
    python sweep/plot_summary.py --tags adaptfix_s0 adaptfix_s1 adaptfix_s2 \
        adaptfix_s3 adaptfix_s4 adaptfix_s5 --out-dir out/sweep/summary_figures
"""
import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent / 'out' / 'sweep'
TRIGGERS = ['1xs', 'vit_eps4', 'vit_eps8']
TRIG_LABEL = {'1xs': '1xs\n(paper)', 'vit_eps4': 'ViT ε=4', 'vit_eps8': 'ViT ε=8'}

# Categorical slots from the validated reference palette, one per seed.
SEED_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
SEED_MARKERS = ['o', 's', '^', 'D', 'v', 'P']
MEAN = '#0b0b0b'
SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'

METRICS = [('cta', 'CTA (%)', True),
           ('pta', 'PTA (%)', True),
           ('kmeans_detect', 'kmeans detection (%)', True),
           ('pca_detect', 'PCA detection (%)', True),
           ('spectre_detect', 'SPECTRE detection (%)', True),
           ('surrogate_asr', 'surrogate ASR', False),
           ('delta_cos', 'cross-image cosine', False)]

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262}


def load(tag):
    with open(ROOT / tag / 'results.csv') as f:
        return {r['trigger']: r for r in csv.DictReader(f)}


def cell(row, key, k):
    v = row.get(key, '')
    return float(v) * k if v != '' else np.nan


def stats(xs):
    xs = np.asarray(xs, dtype=float)
    xs = xs[~np.isnan(xs)]
    if len(xs) < 2:
        m = xs.mean() if len(xs) else np.nan
        return m, m, m
    n = len(xs)
    m = xs.mean()
    s = xs.std(ddof=1)
    half = T975[n - 1] * s / math.sqrt(n)
    return m, m - half, m + half


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=3)
    ax.grid(True, axis='y', color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tags', nargs='+', required=True)
    ap.add_argument('--out-dir', default='out/sweep/summary_figures')
    args = ap.parse_args()
    if len(args.tags) > len(SEED_COLORS):
        raise SystemExit("at most 6 seeds")
    runs = [load(t) for t in args.tags]
    seeds = [t.rsplit('_s', 1)[-1] for t in args.tags]
    out = ROOT.parent.parent / args.out_dir   # relative to the repo root
    out.mkdir(parents=True, exist_ok=True)
    n = len(runs)

    # --- figure 1: metric per trigger, seeds as lines, mean and 95% CI -----
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.2), facecolor=SURFACE)
    axes = axes.ravel()
    x = np.arange(len(TRIGGERS))
    for ax, (key, label, pct) in zip(axes, METRICS):
        k = 100 if pct else 1
        table = np.array([[cell(r[t], key, k) for t in TRIGGERS]
                          for r in runs])
        for i in range(n):
            ax.plot(x, table[i], color=SEED_COLORS[i], lw=1.0, alpha=0.7,
                    zorder=2, marker=SEED_MARKERS[i], markersize=4.5,
                    markeredgecolor=SURFACE, markeredgewidth=0.5,
                    label=f"seed {seeds[i]}" if ax is axes[0] else None)
        means, lo, hi = [], [], []
        for j in range(len(TRIGGERS)):
            m, a, b = stats(table[:, j])
            means.append(m)
            lo.append(m - a)
            hi.append(b - m)
        ax.errorbar(x, means, yerr=[lo, hi], color=MEAN, lw=2.2, capsize=6,
                    capthick=1.6, zorder=5, marker='o', markersize=7,
                    markerfacecolor=MEAN, markeredgecolor=SURFACE,
                    label='mean ± 95% CI' if ax is axes[0] else None)
        ax.set_xticks(x)
        ax.set_xticklabels([TRIG_LABEL[t] for t in TRIGGERS], fontsize=9)
        ax.set_title(label, fontsize=11, color=INK, loc='left')
        style(ax)
        if key == 'pta':
            ax.set_ylim(0, 105)
        if key == 'cta':
            ax.set_ylim(70, 95)
        if key in ('kmeans_detect', 'pca_detect', 'spectre_detect'):
            ax.set_ylim(-5, 115)
        if key == 'surrogate_asr':
            ax.set_ylim(0, 1)
        if key == 'delta_cos':
            ax.set_ylim(0, 1)
    axes[-1].axis('off')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=len(seeds) + 1, frameon=False,
               fontsize=9, labelcolor=INK2, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("Six fixed seeds: each line is one seed, black is the mean "
                 "with 95% confidence interval", fontsize=12, color=INK,
                 x=0.01, ha='left', y=1.03)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / 'summary_metrics.png', dpi=150, facecolor=SURFACE,
                bbox_inches='tight')
    plt.close(fig)

    # --- figure 2: paired difference against 1xs ---------------------------
    diffs_metrics = [('cta', 'CTA'), ('pta', 'PTA'),
                     ('kmeans_detect', 'kmeans det'),
                     ('pca_detect', 'PCA det'),
                     ('spectre_detect', 'SPECTRE det')]
    fig, axes = plt.subplots(1, 5, figsize=(16, 4.6), facecolor=SURFACE,
                             sharey=False)
    for ax, (key, label) in zip(axes, diffs_metrics):
        for j, trig in enumerate(['vit_eps4', 'vit_eps8']):
            d = np.array([(cell(r[trig], key, 100) - cell(r['1xs'], key, 100))
                          for r in runs])
            pos = j
            jitter = np.linspace(-0.12, 0.12, n)
            for i in range(n):
                ax.scatter(pos + jitter[i], d[i], s=28,
                           color=SEED_COLORS[i], marker=SEED_MARKERS[i],
                           edgecolors=SURFACE, linewidths=0.5, zorder=3)
            m, a, b = stats(d)
            ax.errorbar(pos, m, yerr=[[m - a], [b - m]], color=MEAN, lw=2,
                        capsize=6, zorder=4, marker='o', markersize=7,
                        markerfacecolor=MEAN, markeredgecolor=SURFACE)
        ax.axhline(0, color=INK2, lw=0.8, ls=':')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['ViT ε=4', 'ViT ε=8'], fontsize=9)
        ax.set_xlim(-0.5, 1.5)
        ax.set_title(f"{label} − 1xs (points)", fontsize=10, color=INK,
                     loc='left')
        style(ax)
    fig.suptitle("Paired difference against 1xs, same seed. Dots: seeds. "
                 "Black: mean with 95% CI. Below zero = lower detection.",
                 fontsize=11, color=INK, x=0.01, ha='left', y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / 'summary_paired.png', dpi=150, facecolor=SURFACE,
                bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {out / 'summary_metrics.png'} and summary_paired.png")


if __name__ == '__main__':
    main()
