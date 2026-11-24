#!/usr/bin/env python
"""
Aggregates results.csv across seeds: mean, sample std (ddof=1), standard error,
95% CI (t, df = n-1) and the per-seed paired difference against 1xs.

Usage (repo root):
    python sweep/aggregate_seeds.py --tags adapt_s0 adapt_s1 adapt_s2
    python sweep/aggregate_seeds.py --tags adaptfix_s3 adaptfix_s4 adaptfix_s5 --suffix _fixed

Writes out/sweep/seed_summary<suffix>.md and .csv (suffix defaults to '').
"""
import argparse
import csv
import math
from pathlib import Path

# two-sided 97.5% Student-t quantiles by degrees of freedom
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262}

METRICS = [('cta', 'CTA'), ('pta', 'PTA'), ('kmeans_detect', 'kmeans det'),
           ('pca_detect', 'PCA det'), ('spectre_detect', 'SPECTRE det')]
EXTRA = [('surrogate_asr', 'surrogate ASR'), ('delta_cos', 'cross-image cos')]
ROOT = Path(__file__).resolve().parent.parent / 'out' / 'sweep'


def mean(xs):
    return sum(xs) / len(xs)


def std(xs):  # sample standard deviation, n-1 in the denominator
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def load(tag):
    with open(ROOT / tag / 'results.csv') as f:
        return {r['trigger']: r for r in csv.DictReader(f)}


def val(row, key):
    return float(row[key]) if row.get(key, '') != '' else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tags', nargs='+', required=True)
    ap.add_argument('--suffix', default='',
                    help="Output is seed_summary<suffix>.md/.csv")
    args = ap.parse_args()
    runs = [load(t) for t in args.tags]
    n = len(runs)
    t = T975[n - 1]
    triggers = list(runs[0].keys())

    rows, md = [], []
    md.append("| Trigger | Metric | " + " | ".join(args.tags) +
              " | mean | std | SEM | 95% CI |")
    md.append("|---|---|" + "---|" * (n + 4))
    for trig in triggers:
        for key, label in METRICS + EXTRA:
            xs = [val(r[trig], key) for r in runs]
            if any(x is None for x in xs):
                continue
            pct = key not in ('surrogate_asr', 'delta_cos')
            k = 100 if pct else 1
            m, s = mean(xs) * k, std(xs) * k
            sem = s / math.sqrt(n)
            lo, hi = m - t * sem, m + t * sem
            fmt = (lambda v: f"{v:.2f}") if pct else (lambda v: f"{v:.3f}")
            md.append(f"| {trig} | {label} | " +
                      " | ".join(fmt(x * k) for x in xs) +
                      f" | {fmt(m)} | {fmt(s)} | {fmt(sem)} | "
                      f"[{fmt(lo)}, {fmt(hi)}] |")
            rows.append([trig, label, *[x * k for x in xs], m, s, sem, lo, hi])

    md.append("")
    md.append("Paired difference against 1xs, same seed "
              "(positive = higher than 1xs):")
    md.append("")
    md.append("| Trigger | Metric | " + " | ".join(args.tags) +
              " | mean diff | std | 95% CI |")
    md.append("|---|---|" + "---|" * (n + 3))
    for trig in triggers:
        if trig == '1xs':
            continue
        for key, label in METRICS:
            d = [(val(r[trig], key) - val(r['1xs'], key)) * 100 for r in runs]
            m, s = mean(d), std(d)
            half = t * s / math.sqrt(n)
            md.append(f"| {trig} | {label} | " +
                      " | ".join(f"{x:+.2f}" for x in d) +
                      f" | {m:+.2f} | {s:.2f} | "
                      f"[{m - half:+.2f}, {m + half:+.2f}] |")

    out = ROOT / f'seed_summary{args.suffix}.md'
    out.write_text("\n".join(md) + "\n")
    with open(ROOT / f'seed_summary{args.suffix}.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['trigger', 'metric', *args.tags,
                    'mean', 'std', 'sem', 'ci95_lo', 'ci95_hi'])
        w.writerows(rows)
    print("\n".join(md))


if __name__ == '__main__':
    main()
