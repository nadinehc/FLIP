#!/usr/bin/env python
"""
compare_spectre.py — run and/or aggregate SPECTRE evals across triggers.

Examples
--------
# Evaluate several already-trained user models, then compile a table:
python compare_spectre.py \\
  --eval flip_c10_1xu:150:0:1xu \\
  --eval bilevel_c10:150:0:1xs \\
  --eval bilevel_c10:300:0:1xs \\
  --out experiments/spectre_compare

# Spec: <exp>:<budget>:<run>:<poisoner>[:<delta.npy>]
# model  -> experiments/<exp>/user_<budget>/<run>/model.pth
# labels -> experiments/<exp>/{budget,true}.npy
# For 1xu, delta defaults to experiments/<exp>/delta.npy

# Compile existing metrics only (no GPU / no re-eval):
python compare_spectre.py \\
  --aggregate experiments/example_spectre_eval \\
  --aggregate experiments/spectre_compare \\
  --out experiments/spectre_compare

Outputs under --out:
  comparison.csv   flat table for spreadsheets
  COMPARISON.md    side-by-side markdown
  <tag>/           per-eval artefacts (when --eval is used)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import toml

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "modules"))
sys.path.insert(0, str(ROOT))


def _pct(x):
    if x == "" or x is None:
        return ""
    try:
        return f"{100 * float(x):.1f}%"
    except (TypeError, ValueError):
        return str(x)


def parse_eval_spec(spec: str):
    """<exp>:<budget>:<run>:<poisoner>[:<universal_trigger_path>]"""
    parts = spec.split(":")
    if len(parts) < 4:
        raise SystemExit(
            f"Bad --eval spec '{spec}'. Expected "
            "<exp>:<budget>:<run>:<poisoner>[:<delta.npy>]"
        )
    exp, budget, run, poisoner = parts[0], int(parts[1]), int(parts[2]), parts[3]
    trigger = parts[4] if len(parts) > 4 else None
    if poisoner == "1xu" and trigger is None:
        candidate = ROOT / "experiments" / exp / "delta.npy"
        if not candidate.exists():
            raise SystemExit(
                f"poisoner 1xu for '{exp}' needs delta.npy "
                f"(pass as 5th field or place at {candidate})"
            )
        trigger = str(candidate.relative_to(ROOT))
    return {
        "exp": exp,
        "budget": budget,
        "run": run,
        "poisoner": poisoner,
        "trigger": trigger,
    }


def relpath(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def run_one_eval(spec, out: Path):
    from modules.eval_spectre import run_module as eval_spectre

    tag = f"{spec['poisoner']}_b{spec['budget']}_r{spec['run']}"
    art_dir = out / tag
    art_dir.mkdir(parents=True, exist_ok=True)

    model = ROOT / "experiments" / spec["exp"] / f"user_{spec['budget']}" / str(spec["run"]) / "model.pth"
    labels = ROOT / "experiments" / spec["exp"] / f"{spec['budget']}.npy"
    true = ROOT / "experiments" / spec["exp"] / "true.npy"
    for path, label in ((model, "model"), (labels, "labels"), (true, "true labels")):
        if not path.exists():
            raise SystemExit(f"Missing {label} for {spec}: {path}")

    # run_experiment-style config under experiments/_spectre_runs/<out.name>/<tag>
    run_name = f"_spectre_runs/{out.name}/{tag}"
    run_dir = ROOT / "experiments" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    section = {
        "model_path": relpath(model),
        "input_labels": relpath(labels),
        "true_labels": relpath(true),
        "output_dir": relpath(art_dir) + "/",
        "user_model": "r32p",
        "dataset": "cifar",
        "trainer": "sgd",
        "source_label": 9,
        "target_label": 4,
        "poisoner": spec["poisoner"],
        "tag": tag,
        "budget": spec["budget"],
        "retrain": False,
        "backend": "python",
        "filters": ["rcov", "pca", "kmeans"],
    }
    if spec["trigger"]:
        section["universal_trigger_path"] = spec["trigger"]

    with open(run_dir / "config.toml", "w") as f:
        toml.dump({"eval_spectre": section}, f)

    print(f"===== SPECTRE eval {tag} ({spec['poisoner']}, m={spec['budget']}) =====")
    eval_spectre.run(run_name, "eval_spectre")
    return art_dir / "spectre_metrics.json"


def load_metrics(paths):
    rows = []
    for p in paths:
        p = Path(p)
        if not p.is_absolute():
            p = ROOT / p
        if p.is_dir():
            found = sorted(p.rglob("spectre_metrics.json"))
            if not found:
                print(f"warning: no spectre_metrics.json under {p}")
            paths_iter = found
        else:
            paths_iter = [p]
        for mp in paths_iter:
            with open(mp) as f:
                m = json.load(f)
            m["_metrics_path"] = relpath(mp)
            rows.append(m)
    return rows


def flatten_rows(metrics_list):
    flat = []
    for m in metrics_list:
        base = {
            "tag": m.get("tag", ""),
            "poisoner": m.get("poisoner", ""),
            "budget": m.get("budget", ""),
            "eps_target_flips": m.get("eps", ""),
            "n_flips_total": m.get("n_flips_total", ""),
            "pre_cta": (m.get("pre_spectre") or {}).get("cta", ""),
            "pre_pta": (m.get("pre_spectre") or {}).get("pta", ""),
            "post_cta": (m.get("post_spectre") or {}).get("cta", ""),
            "post_pta": (m.get("post_spectre") or {}).get("pta", ""),
            "model_path": m.get("model_path", ""),
            "metrics_path": m.get("_metrics_path", ""),
        }
        filters = m.get("filters") or {}
        if not filters:
            flat.append({
                **base, "filter": "", "poison_recall": "",
                "poison_removed": "", "clean_removed": "", "removed_total": "",
            })
            continue
        for fname, fstats in filters.items():
            flat.append({
                **base,
                "filter": fname,
                "poison_recall": fstats.get("poison_recall", ""),
                "poison_removed": fstats.get("poison_removed", ""),
                "clean_removed": fstats.get("clean_removed", ""),
                "removed_total": fstats.get("removed_total", ""),
            })
    return flat


def write_csv(flat, path: Path):
    if not flat:
        print("no rows to write")
        return
    keys = list(flat[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(flat)


def write_markdown(metrics_list, path: Path):
    lines = [
        "# SPECTRE side-by-side comparison",
        "",
        "Rows are grouped by trigger (`poisoner`) and flip budget. "
        "Poison recall = fraction of flipped-to-target examples removed by the filter.",
        "",
        "## Summary",
        "",
        "| tag | poisoner | budget | eps | pre CTA | pre PTA | "
        "SPECTRE recall | PCA recall | kmeans recall | post CTA | post PTA |",
        "|-----|----------|-------:|----:|--------:|--------:|"
        "---------------:|-----------:|--------------:|---------:|---------:|",
    ]

    for m in sorted(
        metrics_list,
        key=lambda z: (str(z.get("poisoner", "")), z.get("budget") or 0, str(z.get("tag", ""))),
    ):
        f = m.get("filters") or {}
        pre = m.get("pre_spectre") or {}
        post = m.get("post_spectre") or {}
        lines.append(
            "| {tag} | {poisoner} | {budget} | {eps} | {cta} | {pta} | "
            "{rcov} | {pca} | {km} | {pcta} | {ppta} |".format(
                tag=m.get("tag", ""),
                poisoner=m.get("poisoner", ""),
                budget=m.get("budget", ""),
                eps=m.get("eps", ""),
                cta=_pct(pre.get("cta")),
                pta=_pct(pre.get("pta")),
                rcov=_pct((f.get("rcov") or {}).get("poison_recall")),
                pca=_pct((f.get("pca") or {}).get("poison_recall")),
                km=_pct((f.get("kmeans") or {}).get("poison_recall")),
                pcta=_pct(post.get("cta")),
                ppta=_pct(post.get("pta")),
            )
        )

    lines += [
        "",
        "## Per-filter detail",
        "",
        "| tag | poisoner | budget | filter | poison_removed | clean_removed | recall |",
        "|-----|----------|-------:|--------|---------------:|--------------:|-------:|",
    ]
    for m in metrics_list:
        for fname, fstats in (m.get("filters") or {}).items():
            lines.append(
                f"| {m.get('tag', '')} | {m.get('poisoner', '')} | {m.get('budget', '')} | "
                f"{fname} | {fstats.get('poison_removed', '')} | "
                f"{fstats.get('clean_removed', '')} | {_pct(fstats.get('poison_recall'))} |"
            )

    lines += ["", "## Sources", ""]
    for m in metrics_list:
        lines.append(
            f"- `{m.get('tag', '')}`: model `{m.get('model_path', '')}`, "
            f"metrics `{m.get('_metrics_path', '')}`"
        )
    lines.append("")
    path.write_text("\n".join(lines))


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--eval", action="append", default=[],
        help="spec <exp>:<budget>:<run>:<poisoner>[:delta.npy] (repeatable)",
    )
    p.add_argument(
        "--aggregate", action="append", default=[],
        help="path to spectre_metrics.json or a directory to search (repeatable)",
    )
    p.add_argument(
        "--out", default="experiments/spectre_compare",
        help="output directory for comparison.csv / COMPARISON.md and eval runs",
    )
    a = p.parse_args()

    out = Path(a.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    metrics_paths = list(a.aggregate)
    for spec_s in a.eval:
        metrics_paths.append(run_one_eval(parse_eval_spec(spec_s), out))

    rows = load_metrics(metrics_paths)
    if not rows:
        raise SystemExit(
            "No metrics to compare. Pass --eval and/or --aggregate with spectre_metrics.json."
        )

    # Deduplicate by metrics path (same file may appear twice)
    uniq = {}
    for m in rows:
        uniq[m.get("_metrics_path", id(m))] = m
    rows = list(uniq.values())

    write_csv(flatten_rows(rows), out / "comparison.csv")
    write_markdown(rows, out / "COMPARISON.md")
    print(f"Wrote {out / 'comparison.csv'}")
    print(f"Wrote {out / 'COMPARISON.md'}")


if __name__ == "__main__":
    main()
