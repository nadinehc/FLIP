#!/usr/bin/env python
"""Analyse finished FLIP experiments (steganography-v2 matrix).

For each experiment found under experiments/<prefix>*:
  * expert CTA / PTA   -- evaluate each trained expert checkpoint on the clean test set
                          (CTA) and the poisoned source-class test set (PTA); mean +- std
  * flip distribution  -- per budget, the source->target share and top flip categories,
                          from true.npy vs <budget>.npy
  * user CTA / PTA     -- per budget, mean +- std over the user runs (from caccs/paccs.npy)

Everything but the expert evaluation is CPU-only and safe to run while FLIP trains.
Expert evaluation loads models on the GPU; pass --no-experts to skip it, or --device cpu.

    python analyze_flip.py                        # all steg_v2_* experiments
    python analyze_flip.py --no-experts           # skip the GPU expert eval
    python analyze_flip.py --experiments steg_v2_resnet32_ens5
    python analyze_flip.py --csv out/flip_analysis   # also write CSV summaries
"""

import argparse
import csv
import glob
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# --------------------------------------------------------------------------- discovery

def find_experiments(prefix, names):
    if names:
        dirs = [f"experiments/{n}" for n in names]
    else:
        dirs = sorted(glob.glob(f"experiments/{prefix}*"))
    return [d for d in dirs if os.path.isdir(d)]


def parse_name(name):
    """steg_v2_resnet18_ens5 -> (arch, ensemble). Returns (None, None) if it doesn't match."""
    m = re.match(r"steg_v2_(.+)_ens(\d+)$", name)
    return (m.group(1), int(m.group(2))) if m else (None, None)


def final_expert_ckpt(expert_dir):
    """Latest model_<epoch>_<iter>.pth (not *_opt) in an expert dir."""
    best, best_key = None, (-1, -1)
    for p in glob.glob(f"{expert_dir}/model_*.pth"):
        if p.endswith("_opt.pth"):
            continue
        m = re.search(r"model_(\d+)_(\d+)\.pth$", os.path.basename(p))
        if m:
            key = (int(m.group(1)), int(m.group(2)))
            if key > best_key:
                best, best_key = p, key
    return best


# --------------------------------------------------------------------------- expert CTA / PTA

def eval_experts(name, exp_dir, source, target, device):
    """Load each expert's final checkpoint and eval CTA (clean) and PTA (poisoned source)."""
    import torch
    from modules.base_utils.datasets import pick_poisoner, get_matching_datasets
    from modules.base_utils.util import clf_eval
    import modules.base_utils.model.resnet as resnet
    from modules.base_utils.model.model import SequentialImageNetworkMod

    arch, ens = parse_name(name)
    if arch is None:
        print(f"  [experts] cannot parse '{name}', skipping")
        return []
    encoder = f"noise_generator/encoders/{arch}_ens{ens}_k1.pth"
    surrogates = [f"noise_generator/surrogates/{arch}_seed{s}.pth" for s in range(ens)]
    if not os.path.exists(encoder):
        print(f"  [experts] encoder {encoder} missing, skipping")
        return []

    ckpt_root = f"out/checkpoints/{name}"
    expert_dirs = sorted(glob.glob(f"{ckpt_root}/*/"),
                         key=lambda d: int(os.path.basename(d.rstrip("/"))) if os.path.basename(d.rstrip("/")).isdigit() else 1e9)
    expert_dirs = [d for d in expert_dirs if (Path(d) / "DONE").exists()]
    if not expert_dirs:
        print("  [experts] no finished experts yet")
        return []

    # datasets (built once; the poisoner is deterministic)
    poisoner = pick_poisoner("steganography_v2", "cifar", target,
                             encoder_path=encoder, surrogate_paths=surrogates)
    _, _, test_set, poison_test, _ = get_matching_datasets("cifar", poisoner, source)

    rows = []
    for d in expert_dirs:
        e = os.path.basename(d.rstrip("/"))
        ckpt = final_expert_ckpt(d)
        if ckpt is None:
            continue
        model = SequentialImageNetworkMod(resnet.resnet32(10)).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        cta = clf_eval(model, test_set)[0] * 100
        pta = clf_eval(model, poison_test.poison_dataset)[0] * 100
        rows.append((e, cta, pta))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


# --------------------------------------------------------------------------- flip distribution

def flip_distribution(exp_dir, budgets, source, target):
    true_path = Path(exp_dir) / "true.npy"
    if not true_path.exists():
        return None
    true = np.load(true_path).argmax(1)
    out = []
    for b in budgets:
        bp = Path(exp_dir) / f"{b}.npy"
        if not bp.exists():
            continue
        new = np.load(bp).argmax(1)
        flipped = new != true
        n_flips = int(flipped.sum())
        src_tgt = int(((true == source) & (new == target)).sum())
        # top (i -> j) categories among flips
        pairs = {}
        for i, j in zip(true[flipped], new[flipped]):
            pairs[(int(i), int(j))] = pairs.get((int(i), int(j)), 0) + 1
        top = sorted(pairs.items(), key=lambda kv: -kv[1])[:4]
        top_str = ", ".join(f"{CLASSES[i]}->{CLASSES[j]} {c}" for (i, j), c in top)
        out.append(dict(budget=b, n_flips=n_flips, src_tgt=src_tgt,
                        src_tgt_share=(src_tgt / n_flips if n_flips else 0.0), top=top_str))
    return out


# --------------------------------------------------------------------------- user CTA / PTA

def user_metrics(exp_dir, budgets):
    out = []
    for b in budgets:
        ctas, ptas, cta_curves, pta_curves = [], [], [], []
        for run_dir in sorted(glob.glob(f"{exp_dir}/user_{b}/*/")):
            cp, pp = Path(run_dir) / "caccs.npy", Path(run_dir) / "paccs.npy"
            if cp.exists() and pp.exists():
                c, p = np.load(cp), np.load(pp)                    # (epochs, 2): acc, loss
                ctas.append(float(c[-1, 0]) * 100); ptas.append(float(p[-1, 0]) * 100)
                cta_curves.append(c[:, 0] * 100); pta_curves.append(p[:, 0] * 100)
        if ctas:
            sd = lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
            # align run curves to the shortest (in case a run is still training)
            L = min(len(x) for x in pta_curves)
            out.append(dict(budget=b, n_runs=len(ctas), cta_mean=np.mean(ctas), cta_std=sd(ctas),
                            pta_mean=np.mean(ptas), pta_std=sd(ptas),
                            cta_curve=np.stack([x[:L] for x in cta_curves]),
                            pta_curve=np.stack([x[:L] for x in pta_curves])))
    return out


# --------------------------------------------------------------------------- collect

def collect_results(prefix="steg_v2_", experiments=None, budgets=None, source=9, target=4,
                    with_experts=False, device="cpu"):
    """Gather every finished experiment into a list of dicts. Importable from the notebook:

        from analyze_flip import collect_results, plot_user_metric, plot_flip_share
        res = collect_results(with_experts=False)      # CPU-only, safe while FLIP runs
        plot_user_metric(res, "pta"); plot_flip_share(res)

    Each item: {name, experts:[(e,cta,pta)], flips:[{budget,...}], users:[{budget,...}]}.
    """
    dev = None
    if with_experts:
        import torch
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        dev = torch.device(device)

    results = []
    for exp_dir in find_experiments(prefix, experiments):
        name = os.path.basename(exp_dir)
        bs = budgets or discover_budgets(exp_dir)
        results.append(dict(
            name=name,
            experts=eval_experts(name, exp_dir, source, target, dev) if with_experts else [],
            flips=flip_distribution(exp_dir, bs, source, target) or [],
            users=user_metrics(exp_dir, bs) or [],
        ))
    return results


# --------------------------------------------------------------------------- plotting

def _fig(ax=None, figsize=(8, 5)):
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax
    return ax.figure, ax


def _shared_ylim(results, metrics=("pta", "cta"), pad=3.0):
    """Common y-range covering mean +- std of the given metrics across all configs/budgets."""
    lo, hi = np.inf, -np.inf
    for r in results:
        for u in r["users"]:
            for m in metrics:
                if f"{m}_mean" in u:
                    lo = min(lo, u[f"{m}_mean"] - u[f"{m}_std"])
                    hi = max(hi, u[f"{m}_mean"] + u[f"{m}_std"])
    if not np.isfinite(lo):
        return None
    return (max(0, lo - pad), min(100, hi + pad))


def plot_user_metric(results, metric="pta", ax=None, title=None, ylim=None):
    """User CTA or PTA vs budget, one line (with std band) per experiment."""
    fig, ax = _fig(ax)
    for r in results:
        rows = [u for u in r["users"] if f"{metric}_mean" in u]
        if not rows:
            continue
        b = [u["budget"] for u in rows]
        m = np.array([u[f"{metric}_mean"] for u in rows])
        s = np.array([u[f"{metric}_std"] for u in rows])
        line, = ax.plot(b, m, "o-", label=r["name"].replace("steg_v2_", ""))
        ax.fill_between(b, m - s, m + s, alpha=0.15, color=line.get_color())
    ax.set_xlabel("budget (# flips)")
    ax.set_ylabel(f"user {metric.upper()} (%)")
    ax.set_title(title or f"User {metric.upper()} vs budget (mean +- std over runs)")
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    return fig


def plot_flip_share(results, ax=None, title=None):
    """Source->target (truck->deer) share of flips vs budget, one line per experiment."""
    fig, ax = _fig(ax)
    for r in results:
        if not r["flips"]:
            continue
        b = [f["budget"] for f in r["flips"]]
        share = [100 * f["src_tgt_share"] for f in r["flips"]]
        ax.plot(b, share, "s-", label=r["name"].replace("steg_v2_", ""))
    ax.set_xlabel("budget (# flips)")
    ax.set_ylabel("source->target share of flips (%)")
    ax.set_title(title or "Truck->deer share of flips vs budget")
    ax.set_ylim(0, 102); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    return fig


def plot_expert_summary(results, ax=None, title=None):
    """Grouped bars of expert CTA and PTA (mean +- std) per experiment."""
    fig, ax = _fig(ax, figsize=(max(8, 1.4 * len(results)), 5))
    names, cta_m, cta_s, pta_m, pta_s = [], [], [], [], []
    sd = lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    for r in results:
        if not r["experts"]:
            continue
        ctas = [c for _, c, _ in r["experts"]]; ptas = [p for _, _, p in r["experts"]]
        names.append(r["name"].replace("steg_v2_", ""))
        cta_m.append(np.mean(ctas)); cta_s.append(sd(ctas))
        pta_m.append(np.mean(ptas)); pta_s.append(sd(ptas))
    if not names:
        ax.text(0.5, 0.5, "no expert evaluations\n(run with with_experts=True)",
                ha="center", va="center", transform=ax.transAxes)
        return fig
    x = np.arange(len(names)); w = 0.38
    ax.bar(x - w / 2, cta_m, w, yerr=cta_s, capsize=3, label="CTA")
    ax.bar(x + w / 2, pta_m, w, yerr=pta_s, capsize=3, label="PTA")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("accuracy (%)"); ax.set_ylim(0, 105)
    ax.set_title(title or "Expert CTA / PTA (mean +- std over experts)")
    ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)
    return fig


def _pick_budget(results, budget):
    """Default to the largest budget present in any experiment's user results."""
    if budget is not None:
        return budget
    bs = {u["budget"] for r in results for u in r["users"]}
    return max(bs) if bs else None


def plot_metric_curves(results, metric="pta", budget=None, ax=None, title=None):
    """Evolution over epochs (mean +- std over runs) for one budget, one line per config."""
    fig, ax = _fig(ax)
    budget = _pick_budget(results, budget)
    for r in results:
        u = next((u for u in r["users"] if u["budget"] == budget and f"{metric}_curve" in u), None)
        if u is None:
            continue
        curve = u[f"{metric}_curve"]                               # (n_runs, epochs)
        m, s = curve.mean(0), curve.std(0, ddof=1) if curve.shape[0] > 1 else np.zeros(curve.shape[1])
        ep = np.arange(1, len(m) + 1)
        line, = ax.plot(ep, m, label=r["name"].replace("steg_v2_", ""))
        ax.fill_between(ep, m - s, m + s, alpha=0.15, color=line.get_color())
    ax.set_xlabel("epoch")
    ax.set_ylabel(f"user {metric.upper()} (%)")
    ax.set_title(title or f"User {metric.upper()} over epochs (budget {budget}, mean +- std)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    return fig


def plot_cta_vs_pta(results, ax=None, title=None):
    """CTA (y) vs PTA (x): one point per (config, budget), error bars = std over runs."""
    fig, ax = _fig(ax)
    import matplotlib.cm as cm
    names = [r["name"].replace("steg_v2_", "") for r in results if r["users"]]
    colors = cm.tab10(np.linspace(0, 1, max(len(names), 1)))
    ci = 0
    for r in results:
        if not r["users"]:
            continue
        c = colors[ci]; ci += 1
        label = r["name"].replace("steg_v2_", "")
        us = sorted(r["users"], key=lambda u: u["budget"])      # order by budget for the connecting line
        ax.plot([u["pta_mean"] for u in us], [u["cta_mean"] for u in us],
                "-", color=c, alpha=0.6, label=label)
        for u in us:
            ax.errorbar(u["pta_mean"], u["cta_mean"], xerr=u["pta_std"], yerr=u["cta_std"],
                        fmt="o", color=c, capsize=2, alpha=0.85)
            ax.annotate(str(u["budget"]), (u["pta_mean"], u["cta_mean"]),
                        textcoords="offset points", xytext=(4, 3), fontsize=7)
    ax.set_xlabel("user PTA (%)"); ax.set_ylabel("user CTA (%)")
    ax.set_title(title or "CTA vs PTA (point = config x budget)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    return fig


def plot_summary(results, budget_for_curves=None, title="Steganography-v2 FLIP summary"):
    """Every panel arranged in one figure (2x3 grid)."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    yl = _shared_ylim(results)                                  # common scale for PTA and CTA panels
    plot_user_metric(results, "pta", ax=axes[0, 0], ylim=yl)
    plot_user_metric(results, "cta", ax=axes[0, 1], ylim=yl)
    plot_flip_share(results, ax=axes[0, 2])
    plot_metric_curves(results, "pta", budget=budget_for_curves, ax=axes[1, 0])
    plot_metric_curves(results, "cta", budget=budget_for_curves, ax=axes[1, 1])
    plot_expert_summary(results, ax=axes[1, 2])
    fig.suptitle(title, fontsize=15, y=1.0)
    fig.tight_layout()
    return fig


def plot_all(results, out_dir=None, show=True, budget_for_curves=None):
    """Build every figure; save to out_dir if given, show if in a notebook."""
    import matplotlib.pyplot as plt
    yl = _shared_ylim(results)
    figs = {
        "user_pta": plot_user_metric(results, "pta", ylim=yl),
        "user_cta": plot_user_metric(results, "cta", ylim=yl),
        "flip_share": plot_flip_share(results),
        "pta_curves": plot_metric_curves(results, "pta", budget=budget_for_curves),
        "cta_curves": plot_metric_curves(results, "cta", budget=budget_for_curves),
        "expert_summary": plot_expert_summary(results),
        "summary": plot_summary(results, budget_for_curves=budget_for_curves),
    }
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        for k, f in figs.items():
            f.savefig(f"{out_dir}/{k}.png", dpi=120, bbox_inches="tight")
        print(f"figures saved to {out_dir}/")
    if show:
        plt.show()
    return figs


# --------------------------------------------------------------------------- reporting

def discover_budgets(exp_dir):
    bs = []
    for p in glob.glob(f"{exp_dir}/*.npy"):
        m = re.match(r"(\d+)\.npy$", os.path.basename(p))
        if m:
            bs.append(int(m.group(1)))
    return sorted(bs)


def print_report(results):
    sd = lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    for r in results:
        print(f"\n{'='*70}\n### {r['name']}\n{'='*70}")
        if r["experts"]:
            ctas = [c for _, c, _ in r["experts"]]; ptas = [p for _, _, p in r["experts"]]
            print(f"Experts (n={len(r['experts'])}): CTA {np.mean(ctas):5.2f} +- {sd(ctas):.2f} | "
                  f"PTA {np.mean(ptas):5.2f} +- {sd(ptas):.2f}")
            for e, c, pt in r["experts"]:
                print(f"    expert {e}: CTA {c:5.2f} | PTA {pt:5.2f}")
        if r["flips"]:
            print("Flip distribution:")
            print(f"    {'budget':>6} {'flips':>6} {'src->tgt':>9} {'share':>7}   top categories")
            for f in r["flips"]:
                print(f"    {f['budget']:>6} {f['n_flips']:>6} {f['src_tgt']:>9} "
                      f"{f['src_tgt_share']*100:>6.1f}%   {f['top']}")
        if r["users"]:
            print("User models (CTA / PTA, mean +- std over runs):")
            print(f"    {'budget':>6} {'runs':>5} {'CTA':>16} {'PTA':>16}")
            for u in r["users"]:
                print(f"    {u['budget']:>6} {u['n_runs']:>5} "
                      f"{u['cta_mean']:8.2f} +- {u['cta_std']:4.2f} "
                      f"{u['pta_mean']:8.2f} +- {u['pta_std']:4.2f}")
        else:
            print("User models: no finished runs yet")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prefix", default="steg_v2_")
    p.add_argument("--experiments", nargs="+", default=None, help="explicit experiment names")
    p.add_argument("--budgets", nargs="+", type=int, default=None, help="default: whatever is on disk")
    p.add_argument("--source", type=int, default=9, help="source class (truck)")
    p.add_argument("--target", type=int, default=4, help="target class (deer)")
    p.add_argument("--no-experts", action="store_true", help="skip the GPU expert evaluation")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--csv", default=None, help="path prefix for CSV output (writes _experts/_flips/_users.csv)")
    p.add_argument("--plot", default=None, metavar="DIR",
                   help="save user_pta/user_cta/flip_share/expert_summary PNGs to DIR")
    args = p.parse_args()

    if not find_experiments(args.prefix, args.experiments):
        print("No experiments found.")
        return
    if not args.no_experts and args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            print("CUDA not available; running expert eval on CPU.")
            args.device = "cpu"

    # collect once (expert eval, if requested, runs here), then print / csv / plot from it
    results = collect_results(prefix=args.prefix, experiments=args.experiments, budgets=args.budgets,
                              source=args.source, target=args.target,
                              with_experts=not args.no_experts, device=args.device)
    print_report(results)

    expert_rows = [[r["name"], e, round(c, 2), round(p, 2)] for r in results for e, c, p in r["experts"]]
    flip_rows = [[r["name"], f["budget"], f["n_flips"], f["src_tgt"], round(f["src_tgt_share"], 4)]
                 for r in results for f in r["flips"]]
    user_rows = [[r["name"], u["budget"], u["n_runs"], round(u["cta_mean"], 2), round(u["cta_std"], 2),
                  round(u["pta_mean"], 2), round(u["pta_std"], 2)] for r in results for u in r["users"]]

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        _write_csv(f"{args.csv}_experts.csv", ["experiment", "expert", "cta", "pta"], expert_rows)
        _write_csv(f"{args.csv}_flips.csv",
                   ["experiment", "budget", "n_flips", "src_tgt", "src_tgt_share"], flip_rows)
        _write_csv(f"{args.csv}_users.csv",
                   ["experiment", "budget", "n_runs", "cta_mean", "cta_std", "pta_mean", "pta_std"], user_rows)
        print(f"\nCSV written to {args.csv}_(experts|flips|users).csv")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        plot_all(results, out_dir=args.plot, show=False)


def _write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


if __name__ == "__main__":
    main()
