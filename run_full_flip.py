#!/usr/bin/env python
"""
run_full_flip.py -- end-to-end FLIP experiment (CIFAR-10 / ResNet-32 by default).

Put this file in the root of the FLIP repo and run it with the repo's venv:

    python run_full_flip.py --name flip_c10

Stages (all run by default except trigger/spectre; every stage is resumable):

    trigger  pre-train frozen universal δ when --poisoner 1xu (auto-included)
    experts  train N backdoored expert models and save their trajectories
    labels   trajectory matching -> optimized soft (logit) labels
             (use --spectre-proxy for SPECTRE Option A regularizer)
    flips    soft labels -> hard label flips for every budget
    users    train M user models on each budget's labels
    spectre  full SPECTRE/PCA/kmeans filter eval on user models (opt-in stage)
    summary  collect final CTA / PTA (mean +- std over the user runs)

Outputs (same files the repo's modules always write, plus a summary):

    out/checkpoints/<name>/<expert>/model_<epoch>_<iter>[_opt].pth
    experiments/<name>/{labels,true,losses}.npy and <budget>.npy
    experiments/<name>/user_<budget>/<run>/{caccs,paccs,labels}.npy + model.pth
    experiments/<name>/summary_runs.csv, summary.csv, curves_<budget>.npz
    experiments/<name>/run.log, args.json

Use a new --name whenever you change hyperparameters: finished work is detected
by the files that exist, not by the settings that produced them.
"""

import argparse
import csv
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import toml

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)                                   # the repo uses cwd-relative paths
sys.path.insert(0, str(ROOT / "modules"))        # same as run_experiment.py
sys.path.insert(0, str(ROOT))

STAGES = ["trigger", "experts", "labels", "flips", "users", "spectre", "summary"]
DEFAULT_STAGES = ["experts", "labels", "flips", "users", "summary"]

# Paper setup for the ResNet experiments (SGD, batch 256, Nesterov momentum 0.9)
OPTIM = dict(lr=0.1, momentum=0.9, nesterov=True, weight_decay=2e-4)
BATCH_SIZE = 256
CHECKPOINT_ITERS = 50        # expert weights saved every 50 minibatches

log = logging.getLogger("flip")


# --------------------------------------------------------------------------
# arguments and paths
# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="flip_c10_test2", help="experiment name")
    p.add_argument("--n-experts", type=int, default=50)
    p.add_argument("--n-user-runs", type=int, default=10,
                   help="user models trained per budget")
    p.add_argument("--budgets", type=int, nargs="+",
                   default=[150, 300, 500, 1000, 1500])
    p.add_argument("--expert-epochs", type=int, default=20)
    p.add_argument("--user-epochs", type=int, default=200)
    p.add_argument("--milestones", type=int, nargs="+", default=[75, 150],
                   help="epochs at which the lr is divided by 10 (user and expert)")
    p.add_argument("--traj-max", type=int, default=20,
                   help="trajectory epochs are sampled from 1..traj-max "
                        "(must be <= expert-epochs)")
    p.add_argument("--label-iterations", type=int, default=25,
                   help="passes over the data in trajectory matching")
    p.add_argument("--model", default="r32p")
    p.add_argument("--dataset", default="cifar")
    p.add_argument("--source", type=int, default=9, help="source label (truck)")
    p.add_argument("--target", type=int, default=4, help="target label (deer)")
    p.add_argument("--poisoner", default="1xs",
                   help="trigger (1xs = sinusoidal, 1xu = universal)")
    p.add_argument("--trigger-path", default=None,
                   help="path to delta.npy for 1xu (default: experiments/<name>/delta.npy)")
    p.add_argument("--eps", type=float, default=8 / 255,
                   help="L∞ budget in [0,1] for 1xu trigger learning")
    p.add_argument("--delta-steps", type=int, default=1000,
                   help="Adam steps when learning the 1xu trigger")
    p.add_argument("--delta-lr", type=float, default=0.01,
                   help="Adam lr when learning the 1xu trigger")
    p.add_argument("--surrogate-epochs", type=int, default=20,
                   help="clean surrogate epochs before learning δ")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--spectre-proxy", action="store_true",
                   help="enable SPECTRE Option A proxy loss in generate_labels")
    p.add_argument("--spectre-lambda", type=float, default=0.1,
                   help="weight on SPECTRE proxy term when --spectre-proxy is set")
    p.add_argument("--spectre-budgets", type=int, nargs="+", default=None,
                   help="budgets to evaluate with full SPECTRE (default: all --budgets)")
    p.add_argument("--spectre-retrain", action="store_true",
                   help="after SPECTRE filtering, retrain and report post CTA/PTA")
    p.add_argument("--stages", nargs="+", choices=STAGES, default=None)
    p.add_argument("--force", action="store_true",
                   help="redo work even if its outputs already exist")
    p.add_argument("--skip-checks", action="store_true",
                   help="skip the pre-flight checks (GPU, generate_labels bug)")
    a = p.parse_args()
    if a.traj_max > a.expert_epochs:
        p.error("--traj-max must be <= --expert-epochs")
    if a.stages is None:
        a.stages = (["trigger"] + DEFAULT_STAGES) if a.poisoner == "1xu" else list(DEFAULT_STAGES)
    if a.poisoner == "1xu" and a.trigger_path is None:
        a.trigger_path = f"experiments/{a.name}/delta.npy"
    if a.spectre_budgets is None:
        a.spectre_budgets = list(a.budgets)
    return a


def exp_dir(a):
    return f"experiments/{a.name}/"


def expert_dir(a, e):
    return f"out/checkpoints/{a.name}/{e}/"


def user_dir(a, budget, run):
    return f"experiments/{a.name}/user_{budget}/{run}/"


def trigger_path(a):
    return a.trigger_path or f"experiments/{a.name}/delta.npy"


# --------------------------------------------------------------------------
# config files (the repo's modules read experiments/<name>/config.toml)
# --------------------------------------------------------------------------
def write_config(rel_name, sections):
    path = Path("experiments") / rel_name / "config.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        toml.dump(sections, f)


def write_configs(a):
    sched = dict(milestones=a.milestones, gamma=0.1)
    common = dict(dataset=a.dataset, trainer="sgd", source_label=a.source,
                  target_label=a.target, poisoner=a.poisoner)
    if a.poisoner == "1xu":
        common["universal_trigger_path"] = trigger_path(a)

    sections = {}
    if a.poisoner == "1xu":
        sections["train_universal_trigger"] = dict(
            output_path=trigger_path(a), model=a.model, dataset=a.dataset,
            source_label=a.source, target_label=a.target, trainer="sgd",
            eps=a.eps, delta_steps=a.delta_steps, delta_lr=a.delta_lr,
            surrogate_epochs=a.surrogate_epochs, batch_size=BATCH_SIZE,
            seed=a.seed, optim_kwargs=OPTIM, scheduler_kwargs=sched)

    # "{}" in a path is filled with slurm_id (= expert number / run number)
    ckpt = f"out/checkpoints/{a.name}/{{}}/"
    sections.update({
        "train_expert": dict(
            output_dir=ckpt, model=a.model, batch_size=BATCH_SIZE,
            epochs=a.expert_epochs, checkpoint_iters=CHECKPOINT_ITERS,
            optim_kwargs=OPTIM, scheduler_kwargs=sched, **common),
        "generate_labels": dict(
            input_pths=ckpt + "model_{}_{}.pth",
            opt_pths=ckpt + "model_{}_{}_opt.pth",
            expert_model=a.model, output_dir=exp_dir(a),
            expert_config=dict(experts=a.n_experts, min=0, max=a.traj_max),
            attack_config=dict(
                iterations=a.label_iterations,
                spectre_proxy=bool(a.spectre_proxy),
                spectre_lambda=a.spectre_lambda,
            ), **common),
        "select_flips": dict(
            budgets=a.budgets,
            input_label_glob=exp_dir(a) + "labels.npy",
            true_labels=exp_dir(a) + "true.npy",
            output_dir=exp_dir(a)),
    })
    write_config(a.name, sections)

    for b in a.budgets:   # one config per budget; "{}" = user run number
        write_config(f"{a.name}/user_{b}", {
            "train_user": dict(
                input_labels=exp_dir(a) + f"{b}.npy",
                output_dir=exp_dir(a) + f"user_{b}/{{}}/",
                user_model=a.model, soft=False,   # train on the hard flipped labels
                alpha=0,                          # must be set: the code crashes without it
                batch_size=BATCH_SIZE, epochs=a.user_epochs,
                optim_kwargs=OPTIM, scheduler_kwargs=sched, **common)})

    # SPECTRE eval configs (one per budget/run); written always so stage can run alone
    for b in a.spectre_budgets:
        for r in range(a.n_user_runs):
            write_config(f"{a.name}/spectre_{b}_{r}", {
                "eval_spectre": dict(
                    model_path=user_dir(a, b, r) + "model.pth",
                    input_labels=exp_dir(a) + f"{b}.npy",
                    true_labels=exp_dir(a) + "true.npy",
                    output_dir=exp_dir(a) + f"spectre_{b}/{r}/",
                    user_model=a.model,
                    retrain=bool(a.spectre_retrain),
                    batch_size=BATCH_SIZE,
                    epochs=a.user_epochs,
                    optim_kwargs=OPTIM,
                    scheduler_kwargs=sched,
                    **common)})


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def preflight(a):
    # 1) the n_classes bug in generate_labels (used before it is defined)
    if "labels" in a.stages:
        lines = (ROOT / "modules/generate_labels/run_module.py").read_text().splitlines()
        is_def = lambda l: re.match(r"\s*n_classes\s*=", l) is not None
        def_line = next((i for i, l in enumerate(lines) if is_def(l)), None)
        use_line = next((i for i, l in enumerate(lines)
                         if re.search(r"\bn_classes\b", l) and not is_def(l)), None)
        if def_line is not None and use_line is not None and use_line < def_line:
            raise SystemExit(
                "modules/generate_labels/run_module.py uses n_classes (line "
                f"{use_line + 1}) before defining it (line {def_line + 1}).\n"
                "Move the line  n_classes = get_n_classes(dataset_flag)  above its first "
                "use, or pass --skip-checks to ignore this check.")
    # 2) the repo's load_model() calls .cuda()
    if any(s in a.stages for s in ("trigger", "experts", "labels", "users")):
        import torch
        if not torch.cuda.is_available():
            raise SystemExit("No GPU visible (the repo moves models to CUDA). "
                             "Run this on a GPU node.")


def timed(msg, fn, *args, **kwargs):
    log.info("START %s", msg)
    t0 = time.time()
    try:
        out = fn(*args, **kwargs)
    except Exception:
        log.exception("FAILED %s", msg)
        raise
    log.info("DONE  %s (%.1f min)", msg, (time.time() - t0) / 60)
    return out


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------
def stage_trigger(a):
    if a.poisoner != "1xu":
        log.info("trigger: poisoner is %s, skipping universal pre-train", a.poisoner)
        return
    out = Path(trigger_path(a))
    if out.exists() and not a.force:
        log.info("trigger: %s already exists, skipping", out)
        return
    from modules.train_universal_trigger import run_module as train_trigger
    set_seed(a.seed)
    timed("universal trigger pre-training", train_trigger.run,
          a.name, "train_universal_trigger")


def stage_experts(a):
    if a.poisoner == "1xu" and not Path(trigger_path(a)).exists():
        raise SystemExit(
            f"{trigger_path(a)} not found: run the 'trigger' stage first "
            "(or pass --poisoner 1xu so it is included by default)."
        )
    from modules.train_expert import run_module as train_expert
    for e in range(a.n_experts):
        marker = Path(expert_dir(a, e)) / "DONE"
        if marker.exists() and not a.force:
            log.info("expert %d: done already, skipping", e)
            continue
        set_seed(a.seed + e)
        timed(f"expert {e + 1}/{a.n_experts}", train_expert.run,
              a.name, "train_expert", slurm_id=e)
        marker.touch()


def stage_labels(a):
    from modules.generate_labels import run_module as generate_labels
    out = Path(exp_dir(a))
    if all((out / f).exists() for f in ("labels.npy", "true.npy", "losses.npy")) \
            and not a.force:
        log.info("labels: done already, skipping")
        return
    missing = [e for e in range(a.n_experts)
               if not (Path(expert_dir(a, e)) / "DONE").exists()]
    if missing:
        raise SystemExit(f"Experts {missing} are not trained: run the 'experts' stage first.")
    set_seed(a.seed + 10_000)
    timed("label optimization (trajectory matching)", generate_labels.run,
          a.name, "generate_labels")


def stage_flips(a):
    from modules.select_flips import run_module as select_flips
    out = Path(exp_dir(a))
    if all((out / f"{b}.npy").exists() for b in a.budgets) and not a.force:
        log.info("flips: done already, skipping")
        return
    if not (out / "labels.npy").exists():
        raise SystemExit("labels.npy not found: run the 'labels' stage first.")
    timed(f"flip selection for budgets {a.budgets}", select_flips.run,
          a.name, "select_flips")


def stage_users(a):
    from modules.train_user import run_module as train_user
    for b in a.budgets:
        if not (Path(exp_dir(a)) / f"{b}.npy").exists():
            raise SystemExit(f"{exp_dir(a)}{b}.npy not found: run the 'flips' stage first.")
        for r in range(a.n_user_runs):
            if (Path(user_dir(a, b, r)) / "model.pth").exists() and not a.force:
                log.info("budget %d run %d: done already, skipping", b, r)
                continue
            set_seed(a.seed + 100_000 + b * 100 + r)
            timed(f"user model: budget {b}, run {r + 1}/{a.n_user_runs}",
                  train_user.run, f"{a.name}/user_{b}", "train_user", slurm_id=r)


def stage_spectre(a):
    from modules.eval_spectre import run_module as eval_spectre
    for b in a.spectre_budgets:
        if not (Path(exp_dir(a)) / f"{b}.npy").exists():
            raise SystemExit(f"{exp_dir(a)}{b}.npy not found: run the 'flips' stage first.")
        for r in range(a.n_user_runs):
            model = Path(user_dir(a, b, r)) / "model.pth"
            marker = Path(exp_dir(a)) / f"spectre_{b}" / str(r) / "spectre_metrics.json"
            if not model.exists():
                raise SystemExit(
                    f"{model} not found: run the 'users' stage first "
                    f"(budget {b}, run {r})."
                )
            if marker.exists() and not a.force:
                log.info("spectre budget %d run %d: done already, skipping", b, r)
                continue
            timed(f"SPECTRE eval: budget {b}, run {r + 1}/{a.n_user_runs}",
                  eval_spectre.run, f"{a.name}/spectre_{b}_{r}", "eval_spectre")


def stage_summary(a):
    out = Path(exp_dir(a))
    run_rows, agg_rows = [], []
    for b in a.budgets:
        ctas, ptas, cta_curves, pta_curves = [], [], [], []
        for r in range(a.n_user_runs):
            d = Path(user_dir(a, b, r))
            if not ((d / "caccs.npy").exists() and (d / "paccs.npy").exists()):
                continue
            c, p = np.load(d / "caccs.npy"), np.load(d / "paccs.npy")  # (epochs, 2): acc, loss
            cta_curves.append(c[:, 0])
            pta_curves.append(p[:, 0])
            ctas.append(100 * c[-1, 0])
            ptas.append(100 * p[-1, 0])
            run_rows.append([b, r, ctas[-1], ptas[-1]])
        if not ctas:
            log.warning("budget %d: no finished runs", b)
            continue
        np.savez(out / f"curves_{b}.npz", cta=np.stack(cta_curves), pta=np.stack(pta_curves))
        sd = lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
        agg_rows.append([b, len(ctas), np.mean(ctas), sd(ctas), np.mean(ptas), sd(ptas)])

    with open(out / "summary_runs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["budget", "run", "final_cta", "final_pta"])
        w.writerows(run_rows)
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["budget", "n_runs", "cta_mean", "cta_std", "pta_mean", "pta_std"])
        w.writerows(agg_rows)

    log.info("Final-epoch results (mean +- std over runs, in %)")
    log.info("%8s %6s %16s %16s", "budget", "runs", "CTA", "PTA")
    for b, n, cm, cs, pm, ps in agg_rows:
        log.info("%8d %6d %9.2f +- %4.2f %9.2f +- %4.2f", b, n, cm, cs, pm, ps)


STAGE_FUNCS = dict(trigger=stage_trigger, experts=stage_experts, labels=stage_labels,
                   flips=stage_flips, users=stage_users, spectre=stage_spectre,
                   summary=stage_summary)


def main():
    a = parse_args()
    Path(exp_dir(a)).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(Path(exp_dir(a)) / "run.log")])

    stages = [s for s in STAGES if s in a.stages]
    if not a.skip_checks:
        preflight(a)
    if any(s != "summary" for s in stages):
        write_configs(a)
        with open(Path(exp_dir(a)) / "args.json", "w") as f:
            json.dump(vars(a), f, indent=2)

    for s in stages:
        log.info("===== stage: %s =====", s)
        STAGE_FUNCS[s](a)


if __name__ == "__main__":
    main()