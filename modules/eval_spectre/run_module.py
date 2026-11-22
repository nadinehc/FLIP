"""
Run full SPECTRE (and PCA / kmeans) filters on a trained user model.

Independent of spectre_proxy in generate_labels: works on any FLIP user
checkpoint trained with hard flipped labels.
"""

from pathlib import Path
import json
import os
import subprocess
import sys

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from modules.base_utils.datasets import (
    get_matching_datasets,
    get_n_classes,
    pick_poisoner,
    construct_user_dataset,
)
from modules.base_utils.reps import compute_all_reps, default_spectre_layer
from modules.base_utils.util import (
    extract_toml,
    load_model,
    mini_train,
    get_train_info,
    needs_big_ims,
    slurmify_path,
    clf_eval,
)
from modules.eval_spectre.filters import run_all_filters


ROOT = Path(__file__).resolve().parents[2]
SPECTRE_DIR = ROOT / "spectre-defense"
JULIA_FILTER = SPECTRE_DIR / "run_filters_explicit.jl"


class _IndexedDataset(Dataset):
    """Wrap (x, y) dataset so Subset indices stay stable."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __getitem__(self, i):
        return self.dataset[i]

    def __len__(self):
        return len(self.dataset)


def _hard_labels(path):
    arr = np.load(path)
    if arr.ndim == 1:
        return arr.astype(np.int64)
    return arr.argmax(axis=1).astype(np.int64)


def _run_python_filters(reps: np.ndarray, eps: int, out_dir: Path):
    print("Running SPECTRE filters (Python port)...")
    masks = run_all_filters(reps, eps)
    np.save(out_dir / "mask-pca-target.npy", masks["pca"])
    np.save(out_dir / "mask-kmeans-target.npy", masks["kmeans"])
    np.save(out_dir / "mask-rcov-target.npy", masks["rcov"])
    return masks


def _run_julia_filters(reps_path: Path, eps: int, out_dir: Path):
    if not JULIA_FILTER.exists():
        raise FileNotFoundError(f"Missing Julia filter script: {JULIA_FILTER}")
    cmd = [
        "julia",
        f"--project={SPECTRE_DIR}",
        str(JULIA_FILTER),
        str(reps_path.resolve()),
        str(eps),
        str(out_dir.resolve()),
    ]
    print("Running:", " ".join(cmd))
    env = dict(os.environ)
    tmp = ROOT / "temp"
    tmp.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = env["TMP"] = env["TEMP"] = str(tmp)
    subprocess.run(cmd, check=True, cwd=str(SPECTRE_DIR), env=env)


def _mask_metrics(mask, eps):
    """
    SPECTRE masks: True means 'poison / remove' in the Julia scripts' poison_ind
    convention used by train.py (target_mask True = remove).
    Julia writes estimated_poison_ind where True = keep in some filters —
    run_filters_explicit writes the same style as run_filters.jl:
      pca: k_lowest_ind(...) -> True for selected-to-remove
      kmeans/quantum: .! filter -> True for remove
    """
    mask = np.asarray(mask).astype(bool)
    n = len(mask)
    removed = int(mask.sum())
    poison_removed = int(mask[n - eps :].sum()) if eps > 0 else 0
    clean_removed = removed - poison_removed
    return {
        "removed_total": removed,
        "poison_removed": poison_removed,
        "clean_removed": clean_removed,
        "poison_recall": poison_removed / eps if eps > 0 else 0.0,
    }


def run(experiment_name, module_name, **kwargs):
    slurm_id = kwargs.get("slurm_id", None)
    args = extract_toml(experiment_name, module_name)

    model_path = slurmify_path(args["model_path"], slurm_id)
    input_labels = slurmify_path(args["input_labels"], slurm_id)
    true_labels_path = slurmify_path(args["true_labels"], slurm_id)
    output_dir = Path(slurmify_path(args["output_dir"], slurm_id))
    output_dir.mkdir(parents=True, exist_ok=True)

    user_model_flag = args["user_model"]
    dataset_flag = args["dataset"]
    poisoner_flag = args["poisoner"]
    source_label = args["source_label"]
    target_label = args["target_label"]
    universal_trigger_path = args.get("universal_trigger_path", None)
    spectre_layer = args.get("spectre_layer", None)
    if spectre_layer is None:
        spectre_layer = default_spectre_layer(user_model_flag)

    retrain = bool(args.get("retrain", False))
    backend = args.get("backend", "python")  # python | julia
    trainer_flag = args.get("trainer", "sgd")
    batch_size = args.get("batch_size", None)
    epochs = args.get("epochs", None)
    optim_kwargs = args.get("optim_kwargs", {})
    scheduler_kwargs = args.get("scheduler_kwargs", {})
    filters = args.get("filters", ["rcov", "pca", "kmeans"])

    print("Building user training set...")
    poisoner = pick_poisoner(
        poisoner_flag, dataset_flag, target_label,
        universal_trigger_path=universal_trigger_path,
    )
    big_ims = needs_big_ims(user_model_flag)
    _, distillation, test, poison_test, _ = get_matching_datasets(
        dataset_flag, poisoner, source_label, big=big_ims
    )

    hard = _hard_labels(input_labels)
    true = _hard_labels(true_labels_path)
    if len(hard) != len(distillation):
        raise ValueError(
            f"Label length {len(hard)} != distill size {len(distillation)}"
        )

    flipped = hard != true
    clean_target_idx = np.where((hard == target_label) & (true == target_label))[0]
    flip_to_target_idx = np.where((hard == target_label) & flipped)[0]
    eps = int(len(flip_to_target_idx))
    if eps == 0:
        raise ValueError(
            "No flipped-to-target examples found; SPECTRE eval needs label flips "
            "into the target class."
        )

    # SPECTRE / Julia convention: poisons are the last `eps` rows
    ordered_idx = np.concatenate([clean_target_idx, flip_to_target_idx])
    np.save(output_dir / "ordered_indices.npy", ordered_idx)
    np.save(output_dir / "flip_to_target_indices.npy", flip_to_target_idx)

    user_dataset = construct_user_dataset(distillation, torch.tensor(hard))
    target_dataset = Subset(_IndexedDataset(user_dataset), ordered_idx.tolist())

    print(f"Loading model from {model_path}...")
    n_classes = get_n_classes(dataset_flag)
    model = load_model(user_model_flag, n_classes)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    print(f"Extracting layer {spectre_layer} representations "
          f"(n={len(ordered_idx)}, eps={eps})...")
    reps = compute_all_reps(
        model, target_dataset, layers=[spectre_layer], flat=True
    )[spectre_layer].numpy()
    reps_path = output_dir / f"label_{target_label}_reps.npy"
    np.save(reps_path, reps)

    if backend == "julia":
        print("Running SPECTRE filters (Julia)...")
        _run_julia_filters(reps_path, eps, output_dir)
    else:
        _run_python_filters(reps, eps, output_dir)

    # Budget inferred from "{m}.npy" when possible (for comparison tables)
    budget = args.get("budget", None)
    if budget is None:
        stem = Path(input_labels).stem
        budget = int(stem) if stem.isdigit() else None

    metrics = {
        "tag": args.get("tag", experiment_name),
        "experiment": experiment_name,
        "poisoner": poisoner_flag,
        "universal_trigger_path": universal_trigger_path,
        "source_label": int(source_label),
        "target_label": int(target_label),
        "budget": budget,
        "flip_budget_file": str(input_labels),
        "n_flips_total": int(flipped.sum()),
        "eps": eps,
        "n_target_labeled": int(len(ordered_idx)),
        "n_clean_target": int(len(clean_target_idx)),
        "spectre_layer": int(spectre_layer),
        "user_model": user_model_flag,
        "dataset": dataset_flag,
        "model_path": str(model_path),
        "backend": backend,
        "filters": {},
    }

    mask_files = {
        "pca": "mask-pca-target.npy",
        "kmeans": "mask-kmeans-target.npy",
        "rcov": "mask-rcov-target.npy",
    }
    for name in filters:
        fname = mask_files.get(name)
        if fname is None or not (output_dir / fname).exists():
            print(f"Warning: mask for filter '{name}' missing, skipping")
            continue
        mask = np.load(output_dir / fname)
        metrics["filters"][name] = _mask_metrics(mask, eps)
        print(f"{name}: {metrics['filters'][name]}")

    if retrain and (output_dir / "mask-rcov-target.npy").exists():
        print("Retraining on SPECTRE-cleaned labels...")
        mask = np.load(output_dir / "mask-rcov-target.npy").astype(bool)
        # mask aligns with ordered_idx; True = remove
        remove_global = set(ordered_idx[mask].tolist())
        keep = [i for i in range(len(hard)) if i not in remove_global]
        cleaned = Subset(user_dataset, keep)

        model_retrain = load_model(user_model_flag, n_classes)
        bs, ep, opt, sched = get_train_info(
            model_retrain.parameters(),
            trainer_flag,
            batch_size=batch_size,
            epochs=epochs,
            optim_kwargs=optim_kwargs,
            scheduler_kwargs=scheduler_kwargs,
        )
        model_retrain, clean_metrics, poison_metrics = mini_train(
            model=model_retrain,
            train_data=cleaned,
            test_data=[test, poison_test.poison_dataset],
            batch_size=bs,
            opt=opt,
            scheduler=sched,
            epochs=ep,
            record=True,
        )
        np.save(output_dir / "caccs_post.npy", clean_metrics)
        np.save(output_dir / "paccs_post.npy", poison_metrics)
        torch.save(model_retrain.state_dict(), output_dir / "model_retrain.pth")
        metrics["post_spectre"] = {
            "cta": float(clean_metrics[-1][0]),
            "pta": float(poison_metrics[-1][0]),
        }
        print(f"post-SPECTRE CTA={metrics['post_spectre']['cta']:.4f} "
              f"PTA={metrics['post_spectre']['pta']:.4f}")

    # Pre-filter reference metrics on the loaded model
    cta = clf_eval(model, test)[0]
    pta = clf_eval(model, poison_test.poison_dataset)[0]
    metrics["pre_spectre"] = {"cta": float(cta), "pta": float(pta)}
    print(f"pre-SPECTRE CTA={cta:.4f} PTA={pta:.4f}")

    with open(output_dir / "spectre_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote {output_dir / 'spectre_metrics.json'}")


if __name__ == "__main__":
    experiment_name, module_name = sys.argv[1], sys.argv[2]
    run(experiment_name, module_name)
