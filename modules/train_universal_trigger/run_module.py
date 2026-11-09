"""
Pre-train a frozen universal L∞ perturbation for use as poisoner flag 1xu.
"""

from pathlib import Path
import sys

from torch.utils.data import Subset

from modules.base_utils.datasets import (
    TRANSFORM_TRAIN_XY,
    TRANSFORM_TEST_XY,
    MappedDataset,
    get_n_classes,
    load_dataset,
)
from modules.base_utils.util import (
    clf_eval,
    extract_toml,
    get_train_info,
    load_model,
    mini_train,
    needs_big_ims,
    slurmify_path,
)
from modules.train_universal_trigger.utils import (
    build_source_loader,
    eval_delta_pta,
    optimize_universal_delta,
    save_delta,
)


def run(experiment_name, module_name, **kwargs):
    """
    Train a clean surrogate, optimize universal δ, and save delta.npy + meta.json.

    :param experiment_name: Name of the experiment in configuration.
    :param module_name: Name of the module in configuration.
    :param kwargs: Additional arguments (such as slurm id).
    """
    slurm_id = kwargs.get("slurm_id", None)
    args = extract_toml(experiment_name, module_name)

    model_flag = args["model"]
    dataset_flag = args["dataset"]
    trainer_flag = args.get("trainer", "sgd")
    source_label = args["source_label"]
    target_label = args["target_label"]
    eps = float(args.get("eps", 8 / 255))
    delta_steps = int(args.get("delta_steps", 1000))
    delta_lr = float(args.get("delta_lr", 0.01))
    train_pct = float(args.get("train_pct", 1.0))
    batch_size = args.get("batch_size", None)
    surrogate_epochs = args.get("surrogate_epochs", None)
    optim_kwargs = args.get("optim_kwargs", {})
    scheduler_kwargs = args.get("scheduler_kwargs", {})
    seed = int(args.get("seed", 0))

    output_path = slurmify_path(args["output_path"], slurm_id)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    big_ims = needs_big_ims(model_flag)
    if big_ims:
        raise NotImplementedError(
            "Universal trigger pre-training currently supports native-resolution models only"
        )

    print("Building clean surrogate datasets...")
    train_raw = load_dataset(dataset_flag, train=True)
    test_raw = load_dataset(dataset_flag, train=False)
    n_train = int(len(train_raw) * train_pct)
    train_subset = Subset(train_raw, range(n_train))
    train_ds = MappedDataset(train_subset, TRANSFORM_TRAIN_XY[dataset_flag])
    test_ds = MappedDataset(test_raw, TRANSFORM_TEST_XY[dataset_flag])

    print("Training clean surrogate...")
    n_classes = get_n_classes(dataset_flag)
    model = load_model(model_flag, n_classes)
    batch_size, epochs, opt, lr_scheduler = get_train_info(
        model.parameters(),
        trainer_flag,
        batch_size=batch_size,
        epochs=surrogate_epochs,
        optim_kwargs=optim_kwargs,
        scheduler_kwargs=scheduler_kwargs,
    )
    mini_train(
        model=model,
        train_data=train_ds,
        test_data=test_ds,
        batch_size=batch_size,
        opt=opt,
        scheduler=lr_scheduler,
        epochs=epochs,
    )
    clean_acc, _ = clf_eval(model, test_ds)
    print(f"surrogate_clean_acc={clean_acc}")

    print("Optimizing universal delta...")
    source_loader, source_inds = build_source_loader(
        dataset_flag,
        source_label,
        target_label,
        train_pct=train_pct,
        batch_size=batch_size,
    )
    delta = optimize_universal_delta(
        model,
        source_loader,
        dataset_flag=dataset_flag,
        target_label=target_label,
        eps=eps,
        steps=delta_steps,
        lr=delta_lr,
    )
    pta = eval_delta_pta(
        model, source_loader, delta, dataset_flag, target_label
    )
    print(f"surrogate_delta_pta={pta}")

    meta = {
        "eps": eps,
        "delta_steps": delta_steps,
        "delta_lr": delta_lr,
        "source_label": source_label,
        "target_label": target_label,
        "dataset": dataset_flag,
        "model": model_flag,
        "shape": list(delta.shape),
        "seed": seed,
        "n_source": int(len(source_inds)),
        "surrogate_clean_acc": float(clean_acc),
        "surrogate_delta_pta": float(pta),
        "surrogate_epochs": int(epochs),
    }
    out, meta_path = save_delta(delta, output_path, meta)
    print(f"Saved universal trigger to {out} (meta: {meta_path})")


if __name__ == "__main__":
    experiment_name, module_name = sys.argv[1], sys.argv[2]
    run(experiment_name, module_name)
