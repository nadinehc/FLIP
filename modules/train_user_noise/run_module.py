"""
Trains a downstream (user) model on input labels, with the attacker's
image-dependent perturbation baked into the flipped examples.

This is the noise-augmented counterpart of `train_user`, which is left
untouched. The perturbation is applied to the raw images *before* the victim's
augmentation pipeline, since the threat model is that the attacker hands over a
dataset and the victim trains on it however they like. Only the examples whose
labels were actually flipped are perturbed; the rest of the dataset is bit-for-
bit clean.

Evaluation is unchanged: clean test accuracy (CTA) on the clean test set and
poison test accuracy (PTA) on the trigger-stamped test set. The generator is an
attack-time tool, not part of the test-time trigger.
"""

from pathlib import Path
import json
import sys

import numpy as np
import torch
from torch.utils.data import Subset

from modules.base_utils.datasets import TRANSFORM_TRAIN_XY, MappedDataset,\
                                        construct_user_dataset, get_matching_datasets,\
                                        get_n_classes, load_dataset, pick_poisoner
from modules.base_utils.image_noise import PerturbedImageDataset, flipped_indices,\
                                           load_generator
from modules.base_utils.util import extract_toml, get_train_info, mini_train, load_model,\
                                    needs_big_ims, slurmify_path, softmax


def run(experiment_name, module_name, **kwargs):
    """
    Runs user model training on perturbed data and saves metrics.

    :param experiment_name: Name of the experiment in configuration.
    :param module_name: Name of the module in configuration.
    :param kwargs: Additional arguments (such as slurm id).
    """

    slurm_id = kwargs.get('slurm_id', None)
    args = extract_toml(experiment_name, module_name)

    user_model_flag = args["user_model"]
    trainer_flag = args["trainer"]
    dataset_flag = args["dataset"]
    poisoner_flag = args["poisoner"]
    clean_label = args["source_label"]
    target_label = args["target_label"]
    soft = args.get("soft", True)
    batch_size = args.get("batch_size", None)
    epochs = args.get("epochs", None)
    optim_kwargs = args.get("optim_kwargs", {})
    scheduler_kwargs = args.get("scheduler_kwargs", {})
    alpha = args.get("alpha", 0.0) or 0.0

    input_path = slurmify_path(args["input_labels"], slurm_id)
    true_path = slurmify_path(args["true_labels"], slurm_id)
    generator_path = slurmify_path(args.get("generator", None), slurm_id)
    output_path = slurmify_path(args["output_dir"], slurm_id)

    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Build datasets
    print("Building datasets...")
    poisoner = pick_poisoner(poisoner_flag, dataset_flag, target_label)

    big_ims = needs_big_ims(user_model_flag)
    _, _, test, poison_test, _ =\
        get_matching_datasets(dataset_flag, poisoner, clean_label, big=big_ims)

    labels_syn = torch.tensor(np.load(input_path))
    y_true = torch.tensor(np.load(true_path))

    if alpha > 0:
        labels_d = softmax(alpha * y_true + (1 - alpha) * labels_syn)
    else:
        labels_d = softmax(labels_syn)

    if not soft:
        labels_d = labels_d.argmax(dim=1)

    # Identify the flipped examples and perturb exactly those
    flipped = flipped_indices(np.load(input_path), np.load(true_path))
    print(f"Flipped labels: {len(flipped)}")

    raw_train = load_dataset(dataset_flag, train=True)
    stats = {'n_flipped': int(len(flipped)), 'n_perturbed': 0}

    if generator_path is not None and len(flipped) > 0:
        print(f"Loading generator from {generator_path}...")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        generator = load_generator(generator_path, device=device)
        perturbed_train = PerturbedImageDataset(raw_train, generator, flipped,
                                                device=device)
        stats.update(perturbed_train.perturbation_stats())
        stats['epsilon'] = generator.epsilon
        stats['norm'] = generator.norm
        print(f"Perturbation stats: {stats}")
    else:
        print("No generator supplied: training on clean images (label-only).")
        perturbed_train = raw_train

    train_transform = TRANSFORM_TRAIN_XY[dataset_flag + ('_big' if big_ims else '')]
    distillation = MappedDataset(
        Subset(perturbed_train, np.arange(len(perturbed_train))),
        train_transform
    )

    user_dataset = construct_user_dataset(distillation, labels_d)

    # Train user model
    print("Training user model...")
    n_classes = get_n_classes(dataset_flag)
    model_retrain = load_model(user_model_flag, n_classes)

    batch_size, epochs, optimizer_retrain, scheduler = get_train_info(
        model_retrain.parameters(), trainer_flag, batch_size,
        epochs, optim_kwargs, scheduler_kwargs
    )

    model_retrain, clean_metrics, poison_metrics = mini_train(
        model=model_retrain,
        train_data=user_dataset,
        test_data=[test, poison_test.poison_dataset],
        batch_size=batch_size,
        opt=optimizer_retrain,
        scheduler=scheduler,
        epochs=epochs,
        record=True
    )

    # Save results
    print("Saving results...")
    np.save(output_path + "paccs.npy", poison_metrics)
    np.save(output_path + "caccs.npy", clean_metrics)
    np.save(output_path + "labels.npy", labels_d.numpy())
    np.save(output_path + "flipped.npy", flipped)
    torch.save(model_retrain.state_dict(), output_path + "model.pth")

    with open(output_path + "noise_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

    # `mini_train` records (acc, loss) per epoch; the final epoch is the metric
    # reported in the paper's tables.
    print(f"clean_test_acc={clean_metrics[-1][0]}")
    print(f"poison_test_acc={poison_metrics[-1][0]}")


if __name__ == "__main__":
    experiment_name, module_name = sys.argv[1], sys.argv[2]
    run(experiment_name, module_name)
