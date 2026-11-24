"""
Runs backdoor defenses against a poisoned dataset and reports detection rates.

The defenses inspect the penultimate-layer representations of a model trained
on the poisoned data, restricted to the examples that carry the target label --
the usual setup, since a clean-label-style attack hides its poisons inside the
target class. Ground truth is the set of examples whose labels were actually
flipped, so the reported detection rate is the fraction of real poisons the
defense would remove.

Defenses are handed the true number of poisons as their removal budget, which
is the convention favourable to the defender used in the FLIP paper's
Appendix D.2. Lower detection rates therefore mean a stealthier attack.
"""

from pathlib import Path
import json
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset

from modules.base_utils.datasets import TRANSFORM_TEST_XY, MappedDataset,\
                                        get_n_classes, load_dataset
from modules.base_utils.image_noise import PerturbedImageDataset, flipped_indices,\
                                           load_generator
from modules.base_utils.util import extract_toml, load_model, make_pbar,\
                                    needs_big_ims, slurmify_path
from modules.run_defenses.utils import DEFENSES, detection_metrics


def get_feature_extractor(model):
    """
    Returns a callable producing penultimate-layer features.

    Every model built by `load_model` is an `nn.Sequential` whose last module
    is the classifier head, except the torchvision VGG/ViT variants, which are
    not supported here.
    """
    if isinstance(model, nn.Sequential):
        return nn.Sequential(*list(model.children())[:-1])
    raise NotImplementedError(
        "Feature extraction is only implemented for the Sequential ResNet "
        "wrappers (r32p, r18, r18-tin)."
    )


def extract_features(model, dataset, device, batch_size=256):
    '''Computes penultimate-layer features for every example in a dataset.'''
    extractor = get_feature_extractor(model).to(device).eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=4)

    features = []
    with torch.no_grad(), make_pbar(total=len(dataset)) as pbar:
        for x, _ in loader:
            x = x.to(device)
            f = extractor(x)
            features.append(f.flatten(1).cpu().numpy())
            pbar.update(len(x))

    return np.concatenate(features, axis=0)


def run(experiment_name, module_name, **kwargs):
    """
    Runs the configured defenses and saves detection metrics.

    :param experiment_name: Name of the experiment in configuration.
    :param module_name: Name of the module in configuration.
    :param kwargs: Additional arguments (such as slurm id).
    """

    slurm_id = kwargs.get('slurm_id', None)
    args = extract_toml(experiment_name, module_name)

    model_flag = args["model"]
    dataset_flag = args["dataset"]
    target_label = args["target_label"]
    defenses = args.get("defenses", ["kmeans", "pca", "spectre"])
    expansion = args.get("expansion", 1.5)

    model_pth = slurmify_path(args["model_pth"], slurm_id)
    input_path = slurmify_path(args["input_labels"], slurm_id)
    true_path = slurmify_path(args["true_labels"], slurm_id)
    generator_path = slurmify_path(args.get("generator", None), slurm_id)
    output_path = slurmify_path(args["output_dir"], slurm_id)

    Path(output_path).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Rebuild the dataset exactly as the victim trained on it
    print("Building datasets...")
    poisoned_labels = np.load(input_path)
    true_labels = np.load(true_path)
    flipped = flipped_indices(poisoned_labels, true_labels)
    hard_labels = poisoned_labels.argmax(axis=1)

    raw_train = load_dataset(dataset_flag, train=True)
    if generator_path is not None and len(flipped) > 0:
        generator = load_generator(generator_path, device=device)
        poisoned_train = PerturbedImageDataset(raw_train, generator, flipped,
                                               device=device)
    else:
        poisoned_train = raw_train

    big_ims = needs_big_ims(model_flag)
    test_transform = TRANSFORM_TEST_XY[dataset_flag + ('_big' if big_ims else '')]
    feature_dataset = MappedDataset(
        Subset(poisoned_train, np.arange(len(poisoned_train))),
        test_transform
    )

    # Load the victim's model and extract representations
    print(f"Loading model from {model_pth}...")
    n_classes = get_n_classes(dataset_flag)
    model = load_model(model_flag, n_classes)
    model.load_state_dict(torch.load(model_pth, map_location=device))

    print("Extracting features...")
    features = extract_features(model, feature_dataset, device)

    # Restrict to the target class, which is where the poisons hide
    target_inds = np.where(hard_labels == target_label)[0]
    poison_mask = np.isin(target_inds, flipped)
    target_features = features[target_inds].astype(np.float64)

    print(f"Examining {len(target_inds)} target-label examples "
          f"({int(poison_mask.sum())} poisoned).")

    # Run the defenses
    results = {
        'n_flipped_total': int(len(flipped)),
        'n_target_label': int(len(target_inds)),
        'n_poison_in_target': int(poison_mask.sum()),
        'expansion': expansion,
        'defenses': {},
    }

    n_poison = int(poison_mask.sum())
    for name in defenses:
        if name not in DEFENSES:
            raise NotImplementedError(f"Unknown defense {name}.")
        print(f"Running {name}...")
        scores = DEFENSES[name](target_features, n_poison)
        metrics = detection_metrics(scores, poison_mask, expansion=expansion)
        results['defenses'][name] = metrics
        print(f"  {name}: {metrics}")
        np.save(f"{output_path}scores_{name}.npy", scores)

    with open(output_path + "defenses.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved defense results to {output_path}defenses.json")


if __name__ == "__main__":
    experiment_name, module_name = sys.argv[1], sys.argv[2]
    run(experiment_name, module_name)
