"""
Optimizes logit labels by directly solving the bilevel problem (2), with the
inner training replaced by NTK kernel regression (Appendix A).
"""

from pathlib import Path
import sys

import torch
import numpy as np
from torch.utils.data import Subset

from modules.base_utils.datasets import get_matching_datasets, pick_poisoner, get_n_classes,\
                                        MappedDataset, TRANSFORM_TEST_XY
from modules.base_utils.util import extract_toml, load_model, get_module_device, make_pbar,\
                                    needs_big_ims, slurmify_path, softmax
from modules.generate_labels.utils import extract_labels
from modules.generate_labels_ntk.utils import DEFAULT_NTK_CONFIG, ntk_features,\
                                              kernel_regression_operators


def run(experiment_name, module_name, **kwargs):
    """
    Optimizes and saves poisoned logit labels.

    :param experiment_name: Name of the experiment in configuration.
    :param module_name: Name of the module in configuration.
    :param kwargs: Additional arguments (such as slurm id).
    """

    slurm_id = kwargs.get('slurm_id', None)

    args = extract_toml(experiment_name, module_name)

    model_flag = args["model"]
    dataset_flag = args["dataset"]
    poisoner_flag = args["poisoner"]
    clean_label = args["source_label"]
    target_label = args["target_label"]
    lam = args.get("lambda", 0.0)
    train_pct = args.get("train_pct", 1.0)
    input_pth = slurmify_path(args.get("input_pth", None), slurm_id)
    config = {**DEFAULT_NTK_CONFIG, **args.get("ntk_config", {})}

    output_dir = slurmify_path(args["output_dir"], slurm_id)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Build datasets: X (kernel train set whose labels are optimized),
    # P (triggered source images, proxy for PTA) and V (clean images, proxy for CTA)
    print("Building datasets...")
    n_classes = get_n_classes(dataset_flag)
    poisoner = pick_poisoner(poisoner_flag, dataset_flag, target_label)

    big_ims = needs_big_ims(model_flag)
    _, _, _, _, mtt_dataset =\
        get_matching_datasets(dataset_flag, poisoner, clean_label, train_pct=train_pct, big=big_ims)
    distill = mtt_dataset.distill
    transform = TRANSFORM_TEST_XY[dataset_flag + ('_big' if big_ims else '')]

    rng = np.random.RandomState(config['seed'])
    perm = rng.permutation(len(distill))
    train_inds = perm[:config['n_train']]
    val_inds = perm[config['n_train']:config['n_train'] + config['n_val']]
    poison_inds = np.setdiff1d(mtt_dataset.poison_inds, train_inds)
    poison_inds = rng.choice(poison_inds, min(config['n_poison'], len(poison_inds)), replace=False)

    train_set = MappedDataset(Subset(distill, train_inds), transform)
    val_set = MappedDataset(Subset(distill, val_inds), transform)
    poison_set = MappedDataset(MappedDataset(Subset(distill, poison_inds), poisoner), transform)

    labels_true = torch.stack(extract_labels(distill, 1, n_classes))
    labels_all = torch.stack(extract_labels(distill, config['one_hot_temp'], n_classes))

    # Compute empirical NTK features and solve the inner problem in closed form
    print("Computing NTK features...")
    model = load_model(model_flag, n_classes)
    if input_pth is not None:
        model.load_state_dict(torch.load(input_pth))
    device = get_module_device(model)

    phi_train, phi_poison, phi_val = [
        ntk_features(model, d, n_classes, config['sketch_dim'], config['seed'])
        for d in (train_set, poison_set, val_set)
    ]
    op_poison, op_val = kernel_regression_operators(phi_train, [phi_poison, phi_val],
                                                    config['ridge'])

    # Optimize labels on the outer problem: max PTA + cta_weight * CTA,
    # relaxed to squared losses on the kernel regression predictions
    print("Training...")
    y_init = labels_true[train_inds].to(device)
    y_poison = torch.zeros(len(poison_inds), n_classes, device=device)
    y_poison[:, target_label] = 1
    y_val = labels_true[val_inds].to(device)

    labels_syn = labels_all[train_inds].clone().to(device).requires_grad_(True)
    optimizer_labels = torch.optim.Adam([labels_syn], lr=config['lr'])

    losses = []
    with make_pbar(total=config['iterations']) as pbar:
        for _ in range(config['iterations']):
            y_d = softmax(labels_syn)
            pred_poison = op_poison @ y_d
            pred_val = op_val @ y_d

            pta_loss = ((pred_poison - y_poison) ** 2).sum(dim=1).mean()
            cta_loss = ((pred_val - y_val) ** 2).sum(dim=1).mean()
            reg_term = lam * torch.linalg.vector_norm(y_d - y_init, ord=1, axis=1).mean()
            grand_loss = pta_loss + config['cta_weight'] * cta_loss + reg_term

            optimizer_labels.zero_grad()
            grand_loss.backward()
            optimizer_labels.step()

            # Record training information (PTA / CTA of the kernel model)
            losses.append(grand_loss.item())
            pbar.update(1)
            pbar.set_postfix(**{
                'g_loss': "%.4g" % losses[-1],
                'pta': "%.2f" % (pred_poison.argmax(1) == target_label).float().mean().item(),
                'cta': "%.2f" % (pred_val.argmax(1) == y_val.argmax(1)).float().mean().item(),
                'flips': int((labels_syn.argmax(1) != y_init.argmax(1)).sum().item()),
            })

    # Save results in the same format as generate_labels (labels outside of X stay clean)
    print("Saving results...")
    labels_all[train_inds] = labels_syn.detach().cpu()
    np.save(output_dir + "labels.npy", labels_all.numpy())
    np.save(output_dir + "true.npy", labels_true.numpy())
    np.save(output_dir + "losses.npy", losses)

if __name__ == "__main__":
    experiment_name, module_name = sys.argv[1], sys.argv[2]
    run(experiment_name, module_name)
