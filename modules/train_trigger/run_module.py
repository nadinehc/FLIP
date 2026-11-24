"""
Trains a GAN-style trigger generator, to be used in place of a hand-coded
trigger such as the sinusoidal `1xs` stripes.

This module produces a trigger, not an attack. Nothing here touches the FLIP
label-flipping procedure: the output is a frozen generator G that
`pick_poisoner`'s "gan" flag turns into an ordinary `Poisoner`, which expert
training and the test-time PTA evaluation then consume exactly as they consume
`StripePoisoner`. FLIP itself stays label-only.

Three networks are trained against each other:

  f -- a surrogate classifier, trained on clean images with true labels plus
       triggered candidate images relabelled to the target. It stands in for
       the victim and tells G whether the trigger is learnable as a shortcut.
  D -- a discriminator separating clean from triggered images, which pushes G
       toward imperceptibility beyond what the epsilon ball alone enforces.
  G -- the trigger generator, maximizing f's confidence in the target label
       while fooling D, inside a hard L-infinity (or L2) ball.

The epsilon bound is structural (`epsilon * tanh`), so G is feasible at every
step and the reported bound is not a projection artifact.
"""

import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from modules.base_utils.datasets import (MappedDataset, TRANSFORM_TRAIN_XY,
                                         TRANSFORM_TEST_XY, get_n_classes,
                                         load_dataset)
from modules.base_utils.image_noise import (Discriminator, build_generator,
                                            coalesce_trigger_config,
                                            flip_candidate_mask,
                                            pairwise_cosine,
                                            save_generator, total_variation,
                                            trigger_generator_kwargs)
from modules.base_utils.util import (extract_toml, load_model, needs_big_ims,
                                     slurmify_path)


def candidate_mask(labels, source_label, target_label, device):
    '''Marks the images the trigger is applied to, as a batch tensor.'''
    mask = flip_candidate_mask(labels.cpu().numpy(), source_label,
                               target_label)
    return torch.from_numpy(mask.astype(np.float32)).to(device)


def run(experiment_name, module_name, **kwargs):
    """
    Trains the trigger generator and saves it.

    :param experiment_name: Name of the experiment in configuration.
    :param module_name: Name of the module in configuration.
    :param kwargs: Additional arguments (such as slurm id).
    """

    slurm_id = kwargs.get('slurm_id', None)
    args = extract_toml(experiment_name, module_name)

    model_flag = args["model"]
    dataset_flag = args["dataset"]
    source_label = args["source_label"]
    target_label = args["target_label"]
    output_dir = slurmify_path(args["output_dir"], slurm_id)
    config = coalesce_trigger_config(args.get("trigger_config", {}))

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Trigger config: {config}")

    # Build datasets ------------------------------------------------------
    print("Building datasets...")
    big_ims = needs_big_ims(model_flag)
    key = dataset_flag + ('_big' if big_ims else '')
    n_classes = get_n_classes(dataset_flag)

    raw_train = load_dataset(dataset_flag, train=True)
    raw_test = load_dataset(dataset_flag, train=False)
    train_set = MappedDataset(raw_train, TRANSFORM_TRAIN_XY[key])
    loader = DataLoader(train_set, batch_size=config['batch_size'],
                        shuffle=True, num_workers=2, drop_last=True)

    # Held-out source-class images, for the final sanity check.
    test_labels = np.array([y for _, y in raw_test])
    test_candidates = np.where(
        flip_candidate_mask(test_labels, source_label, target_label))[0]
    test_set = MappedDataset(Subset(raw_test, test_candidates),
                             TRANSFORM_TEST_XY[key])
    test_loader = DataLoader(test_set, batch_size=256, num_workers=2)

    # Networks ------------------------------------------------------------
    generator = build_generator(
        config['arch'],
        **trigger_generator_kwargs(config, dataset_flag)).to(device)
    print(f"Generator: {config['arch']}, "
          f"{sum(p.numel() for p in generator.parameters())} parameters")

    # The discriminator is optional. With lambda_adv = 0 the imperceptibility
    # pressure comes from the epsilon ball and the TV penalty alone, which is
    # the cheaper and more stable setup; D is there for when that is not
    # enough.
    use_d = config['lambda_adv'] > 0
    discriminator = Discriminator(width=config['d_width']).to(device) \
        if use_d else None
    surrogate = load_model(model_flag, n_classes)

    opt_g = torch.optim.Adam(generator.parameters(), lr=config['g_lr'])
    opt_d = (torch.optim.Adam(discriminator.parameters(), lr=config['d_lr'],
                              betas=(0.5, 0.999)) if use_d else None)
    opt_f = torch.optim.SGD(surrogate.parameters(), lr=config['f_lr'],
                            momentum=0.9, weight_decay=5e-4)

    bce = nn.BCEWithLogitsLoss()
    history = []

    # Training ------------------------------------------------------------
    print("Training trigger...")
    for epoch in range(config['epochs']):
        g_active = epoch >= config['warmup']
        sums, n_batches = {}, 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            mask = candidate_mask(y, source_label, target_label, device)
            if mask.sum() == 0:
                continue
            idx = mask.bool()

            x_pix = generator.to_pixel(x)
            y_target = torch.full_like(y[idx], target_label)

            # -- surrogate: clean images, plus triggered ones as target ----
            with torch.no_grad():
                delta = generator(x[idx])
                x_trig = x[idx] + delta

            surrogate.train()
            loss_f = (F.cross_entropy(surrogate(x), y)
                      + F.cross_entropy(surrogate(x_trig), y_target))
            opt_f.zero_grad()
            loss_f.backward()
            opt_f.step()

            if not g_active:
                sums['loss_f'] = sums.get('loss_f', 0.0) + loss_f.item()
                n_batches += 1
                continue

            # -- discriminator: clean vs triggered -------------------------
            loss_d = torch.zeros((), device=device)
            if use_d:
                with torch.no_grad():
                    trig_pix = torch.clamp(
                        x_pix[idx] + generator.delta_pixel(x_pix[idx]),
                        0.0, 1.0)

                loss_d = (bce(discriminator(x_pix),
                              torch.ones(x_pix.shape[0], device=device))
                          + bce(discriminator(trig_pix),
                                torch.zeros(trig_pix.shape[0], device=device)))
                opt_d.zero_grad()
                loss_d.backward()
                opt_d.step()

            # -- generator: fool f into the target, stay invisible ---------
            delta_pix = generator.delta_pixel(x_pix[idx])
            trig_pix = x_pix[idx] + delta_pix
            trig_norm = generator.to_norm(trig_pix)

            surrogate.eval()
            loss_bd = F.cross_entropy(surrogate(trig_norm), y_target)
            loss_tv = total_variation(delta_pix)
            # Cosine similarity between different images' perturbations. It
            # is always logged; it only enters the loss when lambda_div > 0,
            # and then only above div_tau, so the generator is not pushed to
            # destroy the signal the experts need.
            delta_cos = pairwise_cosine(delta_pix)
            loss_div = F.relu(delta_cos - config['div_tau'])
            loss_adv = (bce(discriminator(trig_pix),
                            torch.ones(trig_pix.shape[0], device=device))
                        if use_d else torch.zeros((), device=device))
            loss_g = (config['lambda_bd'] * loss_bd
                      + config['lambda_tv'] * loss_tv
                      + config['lambda_div'] * loss_div
                      + config['lambda_adv'] * loss_adv)

            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

            for k, v in [('loss_f', loss_f), ('loss_d', loss_d),
                         ('loss_g', loss_g), ('loss_bd', loss_bd),
                         ('loss_tv', loss_tv), ('loss_adv', loss_adv),
                         ('delta_cos', delta_cos)]:
                sums[k] = sums.get(k, 0.0) + v.item()
            sums['linf_255'] = max(
                sums.get('linf_255', 0.0),
                delta_pix.abs().max().item() * 255)
            n_batches += 1

        means = {k: (v / n_batches if k != 'linf_255' else v)
                 for k, v in sums.items()}
        means['epoch'] = epoch
        means['g_active'] = g_active
        history.append(means)
        print(f"  epoch {epoch}: " + "  ".join(
            f"{k}={v:.4f}" for k, v in means.items()
            if isinstance(v, float)))

    # Final check: does the frozen trigger flip held-out source images? ----
    print("Evaluating trigger against the surrogate...")
    generator.eval()
    surrogate.eval()
    hits = total = 0
    cos_sum = cos_n = 0
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            delta = generator(x, quantize=True)
            pred = surrogate(x + delta).argmax(dim=1)
            hits += int((pred == target_label).sum().item())
            total += x.shape[0]
            if x.shape[0] > 1:
                cos_sum += float(pairwise_cosine(delta)) * x.shape[0]
                cos_n += x.shape[0]
    surrogate_asr = hits / max(total, 1)
    delta_cos = cos_sum / max(cos_n, 1)
    print(f"{surrogate_asr=:.4f} on {total} held-out source images")
    print(f"{delta_cos=:.4f} (1.0 = same perturbation on every image)")

    # Save -----------------------------------------------------------------
    generator.quantize = config['quantize']
    save_generator(generator, output_dir)
    if use_d:
        torch.save(discriminator.state_dict(),
                   f"{output_dir}discriminator.pth")
    with open(f"{output_dir}trigger_stats.json", 'w') as f:
        json.dump({
            'config': {k: v for k, v in config.items()},
            'epsilon_255': config['epsilon'] * 255,
            'surrogate_asr': surrogate_asr,
            'delta_cos': delta_cos,
            'n_eval': total,
            'history': history,
        }, f, indent=2)
    print(f"Saved trigger generator to {output_dir}")


if __name__ == "__main__":
    experiment_name, module_name = sys.argv[1], sys.argv[2]
    run(experiment_name, module_name)
