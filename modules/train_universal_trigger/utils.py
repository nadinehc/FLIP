"""Helpers for learning a frozen universal L∞ perturbation."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from modules.base_utils.datasets import (
    CIFAR_100_TRANSFORM_NORMALIZE_MEAN,
    CIFAR_100_TRANSFORM_NORMALIZE_STD,
    CIFAR_TRANSFORM_NORMALIZE_MEAN,
    CIFAR_TRANSFORM_NORMALIZE_STD,
    TINY_IMAGENET_TRANSFORM_NORMALIZE_MEAN,
    TINY_IMAGENET_TRANSFORM_NORMALIZE_STD,
    load_dataset,
)
from modules.base_utils.util import clf_correct, clf_loss, default_device, make_pbar


NORMALIZE = {
    "cifar": (CIFAR_TRANSFORM_NORMALIZE_MEAN, CIFAR_TRANSFORM_NORMALIZE_STD),
    "cifar_100": (CIFAR_100_TRANSFORM_NORMALIZE_MEAN, CIFAR_100_TRANSFORM_NORMALIZE_STD),
    "tiny_imagenet": (
        TINY_IMAGENET_TRANSFORM_NORMALIZE_MEAN,
        TINY_IMAGENET_TRANSFORM_NORMALIZE_STD,
    ),
}

IMAGE_SIZE = {
    "cifar": 32,
    "cifar_100": 32,
    "tiny_imagenet": 64,
}


class ToTensorDataset(Dataset):
    """PIL dataset -> float tensor in [0, 1], no normalize / augment."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        x, y = self.dataset[i]
        x = torch.as_tensor(np.asarray(x), dtype=torch.float32).permute(2, 0, 1) / 255.0
        return x, y


def normalize_batch(x, dataset_flag):
    mean, std = NORMALIZE[dataset_flag]
    mean = torch.tensor(mean, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def select_source_indices(train_data, source_label, target_label, train_pct=1.0):
    labels = np.array([y for _, y in train_data])
    labels = labels[: int(len(labels) * train_pct)]
    n_classes = len(train_data.classes)
    n_poisons = int((len(train_data) // n_classes) * train_pct)
    if source_label == -1:
        inds = np.where(labels != target_label)[0][-n_poisons:]
    else:
        inds = np.where(labels == source_label)[0][-n_poisons:]
    return inds


def optimize_universal_delta(
    model,
    source_loader,
    *,
    dataset_flag,
    target_label,
    eps,
    steps,
    lr,
    device=None,
):
    """Projected Adam on a free δ with ||δ||_∞ ≤ eps (in [0, 1] pixel space)."""
    device = device or default_device
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    size = IMAGE_SIZE[dataset_flag]
    delta = torch.zeros(1, 3, size, size, device=device, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    target = torch.tensor([target_label], device=device)

    step = 0
    with make_pbar(total=steps, desc="optimize_delta") as pbar:
        while step < steps:
            for x, _ in source_loader:
                if step >= steps:
                    break
                x = x.to(device)
                opt.zero_grad()
                x_adv = (x + delta).clamp(0.0, 1.0)
                logits = model(normalize_batch(x_adv, dataset_flag))
                y = target.expand(x.size(0))
                loss = clf_loss(logits, y)
                loss.backward()
                opt.step()
                with torch.no_grad():
                    delta.clamp_(-eps, eps)
                step += 1
                pbar.update(1)
                pbar.set_postfix(loss=f"{loss.item():.4g}",
                                 linf=f"{delta.detach().abs().max().item():.4g}")

    return delta.detach().squeeze(0).cpu()


@torch.no_grad()
def eval_delta_pta(model, source_loader, delta, dataset_flag, target_label, device=None):
    device = device or default_device
    model.eval()
    delta = delta.to(device)
    if delta.dim() == 3:
        delta = delta.unsqueeze(0)
    total_correct, n = 0, 0
    target = torch.tensor([target_label], device=device)
    for x, _ in source_loader:
        x = x.to(device)
        x_adv = (x + delta).clamp(0.0, 1.0)
        logits = model(normalize_batch(x_adv, dataset_flag))
        y = target.expand(x.size(0))
        total_correct += int(clf_correct(logits, y).item())
        n += x.size(0)
    return total_correct / max(n, 1)


def save_delta(delta_chw, output_path, meta):
    """Save HxWx3 pixel-unit delta.npy next to meta.json."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    delta_hwc = delta_chw.detach().cpu().numpy().transpose(1, 2, 0) * 255.0
    np.save(output_path, delta_hwc.astype(np.float32))
    meta_path = output_path.with_name("meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return output_path, meta_path


def build_source_loader(dataset_flag, source_label, target_label,
                        train_pct=1.0, batch_size=256):
    train_data = load_dataset(dataset_flag, train=True)
    inds = select_source_indices(train_data, source_label, target_label, train_pct)
    subset = ToTensorDataset(Subset(train_data, inds))
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    return loader, inds
