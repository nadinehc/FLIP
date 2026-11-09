"""
Intermediate-representation helpers for SPECTRE proxy / eval.

Layer indices match spectre-defense/rep_saver.py (r32p: 14, r18: 13).
"""

from typing import Collection, Dict, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from modules.base_utils.util import either_dataloader_dataset_to_both, get_module_device


DEFAULT_SPECTRE_LAYER = {
    "r32p": 14,
    "r18": 13,
    "r18-tin": 13,
}


def default_spectre_layer(model_flag: str) -> int:
    if model_flag not in DEFAULT_SPECTRE_LAYER:
        raise ValueError(
            f"No default SPECTRE layer for model '{model_flag}'. "
            f"Supported: {sorted(DEFAULT_SPECTRE_LAYER)}. Pass spectre_layer explicitly."
        )
    return DEFAULT_SPECTRE_LAYER[model_flag]


def forward_to_layer(model: nn.Sequential, x: torch.Tensor, layer: int) -> torch.Tensor:
    """Forward through Sequential modules 0..layer inclusive (keeps autograd)."""
    if not isinstance(model, nn.Sequential):
        raise TypeError(
            f"forward_to_layer expects nn.Sequential, got {type(model).__name__}"
        )
    if layer < 0 or layer >= len(model):
        raise IndexError(f"layer={layer} out of range for Sequential(len={len(model)})")
    for i, mod in enumerate(model):
        x = mod(x)
        if i == layer:
            return x
    raise RuntimeError("unreachable")


def features_under_params(
    model: nn.Sequential,
    param_values,
    x: torch.Tensor,
    layer: int,
) -> torch.Tensor:
    """
    Forward to `layer` after temporarily writing `param_values` into `model`.

    Gradients through the written weights are intentionally detached (PyTorch 1.11
    has no functional_call). Callers should route label gradients through soft
    weights in the SPECTRE proxy, not through this feature tensor.
    """
    saved = [p.data.clone() for p in model.parameters()]
    try:
        with torch.no_grad():
            for p, v in zip(model.parameters(), param_values):
                p.data.copy_(v.data if torch.is_tensor(v) else v)
            feats = forward_to_layer(model, x, layer).detach()
    finally:
        with torch.no_grad():
            for p, s in zip(model.parameters(), saved):
                p.data.copy_(s)
    return feats


def compute_all_reps(
    model: nn.Sequential,
    data: Union[DataLoader, Dataset],
    *,
    layers: Collection[int],
    flat: bool = False,
) -> Dict[int, torch.Tensor]:
    """
    Extract intermediate activations (no grad). Port of spectre-defense/util.compute_all_reps.
    """
    device = get_module_device(model)
    dataloader, dataset = either_dataloader_dataset_to_both(data, eval=True)
    n = len(dataset)
    max_layer = max(layers)
    if max_layer >= len(model):
        raise IndexError(f"max layer {max_layer} >= len(model)={len(model)}")

    reps: Dict[int, torch.Tensor] = {}
    x0 = dataset[0][0][None, ...].to(device)
    with torch.no_grad():
        for i, layer_mod in enumerate(model):
            if i > max_layer:
                break
            x0 = layer_mod(x0)
            if i in layers:
                reps[i] = torch.empty(n, *x0.shape[1:])

        model.eval()
        start = 0
        for batch in dataloader:
            x = batch[0].to(device)
            mb = len(x)
            for i, layer_mod in enumerate(model):
                if i > max_layer:
                    break
                x = layer_mod(x)
                if i in layers:
                    reps[i][start : start + mb] = x.cpu()
            start += mb

    if flat:
        for i in list(reps.keys()):
            t = reps[i]
            reps[i] = t.reshape(t.shape[0], -1)
    return reps
