"""Surrogate classifiers for the deer-in-truck steganography attack.

Defines several architectures (ResNet-18, ResNet-32, EfficientNet-B0), each trainable with
multiple seeds, and a helper that loads a checkpoint if it exists or trains and saves it otherwise.

Everything uses the same [-1, 1] normalisation as the rest of the project (Normalize(0.5, 0.5)).
Checkpoints are plain state_dicts at  <ckpt_dir>/<arch>_seed<seed>.pth  (ckpt_dir defaults to
noise_generator/surrogates), so the ResNet-18 checkpoints trained earlier in the notebook are reused.

CLI:  python -m modules.Steganography.surrogates --archs resnet18 resnet32 efficientnet --seeds 0 1 2 3 4
"""

import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from modules.pytorch_cifar.models.resnet import ResNet18
from modules.pytorch_cifar.models.efficientnet import EfficientNetB0
from modules.base_utils.model.resnet import resnet32

DEFAULT_CKPT_DIR = "noise_generator/surrogates"
DEFAULT_DATA_ROOT = "data/data_cifar10"

# name -> (builder, output feature dim before the final linear layer)
ARCHS = {
    "resnet18": (ResNet18, 512),
    "resnet32": (lambda: resnet32(10), 64),
    "efficientnet": (EfficientNetB0, 320),
}


def build_surrogate(arch):
    if arch not in ARCHS:
        raise ValueError(f"unknown arch {arch!r}; choose from {list(ARCHS)}")
    return ARCHS[arch][0]()


def feature_dim(arch):
    return ARCHS[arch][1]


def extract_features(model, x, arch):
    """Penultimate features (output of global pooling, just before `model.linear`)."""
    if arch == "resnet18":
        out = F.relu(model.bn1(model.conv1(x)))
        out = model.layer4(model.layer3(model.layer2(model.layer1(out))))
        return F.avg_pool2d(out, 4).flatten(1)                       # [B, 512]
    if arch == "resnet32":
        out = model.conv1(x)
        out = model.layer3(model.layer2(model.layer1(out)))
        out = F.leaky_relu(model.final_bn(out), 0.1)
        return F.avg_pool2d(out, out.size(3)).flatten(1)             # [B, 64]
    if arch == "efficientnet":
        out = _swish(model.bn1(model.conv1(x)))                      # swish stem, as in EfficientNet.forward
        out = model.layers(out)
        return F.adaptive_avg_pool2d(out, 1).flatten(1)              # [B, 320]
    raise ValueError(arch)


def _swish(x):
    return x * x.sigmoid()


# --------------------------------------------------------------------------- data

def _loaders(data_root, batch_size=128, seed=0, num_workers=4):
    transform_aug = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_set = datasets.CIFAR10(data_root, train=True, download=False, transform=transform_aug)
    test_set = datasets.CIFAR10(data_root, train=False, download=False, transform=transform_test)
    g = torch.Generator(); g.manual_seed(seed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, generator=g, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        n += len(y)
    return correct / n


# --------------------------------------------------------------------------- training

def train_surrogate(arch, seed, device, data_root=DEFAULT_DATA_ROOT, epochs=40, lr=0.1, verbose=True):
    """Train one surrogate from scratch. Same recipe for every architecture
    (SGD + Nesterov, OneCycle, AMP, channels_last)."""
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)
    train_loader, test_loader = _loaders(data_root, seed=seed)

    model = build_surrogate(arch).to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader))
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        model.train(); t0 = time.time(); tot = correct = n = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(y); correct += (logits.argmax(1) == y).sum().item(); n += len(y)
        if verbose:
            msg = f"[{arch} seed {seed}] epoch {epoch+1}/{epochs}  loss {tot/n:.3f}  train acc {correct/n:.3f}  ({time.time()-t0:.0f}s)"
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                msg += f"  test acc {accuracy(model, test_loader, device):.4f}"
            print(msg)
    return model


def ckpt_path(arch, seed, ckpt_dir=DEFAULT_CKPT_DIR):
    return os.path.join(ckpt_dir, f"{arch}_seed{seed}.pth")


def get_surrogates(archs=("resnet18", "resnet32", "efficientnet"), seeds=(0, 1, 2, 3, 4),
                   device=None, ckpt_dir=DEFAULT_CKPT_DIR, data_root=DEFAULT_DATA_ROOT,
                   epochs=40, lr=0.1, freeze=True, verbose=True):
    """Load every (arch, seed) checkpoint, training and saving any that are missing.

    Returns a dict keyed by (arch, seed) -> model (eval mode; frozen if freeze=True).
    A checkpoint that already exists is loaded, never retrained.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(ckpt_dir, exist_ok=True)
    surrogates = {}
    for arch in archs:
        for seed in seeds:
            path = ckpt_path(arch, seed, ckpt_dir)
            model = build_surrogate(arch).to(device)
            if os.path.exists(path):
                model.load_state_dict(torch.load(path, map_location=device))
                if verbose:
                    print(f"loaded {path}")
            else:
                if verbose:
                    print(f"training {path}")
                model = train_surrogate(arch, seed, device, data_root=data_root, epochs=epochs, lr=lr, verbose=verbose)
                torch.save(model.state_dict(), path)
            model = model.to(memory_format=torch.contiguous_format).eval()
            if freeze:
                for p in model.parameters():
                    p.requires_grad_(False)
            surrogates[(arch, seed)] = model
    return surrogates


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train surrogate classifiers for the steganography attack.")
    parser.add_argument("--archs", nargs="+", default=["resnet18", "resnet32", "efficientnet"], choices=list(ARCHS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--ckpt-dir", default=DEFAULT_CKPT_DIR)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    surrogates = get_surrogates(args.archs, args.seeds, device=device, ckpt_dir=args.ckpt_dir,
                                data_root=args.data_root, epochs=args.epochs, lr=args.lr)

    _, test_loader = _loaders(args.data_root)
    print("\n=== clean test accuracy ===")
    for (arch, seed), model in surrogates.items():
        print(f"{arch} seed {seed}: {accuracy(model, test_loader, device):.4f}")
