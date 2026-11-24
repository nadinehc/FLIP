"""
Unit tests for the FLIP+G extension: the perturbation generator, the mask
helpers, the perturbed-image dataset, and the defense scores. CPU only, no
dataset download, runs in seconds.

    python sweep/tests/test_extension.py

`modules.base_utils.datasets` imports torchvision at module scope, so if
torchvision is missing a minimal stub is installed to let these tests run
anyway. Nothing here depends on real torchvision behaviour.
"""

import shutil
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "modules"))


def install_torchvision_stub():
    tv = types.ModuleType("torchvision")
    transforms = types.ModuleType("torchvision.transforms")
    datasets = types.ModuleType("torchvision.datasets")

    class _T:
        def __init__(self, *a, **k):
            self.a, self.k = a, k

        def __call__(self, x):
            return x

    for name in ["Normalize", "Compose", "RandomCrop", "RandomHorizontalFlip",
                 "ToTensor", "Resize", "CenterCrop"]:
        setattr(transforms, name, _T)
    for name in ["CIFAR10", "CIFAR100", "ImageFolder"]:
        setattr(datasets, name, _T)

    tv.transforms = transforms
    tv.datasets = datasets
    sys.modules["torchvision"] = tv
    sys.modules["torchvision.transforms"] = transforms
    sys.modules["torchvision.datasets"] = datasets


try:
    import torchvision  # noqa: F401
except ImportError:
    install_torchvision_stub()

from modules.base_utils.image_noise import (  # noqa: E402
    NoiseGenerator, PerturbedImageDataset, coalesce_noise_config,
    flip_candidate_mask, flipped_indices, get_normalization, load_generator,
    save_generator,
)
from modules.run_defenses.utils import (  # noqa: E402
    DEFENSES, detection_metrics, kmeans_scores, pca_scores, spectre_scores,
)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))


def random_norm_batch(n=8, c=3, s=32, dataset="cifar"):
    """A batch of valid images, in normalized space."""
    mean, std = get_normalization(dataset)
    x_pix = torch.rand(n, c, s, s)
    mean = torch.tensor(mean).view(1, -1, 1, 1)
    std = torch.tensor(std).view(1, -1, 1, 1)
    return (x_pix - mean) / std, x_pix


print("\n== config coalescing ==")
cfg = coalesce_noise_config({"epsilon_255": 8})
check("epsilon_255 converts to pixel units", abs(cfg["epsilon"] - 8 / 255) < 1e-12,
      f"eps={cfg['epsilon']:.6f}")
check("defaults fill in", cfg["norm"] == "linf" and cfg["width"] == 32)
try:
    coalesce_noise_config({"epsilon": 0})
    check("rejects eps<=0", False)
except ValueError:
    check("rejects eps<=0", True)
try:
    coalesce_noise_config({"norm": "l7"})
    check("rejects unknown norm", False)
except ValueError:
    check("rejects unknown norm", True)


print("\n== generator: L-infinity bound and feasibility ==")
for eps255 in [1, 2, 4, 8]:
    eps = eps255 / 255
    g = NoiseGenerator(dataset_flag="cifar", epsilon=eps, norm="linf")
    # Push the generator away from its near-zero init so the bound is tested
    # against a generator that actually wants to produce large perturbations.
    with torch.no_grad():
        for p in g.parameters():
            p.add_(torch.randn_like(p) * 0.5)

    x_norm, x_pix = random_norm_batch()
    delta_pix = g.delta_pixel(x_pix, quantize=False)
    adv = x_pix + delta_pix

    check(f"eps={eps255}/255 respects Linf bound",
          delta_pix.abs().max().item() <= eps + 1e-6,
          f"max|d|={delta_pix.abs().max().item() * 255:.3f}/255")
    check(f"eps={eps255}/255 keeps image in [0,1]",
          adv.min().item() >= -1e-6 and adv.max().item() <= 1 + 1e-6,
          f"range=[{adv.min().item():.4f}, {adv.max().item():.4f}]")

    delta_norm = g(x_norm, quantize=False)
    expected = delta_pix / g.std
    # Tolerance is float32 slack: forward() recovers x_pix from x_norm, and
    # that roundtrip error is divided by the per-channel std (~0.2), so it is
    # amplified ~5x and grows with epsilon.
    check(f"eps={eps255}/255 normalized delta matches pixel delta / std",
          torch.allclose(delta_norm, expected, atol=1e-5),
          f"max diff={(delta_norm - expected).abs().max().item():.2e}")


print("\n== generator: L2 bound ==")
g2 = NoiseGenerator(dataset_flag="cifar", epsilon=1.0, norm="l2")
with torch.no_grad():
    for p in g2.parameters():
        p.add_(torch.randn_like(p) * 0.5)
x_norm, x_pix = random_norm_batch()
d = g2.delta_pixel(x_pix, quantize=False)
norms = d.flatten(1).norm(dim=1)
check("L2 delta within bound", norms.max().item() <= 1.0 + 1e-5,
      f"max||d||_2={norms.max().item():.4f}")


print("\n== generator: quantization ==")
g = NoiseGenerator(dataset_flag="cifar", epsilon=8 / 255, norm="linf")
with torch.no_grad():
    for p in g.parameters():
        p.add_(torch.randn_like(p) * 0.5)
x_norm, x_pix = random_norm_batch()
dq = g.delta_pixel(x_pix, quantize=True)
steps = dq * 255
# After clamping to [-x, 1-x] the values can land off-grid only where the
# clamp bound itself is off-grid, which cannot happen for 8-bit inputs.
x_pix_8bit = torch.round(x_pix * 255) / 255
dq8 = g.delta_pixel(x_pix_8bit, quantize=True)
check("quantized delta lies on the 1/255 grid for 8-bit inputs",
      torch.allclose(dq8 * 255, torch.round(dq8 * 255), atol=1e-4),
      f"max off-grid={(dq8 * 255 - torch.round(dq8 * 255)).abs().max().item():.2e}")


print("\n== generator: gradients flow ==")
g = NoiseGenerator(dataset_flag="cifar", epsilon=8 / 255, norm="linf")
x_norm, _ = random_norm_batch()
delta = g(x_norm)
loss = (delta ** 2).sum()
loss.backward()
grad_norm = sum(p.grad.abs().sum().item() for p in g.parameters() if p.grad is not None)
check("gradients reach generator parameters", grad_norm > 0, f"sum|grad|={grad_norm:.4g}")
check("all parameters receive gradients",
      all(p.grad is not None for p in g.parameters()))


print("\n== generator: init starts near vanilla FLIP ==")
g = NoiseGenerator(dataset_flag="cifar", epsilon=8 / 255, norm="linf")
_, x_pix = random_norm_batch()
d0 = g.delta_pixel(x_pix, quantize=False).abs().max().item()
check("initial perturbation is tiny", d0 * 255 < 1.0, f"max|d|={d0 * 255:.4f}/255")


print("\n== generator: save / load roundtrip ==")
tmp = Path(tempfile.mkdtemp(prefix="flip_gen_roundtrip_"))
g = NoiseGenerator(dataset_flag="cifar", epsilon=4 / 255, norm="linf", width=16)
with torch.no_grad():
    for p in g.parameters():
        p.add_(torch.randn_like(p) * 0.1)
save_generator(g, str(tmp))
g_loaded = load_generator(str(tmp / "generator.pth"))
x_norm, x_pix = random_norm_batch()
check("reloaded generator has same epsilon/norm/width",
      g_loaded.epsilon == g.epsilon and g_loaded.norm == g.norm
      and g_loaded.width == g.width)
with torch.no_grad():
    check("reloaded generator reproduces the perturbation",
          torch.allclose(g.delta_pixel(x_pix), g_loaded.delta_pixel(x_pix), atol=1e-6))


print("\n== mask helpers ==")
labels = np.array([9, 4, 9, 1, 4, 9, 2])
m = flip_candidate_mask(labels, source_label=9, target_label=4)
check("targeted mask selects source class only",
      m.tolist() == [True, False, True, False, False, True, False])
m_all = flip_candidate_mask(labels, source_label=-1, target_label=4)
check("source=-1 selects every non-target example",
      m_all.tolist() == [True, False, True, True, False, True, True])

true = np.eye(10)[np.array([9, 9, 9, 4, 1])]
pois = true.copy()
pois[1] = np.eye(10)[4]
pois[4] = np.eye(10)[7]
check("flipped indices recovered from label arrays",
      flipped_indices(pois, true).tolist() == [1, 4])
check("no flips recovered from identical arrays",
      flipped_indices(true, true).tolist() == [])


print("\n== PerturbedImageDataset ==")


class FakeRawDataset:
    """Stands in for torchvision's CIFAR10: returns (PIL image, int label)."""

    def __init__(self, n=40, seed=0):
        rng = np.random.RandomState(seed)
        self.images = [
            Image.fromarray(rng.randint(0, 256, (32, 32, 3), dtype=np.uint8))
            for _ in range(n)
        ]
        self.labels = [int(rng.randint(10)) for _ in range(n)]

    def __getitem__(self, i):
        return self.images[i], self.labels[i]

    def __len__(self):
        return len(self.images)


raw = FakeRawDataset()
eps255 = 8
g = NoiseGenerator(dataset_flag="cifar", epsilon=eps255 / 255, norm="linf")
with torch.no_grad():
    for p in g.parameters():
        p.add_(torch.randn_like(p) * 0.5)

flip_idx = [3, 7, 11, 19]
pert = PerturbedImageDataset(raw, g, flip_idx, device=torch.device("cpu"))

check("length preserved", len(pert) == len(raw))
check("labels preserved", all(pert[i][1] == raw[i][1] for i in range(len(raw))))
check("perturbed items are PIL images",
      all(isinstance(pert[i][0], Image.Image) for i in flip_idx))

untouched = [i for i in range(len(raw)) if i not in flip_idx]
check("non-flipped images are bit-identical",
      all(np.array_equal(np.asarray(pert[i][0]), np.asarray(raw[i][0]))
          for i in untouched))
changed = [i for i in flip_idx
           if not np.array_equal(np.asarray(pert[i][0]), np.asarray(raw[i][0]))]
check("flipped images actually changed", len(changed) == len(flip_idx),
      f"{len(changed)}/{len(flip_idx)} changed")

stats = pert.perturbation_stats()
check("realized Linf respects epsilon after 8-bit rounding",
      stats["linf_max_255"] <= eps255,
      f"linf_max={stats['linf_max_255']:.4f}/255 (eps={eps255}/255)")
check("stats count matches flip count", stats["n_perturbed"] == len(flip_idx))

empty = PerturbedImageDataset(raw, g, [], device=torch.device("cpu"))
check("empty flip set is a no-op",
      empty.perturbation_stats()["n_perturbed"] == 0
      and np.array_equal(np.asarray(empty[0][0]), np.asarray(raw[0][0])))


print("\n== defenses: planted signal ==")
rng = np.random.RandomState(0)
n_clean, n_poison, dim = 500, 50, 64
direction = rng.randn(dim)
direction /= np.linalg.norm(direction)
clean = rng.randn(n_clean, dim)
poison = rng.randn(n_poison, dim) + 4.0 * direction
features = np.vstack([clean, poison])
mask = np.zeros(len(features), dtype=bool)
mask[n_clean:] = True

for name, fn in DEFENSES.items():
    scores = fn(features, n_poison)
    met = detection_metrics(scores, mask)
    check(f"{name} returns one score per example", len(scores) == len(features))
    check(f"{name} detects a strongly planted signal",
          met["auroc"] is not None and met["auroc"] > 0.9,
          f"auroc={met['auroc']:.4f}, detect={met['detection_rate']:.3f}")
    check(f"{name} budget is 1.5x the poison count",
          met["budget"] == int(np.ceil(1.5 * n_poison)))

print("\n== defenses: no planted signal ==")
features_null = rng.randn(550, dim)
for name, fn in DEFENSES.items():
    scores = fn(features_null, n_poison)
    met = detection_metrics(scores, mask)
    check(f"{name} is near chance with no signal",
          0.25 < met["auroc"] < 0.75, f"auroc={met['auroc']:.4f}")

print("\n== defenses: metrics edge cases ==")
met = detection_metrics(np.random.randn(100), np.zeros(100, dtype=bool))
check("no poisons -> None metrics rather than a crash",
      met["detection_rate"] is None and met["n_poison"] == 0)
met = detection_metrics(np.random.randn(100), np.ones(100, dtype=bool))
check("all poisons -> None metrics rather than a crash",
      met["detection_rate"] is None)
met = detection_metrics(np.zeros(100), np.array([True] * 10 + [False] * 90))
check("constant scores give AUROC 0.5 via midranks",
      abs(met["auroc"] - 0.5) < 1e-9, f"auroc={met['auroc']}")

perfect = np.array([0.0] * 90 + [1.0] * 10)
met = detection_metrics(perfect, np.array([False] * 90 + [True] * 10))
check("perfectly separable scores give AUROC 1.0",
      abs(met["auroc"] - 1.0) < 1e-9 and met["detection_rate"] == 1.0)

print("\n== defenses: small-sample robustness ==")
for n in [5, 12, 30]:
    f = rng.randn(n, 64)
    m = np.zeros(n, dtype=bool)
    m[: max(1, n // 10)] = True
    for name, fn in DEFENSES.items():
        try:
            s = fn(f, int(m.sum()))
            ok = len(s) == n and np.isfinite(s).all()
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"       {name} n={n} raised {type(e).__name__}: {e}")
        check(f"{name} handles n={n}", ok)

shutil.rmtree(tmp, ignore_errors=True)

print("\n" + "=" * 60)
print(f"{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED: " + ", ".join(FAIL))
sys.exit(1 if FAIL else 0)
