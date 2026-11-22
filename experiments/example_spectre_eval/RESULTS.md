# SPECTRE evaluation results — `example_spectre_eval`

## Setup

| Field | Value |
|-------|--------|
| Model | `experiments/flip_c10_1xu/user_150/0/model.pth` |
| Labels | `experiments/flip_c10_1xu/150.npy` |
| True labels | `experiments/flip_c10_1xu/true.npy` |
| Architecture | ResNet-32 (`r32p`) |
| Dataset | CIFAR-10 |
| Trigger | `1xu` (universal), `delta.npy` |
| Source / target | 9 (truck) → 4 (deer) |
| Flip budget \(m\) | 150 |
| Backend | Python SPECTRE port (`backend = "python"`) |
| Representation layer | 14 |
| Retrain after filter | no |

Evaluated representations of all training examples currently labeled as **target = 4**.

## Attack strength (before filtering)

| Metric | Value |
|--------|--------|
| CTA (clean test accuracy) | **92.44%** |
| PTA (poison / triggered test accuracy) | **97.70%** |

The user model looks healthy on clean data while the backdoor fires reliably. This is a successful FLIP-style label-only attack at budget 150.

## What SPECTRE saw

| Quantity | Count |
|----------|------:|
| Clean target examples | 5000 |
| Flips into target (`eps`) | **128** |
| Total target-labeled | 5128 |
| Removal budget (\(1.5 \times eps\)) | 192 |

Only **128 / 150** hard flips land on class 4; the remaining 22 flips go to other classes and are invisible to target-class SPECTRE.

Artifacts written under this folder:

- `label_4_reps.npy` — layer-14 representations (ordered `[clean target | flips-to-target]`)
- `ordered_indices.npy`, `flip_to_target_indices.npy`
- `mask-{pca,kmeans,rcov}-target.npy` — True = remove
- `spectre_metrics.json` — numeric summary

## Filter results

| Defense | Poisons removed | Clean removed | Poison recall |
|---------|----------------:|--------------:|--------------:|
| SPECTRE (`rcov`) | **0 / 128** | 192 | **0.0%** |
| PCA | **0 / 128** | 192 | **0.0%** |
| kmeans | 4 / 128 | 188 | **3.1%** |

All three filters mostly delete **clean** target examples. They do **not** catch the flipped subset in any meaningful way.

## Interpretation

1. **Defenses fail on this checkpoint.** SPECTRE, PCA, and kmeans do not mitigate the attack here (poison recall ≈ 0). With `retrain = false`, post-filter CTA/PTA were not measured; given zero poison recall, retraining on the kept set would not be expected to remove the backdoor.

2. **Why this can differ from the FLIP paper (Appendix D.2).** Several factors weaken the spectral signature relative to the paper’s SPECTRE-wins setting:
   - Small flip budget (\(m = 150\); only 128 target flips).
   - Label-only poisoning with a `1xu` trigger (paper tables emphasize sinusoidal / Turner / pixel).
   - Python SPECTRE port rather than the official Julia `spectre-defense` stack (Julia 1.12 was incompatible with the vendored deps).

3. **Practical takeaway.** For this `flip_c10_1xu` user_150 run, representation-based filtering is ineffective. Useful follow-ups:
   - Larger budgets (`300`, `500`, `1000`, `1500`) on the same experiment family.
   - A vanilla `1xs` FLIP user model, closer to the paper setting.
   - Optional `retrain = true` once a filter shows non-trivial poison recall.
   - Official Julia SPECTRE backend if a compatible Julia environment is available (`backend = "julia"`).

## Source metrics

Raw numbers are in [`spectre_metrics.json`](spectre_metrics.json).
