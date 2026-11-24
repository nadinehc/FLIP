# FLIP with a learned ViT trigger: recap, part 3 (six seeds)

This continues `RECAP.md` (first sweep, `trig`) and `RECAP_2.md` (adaptive sweep, `adapt`). Those had one run per setting. This part reruns the adaptive configuration with six seeds, so every comparison has an estimate of its run-to-run spread.

All numbers come from `out/sweep/adapt_s{0,1,2}/` and `out/sweep/adaptfix_s{3,4,5}/` (`results.csv`, `flip_table.md`, logs). The statistics are from `sweep/aggregate_seeds.py` (`out/sweep/seed_summary_all6.md`), and the curves are from `sweep/plot_training_curves.py` (`out/sweep/training_curves_all6/`).

> **Read this first: seeds 0–2 and seeds 3–5 are not the same procedure.**
> Seeds 0–2 ran with a seeding bug. Every process in a run got the same seed, so the four experts of each setting started from the same random state and were identical. Each of those runs effectively used one expert trajectory. Seeds 3–5 ran after the fix, and their four experts differ within each setting.
>
> All six seeds are merged below, as requested, and the three unfixed seeds are marked "(unfixed)" in every table and drawn with dashed lines in every figure. The merged statistics mix the two procedures. The cleanest fix is to rerun seeds 0–2 with the same fixed code (about 8 hours on Quetzal), so that all six seeds share one procedure. The unfixed seeds are still useful as a check on whether the conclusions depend on the bug.

## 1. Aim

FLIP (Jha, Hayase, Oh, NeurIPS 2023) is label-only: the victim trains on clean images with flipped labels, and trajectory matching against experts trained on triggered data decides which labels to flip. The paper's trigger is a fixed sinusoid (`1xs`).

Question: can a learned trigger keep the attack's effect (PTA, with CTA unharmed) while making the flipped set harder for detectors (kmeans, PCA, SPECTRE) to find? Lower detection is better for the attacker.

The specific question of this part: **are the differences seen in the single-seed sweeps larger than seed-to-seed noise?**

## 2. Approach

Fixed throughout: CIFAR-10, ResNet-32 (`r32p`), SGD, source class 9 (truck), target class 4 (deer), m = 1500 flipped labels, 4 experts of 20 epochs, victim trained 60 epochs.

**Triggers compared**

| Name | Description |
|---|---|
| `1xs` | The paper's sinusoidal stripes (hand-coded baseline). |
| `vit_eps4`, `vit_eps8` | A ViT generator, `delta = epsilon * tanh(G(x))`, epsilon 4/255 and 8/255, trained from random initialization (no sinusoid is used as a starting point). |

Both ViT triggers use the adaptive options from RECAP_2: `--adaptive-floor 0.25` (the per-pixel bound is scaled by a texture map of the clean image) and `--lambda-div 1.0 --div-tau 0.3` (a penalty on the cosine similarity between different images' perturbations above 0.3), with 20 trigger epochs.

**Seeding.** A `--seed` flag in `sweep/run_trigger_sweep.py` sets `FLIP_SEED`. `run_experiment.py` reads it and seeds Python, NumPy and PyTorch in each step (trigger, experts, attack, victim), with cuDNN set to deterministic.

**The bug and its fix.** The first version gave every process the same seed, so the four experts of a setting were identical (seeds 0–2). `run_experiment.py` now seeds each experiment with `util.derive_seed(base_seed, experiment_name)`, which mixes a CRC32 of the experiment name into the seed. Seeds 3–5 were run with this, and their four experts differ within each setting. The expert accuracies in the logs show it; for example `vit_eps4` in seed 4 ranges from 0.06 to 0.92.

**Scripts**

```bash
bash sweep/run_seeds.sh                                  # runs seeds on Quetzal, tags adaptfix_s<n>
python sweep/aggregate_seeds.py --tags adapt_s0 adapt_s1 adapt_s2 adaptfix_s3 adaptfix_s4 adaptfix_s5 --suffix _all6
python sweep/plot_training_curves.py --tags adapt_s0 adapt_s1 adapt_s2 adaptfix_s3 adaptfix_s4 adaptfix_s5 --unfixed adapt_s0 adapt_s1 adapt_s2 --out-dir training_curves_all6
```

Note: a same-seed-twice check of the whole pipeline has not been seen, so the reproducibility of a given seed is not verified end to end.

## 3. Results per seed

CTA, PTA and detection in %. Surrogate ASR and cross-image cosine are fractions. Lower detection is better for the attacker.

| Trigger | Seed | CTA | PTA | Surrogate ASR | Cross-image cos | kmeans det | PCA det | SPECTRE det |
|---|---|---|---|---|---|---|---|---|
| `1xs` | 0 (unfixed) | 84.81 | 98.20 | -- | -- | 97.55 | 84.02 | 94.86 |
| `1xs` | 1 (unfixed) | 84.83 | 99.00 | -- | -- | 98.30 | 87.26 | 94.97 |
| `1xs` | 2 (unfixed) | 85.94 | 94.90 | -- | -- | 97.03 | 78.29 | 89.74 |
| `1xs` | 3 | 86.16 | 99.90 | -- | -- | 97.03 | 79.64 | 91.43 |
| `1xs` | 4 | 83.52 | 99.90 | -- | -- | 98.75 | 86.75 | 91.25 |
| `1xs` | 5 | 79.58 | 93.20 | -- | -- | 97.32 | 82.97 | 90.59 |
| `vit_eps4` | 0 (unfixed) | 83.62 | 99.90 | 59.9 | 0.874 | 98.37 | 83.89 | 80.03 |
| `vit_eps4` | 1 (unfixed) | 85.25 | 99.80 | 56.4 | 0.864 | 95.72 | 84.03 | 90.26 |
| `vit_eps4` | 2 (unfixed) | 86.66 | 28.60 | 24.8 | 0.492 | 99.85 | 96.15 | 99.13 |
| `vit_eps4` | 3 | 84.15 | 88.90 | 26.4 | 0.727 | 93.38 | 79.46 | 82.25 |
| `vit_eps4` | 4 | 84.37 | 16.40 | 32.5 | 0.889 | 100.00 | 99.80 | 99.93 |
| `vit_eps4` | 5 | 85.75 | 39.60 | 52.7 | 0.886 | 99.93 | 99.47 | 99.93 |
| `vit_eps8` | 0 (unfixed) | 83.11 | 62.20 | 61.3 | 0.900 | 99.25 | 87.18 | 95.70 |
| `vit_eps8` | 1 (unfixed) | 85.15 | 99.90 | 40.3 | 0.900 | 0.71 | 50.00 | 56.16 |
| `vit_eps8` | 2 (unfixed) | 83.71 | 100.00 | 66.1 | 0.899 | 81.95 | 65.76 | 78.22 |
| `vit_eps8` | 3 | 83.56 | 99.60 | 24.1 | 0.900 | 86.47 | 73.07 | 81.52 |
| `vit_eps8` | 4 | 83.97 | 100.00 | 40.8 | 0.858 | **0.11** | 59.42 | 50.44 |
| `vit_eps8` | 5 | 86.62 | 100.00 | 34.9 | 0.515 | 88.87 | 72.18 | 77.74 |

Mean ± std over the six seeds, in % (full table with SEM, 95% CI and paired differences in Section 6):

| Trigger | CTA | PTA | kmeans det | PCA det | SPECTRE det |
|---|---|---|---|---|---|
| `1xs` | 84.14 ± 2.43 | 97.52 ± 2.81 | 97.66 ± 0.71 | 83.15 ± 3.65 | 92.14 ± 2.23 |
| `vit_eps4` | 84.97 ± 1.13 | 62.20 ± 38.17 | 97.88 ± 2.75 | 90.47 ± 9.01 | 91.92 ± 9.14 |
| `vit_eps8` | 84.35 ± 1.30 | 93.62 ± 15.39 | 59.56 ± 46.17 | 67.94 ± 12.76 | 73.30 ± 16.91 |

The `vit_eps4` PTA mean is not a useful summary, because the values split into two groups (Section 10). The `vit_eps8` kmeans mean is dominated by the two runs near zero.

## 4. Flipped labels

**Class numbers.** CIFAR-10 uses its standard label order:

| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| airplane | automobile | bird | cat | deer | dog | frog | horse | ship | truck |

So the source class 9 is **truck** and the target class 4 is **deer**. In the flip tables, "9→4" is a truck image labelled deer, "9→1" is truck→automobile, "9→8" is truck→ship and "9→0" is truck→airplane.

Share of the 1500 flips that are truck→deer (9→4), from each seed's `flip_table.md`:

| Trigger | 0 (unfixed) | 1 (unfixed) | 2 (unfixed) | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| `1xs` | 62.5% | 64.3% | 51.2% | 53.2% | 62.1% | 63.4% |
| `vit_eps4` | 46.2% | 49.5% | **87.2%** | 51.2% | **100.0%** | **100.0%** |
| `vit_eps8` | 41.1% | 12.1% | 19.6% | 34.2% | 18.5% | 22.7% |

For `vit_eps4`, the three runs with low PTA (seeds 2, 4 and 5) are the three runs where 87–100% of the flips went to the target. The three runs with high PTA (seeds 0, 1 and 3) had 46–51%. This is a clean split across six runs, which is a correlation and not a tested cause (Section 10).

The "all flips to class 4" count (`n_poison_in_target` in `results.csv`) follows the same pattern for `vit_eps4`: 1500 of 1500 in seeds 4 and 5, against 1027–1376 in the other runs.

Other off-target flips of truck images go mostly to the visually closest classes: automobile (1), ship (8) and airplane (0). In `vit_eps8` seed 1 the largest off-target group is truck→automobile (17.2%), and in seed 2 it is truck→automobile at 23.6%.

## 5. Trigger images

**What the trigger is meant to do:** a triggered image of a **truck** (source class 9) should be classified as a **deer** (target class 4) by the poisoned model, while the trigger leaves the clean accuracy unchanged. Each row below is a random truck from the training set.

Six random source-class (class 9, trucks) training images per trigger: the original, the triggered image, the perturbation at true scale, and the perturbation amplified (the amplification factor is printed above the last column). The first two columns are at the same scale with no enhancement.

**These figures are from the earlier single-run `adapt` sweep (`out/sweep/adapt/`), not from the six seeds.** The generators trained in the seeded runs are in `experiments/sweep_adaptfix_s*_trigger_*` and `experiments/sweep_adapt_s*_trigger_*` on Quetzal and on the Mac, and they have not been rendered yet. The figure script can render them: `python sweep/make_figures.py --generator <path to generator.pth>`.

`vit_eps8` (adaptive):

![adapt vit_eps8](out/sweep/adapt/figures/vit_eps8/trigger_examples.png)

`vit_eps4` (adaptive):

![adapt vit_eps4](out/sweep/adapt/figures/vit_eps4/trigger_examples.png)

As described in RECAP_2: the amplified perturbation is dominated by a vertical stripe pattern that looks alike on every image (consistent with a cosine near 0.9), with larger amplitude on edges and texture. At true scale it looks like flat grey with faint vertical lines, most visible on smooth regions at epsilon 8. Visibility was judged by eye only.

The seeds with the lowest cosine (`vit_eps8` seed 5 at 0.52, `vit_eps4` seed 2 at 0.49) are the ones to render first, because they are the least uniform triggers.

## 6. Error bars

### Method

For each trigger and metric, with the six per-seed values x1…x6 (n = 6):

- mean = (x1 + … + x6) / n
- sample standard deviation s = sqrt( sum (xi − mean)² / (n − 1) )
- standard error of the mean SEM = s / sqrt(n)
- 95% confidence interval = mean ± t × SEM, with t = 2.571 (two-sided 97.5% Student-t quantile for n − 1 = 5 degrees of freedom)

**Worked example, `1xs` PCA detection** (values 84.02, 87.26, 78.29, 79.64, 86.75, 82.97):

- mean = 498.93 / 6 = 83.15
- deviations: +0.87, +4.11, −4.87, −3.52, +3.60, −0.19; squares sum to 66.58
- s = sqrt(66.58 / 5) = sqrt(13.32) = 3.65
- SEM = 3.65 / sqrt(6) = 1.49
- 95% CI = 83.15 ± 2.571 × 1.49 = 83.15 ± 3.83 = [79.32, 86.98]

**Paired differences.** Within a seed, a ViT trigger and `1xs` share the same seed, so the difference `ViT − 1xs` is computed per seed and then summarized with the same formulas. This removes part of the seed-level variation. Worked example, `vit_eps8` SPECTRE minus `1xs`: per-seed differences +0.84, −38.81, −11.51, −9.90, −40.81, −12.85; mean = −113.04 / 6 = −18.84; s = 16.96; SEM = 6.92; 95% CI = −18.84 ± 2.571 × 6.92 = [−36.6, −1.0].

The intervals for detection rates that run above 100% are a limit of the t-interval for a bounded quantity, and they should be read as rough.

### Full table (%)

| Trigger | Metric | 0 (u) | 1 (u) | 2 (u) | 3 | 4 | 5 | mean | std | SEM | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1xs | CTA | 84.81 | 84.83 | 85.94 | 86.16 | 83.52 | 79.58 | 84.14 | 2.43 | 0.99 | [81.59, 86.69] |
| 1xs | PTA | 98.20 | 99.00 | 94.90 | 99.90 | 99.90 | 93.20 | 97.52 | 2.81 | 1.15 | [94.57, 100.47] |
| 1xs | kmeans det | 97.55 | 98.30 | 97.03 | 97.03 | 98.75 | 97.32 | 97.66 | 0.71 | 0.29 | [96.92, 98.41] |
| 1xs | PCA det | 84.02 | 87.26 | 78.29 | 79.64 | 86.75 | 82.97 | 83.15 | 3.65 | 1.49 | [79.32, 86.98] |
| 1xs | SPECTRE det | 94.86 | 94.97 | 89.74 | 91.43 | 91.25 | 90.59 | 92.14 | 2.23 | 0.91 | [89.80, 94.48] |
| vit_eps4 | CTA | 83.62 | 85.25 | 86.66 | 84.15 | 84.37 | 85.75 | 84.97 | 1.13 | 0.46 | [83.78, 86.15] |
| vit_eps4 | PTA | 99.90 | 99.80 | 28.60 | 88.90 | 16.40 | 39.60 | 62.20 | 38.17 | 15.58 | [22.13, 102.27] |
| vit_eps4 | kmeans det | 98.37 | 95.72 | 99.85 | 93.38 | 100.00 | 99.93 | 97.88 | 2.75 | 1.12 | [94.99, 100.76] |
| vit_eps4 | PCA det | 83.89 | 84.03 | 96.15 | 79.46 | 99.80 | 99.47 | 90.47 | 9.01 | 3.68 | [81.01, 99.93] |
| vit_eps4 | SPECTRE det | 80.03 | 90.26 | 99.13 | 82.25 | 99.93 | 99.93 | 91.92 | 9.14 | 3.73 | [82.33, 101.52] |
| vit_eps4 | surrogate ASR | 0.599 | 0.564 | 0.248 | 0.264 | 0.325 | 0.527 | 0.421 | 0.159 | 0.065 | [0.254, 0.589] |
| vit_eps4 | cross-image cos | 0.874 | 0.864 | 0.492 | 0.727 | 0.889 | 0.886 | 0.789 | 0.158 | 0.064 | [0.623, 0.954] |
| vit_eps8 | CTA | 83.11 | 85.15 | 83.71 | 83.56 | 83.97 | 86.62 | 84.35 | 1.30 | 0.53 | [82.98, 85.72] |
| vit_eps8 | PTA | 62.20 | 99.90 | 100.00 | 99.60 | 100.00 | 100.00 | 93.62 | 15.39 | 6.28 | [77.46, 109.77] |
| vit_eps8 | kmeans det | 99.25 | 0.71 | 81.95 | 86.47 | 0.11 | 88.87 | 59.56 | 46.17 | 18.85 | [11.10, 108.02] |
| vit_eps8 | PCA det | 87.18 | 50.00 | 65.76 | 73.07 | 59.42 | 72.18 | 67.94 | 12.76 | 5.21 | [54.55, 81.33] |
| vit_eps8 | SPECTRE det | 95.70 | 56.16 | 78.22 | 81.52 | 50.44 | 77.74 | 73.30 | 16.91 | 6.90 | [55.55, 91.04] |
| vit_eps8 | surrogate ASR | 0.613 | 0.403 | 0.661 | 0.241 | 0.408 | 0.349 | 0.446 | 0.161 | 0.066 | [0.277, 0.614] |
| vit_eps8 | cross-image cos | 0.900 | 0.900 | 0.899 | 0.900 | 0.858 | 0.515 | 0.829 | 0.155 | 0.063 | [0.666, 0.991] |

(u) = unfixed seed.

### Paired difference against `1xs` (percentage points, same seed)

| Trigger | Metric | mean diff | std | 95% CI |
|---|---|---|---|---|
| vit_eps4 | CTA | +0.83 | 2.86 | [−2.17, +3.83] |
| vit_eps4 | PTA | −35.32 | 37.10 | [−74.26, +3.62] |
| vit_eps4 | kmeans det | +0.21 | 2.71 | [−2.63, +3.06] |
| vit_eps4 | PCA det | +7.31 | 9.50 | [−2.66, +17.28] |
| vit_eps4 | SPECTRE det | −0.22 | 10.74 | [−11.49, +11.06] |
| vit_eps8 | CTA | +0.21 | 3.58 | [−3.55, +3.97] |
| vit_eps8 | PTA | −3.90 | 15.99 | [−20.68, +12.88] |
| vit_eps8 | kmeans det | −38.10 | 46.81 | [−87.23, +11.03] |
| vit_eps8 | PCA det | −15.22 | 14.64 | [−30.59, +0.15] |
| vit_eps8 | SPECTRE det | −18.84 | 16.96 | [−36.64, −1.04] |

Per-seed differences, for the sign counts in Section 8: `vit_eps8` PCA is lower than `1xs` in 5 of 6 seeds (exception: seed 0), and SPECTRE is lower in 5 of 6 (exception: seed 0). `vit_eps4` PCA is lower in 3 of 6 seeds (0, 1 and 3) and SPECTRE in 3 of 6 (0, 1 and 3).

## 7. Training curves

Per-epoch curves for all six seeds: one colour and marker per seed, with dashed lines for the three unfixed seeds (0–2). The black line is the mean over six seeds and the grey band is ±1 sample standard deviation across seeds. The values come from the progress-bar lines of the step logs. In the last victim epoch, `acc0` equals the CTA and `acc1` the PTA of `results.csv`. Every plotted value is in `out/sweep/training_curves_all6/curves.csv`.

### Victim

![victim curves](out/sweep/training_curves_all6/victim_curves.png)

- **CTA** (top row): all 18 curves overlap and rise to about 84–86% by epoch 60, so CTA does not separate the triggers or the seeds.
- **PTA** (bottom row): `1xs` is at 93–100% for every seed at epoch 60. For `vit_eps8`, five seeds are near 100% and seed 0 oscillates. For `vit_eps4` the seeds split into two groups, and seeds 4 and 5 oscillate.

Over the last 10 epochs, PTA at epoch 60, with min, max and std:

| Trigger | Seed | Epoch 60 | Min | Max | Std |
|---|---|---|---|---|---|
| `1xs` | 0 (u) | 98.2 | 87.3 | 99.8 | 3.8 |
| `1xs` | 1 (u) | 99.0 | 98.0 | 99.9 | 0.6 |
| `1xs` | 2 (u) | 94.9 | 94.9 | 99.8 | 1.5 |
| `1xs` | 3 | 99.9 | 99.6 | 100.0 | 0.1 |
| `1xs` | 4 | 99.9 | 95.1 | 99.9 | 1.7 |
| `1xs` | 5 | 93.2 | 93.2 | 99.5 | 2.2 |
| `vit_eps4` | 0 (u) | 99.9 | 99.2 | 100.0 | 0.3 |
| `vit_eps4` | 1 (u) | 99.8 | 99.8 | 100.0 | 0.1 |
| `vit_eps4` | 2 (u) | 28.6 | 9.6 | 51.8 | 11.4 |
| `vit_eps4` | 3 | 88.9 | 88.9 | 100.0 | 3.6 |
| `vit_eps4` | 4 | 16.4 | 16.4 | 60.6 | 12.2 |
| `vit_eps4` | 5 | 39.6 | 5.6 | 64.2 | 18.1 |
| `vit_eps8` | 0 (u) | 62.2 | 21.4 | 81.6 | 22.3 |
| `vit_eps8` | 1 (u) | 99.9 | 99.9 | 100.0 | 0.0 |
| `vit_eps8` | 2 (u) | 100.0 | 99.9 | 100.0 | 0.0 |
| `vit_eps8` | 3 | 99.6 | 92.7 | 100.0 | 2.2 |
| `vit_eps8` | 4 | 100.0 | 99.9 | 100.0 | 0.0 |
| `vit_eps8` | 5 | 100.0 | 100.0 | 100.0 | 0.0 |

The reported PTA is the epoch-60 value. For the oscillating runs (`vit_eps4` seeds 2, 4 and 5, and `vit_eps8` seed 0), the same run could have reported a value 20–50 points higher or lower a few epochs earlier or later. The stable runs are representative.

### Experts

![expert curves](out/sweep/training_curves_all6/expert_curves.png)

The experts learn every trigger. Their accuracy on triggered test images is close to 100% from about epoch 8 in the seeds where the trigger transfers, and mostly 95–100% with dips to about 85–90% otherwise. For `vit_eps8` it is about 97–100% from the first epoch. The seeds 0–2 curves show one expert per seed, because their four experts were identical. The seeds 3–5 curves are for expert 0 of each seed; their other experts differ (Section 2).

### Attack

![attack curves](out/sweep/training_curves_all6/attack_curves.png)

The loss of the label optimization, averaged over bins of 0.1 epoch over its 15 epochs, with the same vertical scale in the three panels. The curves are noisy and not monotone for any trigger. `vit_eps4` seed 2 (unfixed) has the lowest loss of the six `vit_eps4` runs, about 0.2–0.3 against 0.4–0.6 for the others, even though its attack failed. I have not explained this.

### Trigger training (ViT only)

![trigger curves](out/sweep/training_curves_all6/trigger_curves.png)

There is no trigger training for `1xs`. For the ViT triggers, over the 19 generator epochs after the warmup epoch:

- **Backdoor loss** (top row): it rises over most of training in most seeds instead of falling, so the generator does not converge within 20 epochs.
- **Cross-image cosine** (bottom row, measured on the training batches): for `vit_eps8` it stays around 0.1–0.3 and then jumps to about 0.85. In seeds 0–2 the jump happens at a different epoch in each seed, and in seed 1 it is still rising at the end. For `vit_eps4` the training cosine is mostly low (0.05–0.4), while the cosine measured on the final trigger is 0.49–0.89.

## 8. What the results support

Six seeds is still a small sample, and the two procedures are mixed. With that caveat:

- **`1xs` is stable.** Its PTA is 97.5 ± 2.8 and SPECTRE detection is 92.1 ± 2.2 across six seeds. The CTA spread (79.6 to 86.2) is the largest of any trigger's.
- **`vit_eps8` transfers reliably.** Five of six seeds have PTA of 99.6 or more. The exception is seed 0 (62.2, unfixed), and its curve oscillates.
- **`vit_eps4` transfers in two groups.** Three seeds have PTA of 88.9 or more (seeds 0, 1 and 3) and three have PTA of 16–40 (seeds 2, 4 and 5). The mean of 62 ± 38 describes neither group. The split is perfect in six runs, so it is a clear pattern, but still a small sample. Section 10 has the candidate explanation.
- **ε=8 may lower PCA and SPECTRE detection.** The mean paired difference against `1xs` is −15.2 (95% CI [−30.6, +0.2]) for PCA and −18.8 (95% CI [−36.6, −1.0]) for SPECTRE. The SPECTRE interval excludes zero, but it is computed on a mix of unfixed and fixed seeds, and the sign of the difference is negative in five of six seeds. A sign test on five of six gives p ≈ 0.22, so the evidence is suggestive rather than conclusive.
- **ε=4 does not lower detection.** Its paired PCA difference is +7.3 (95% CI [−2.7, +17.3]), and its SPECTRE difference is −0.2 (95% CI [−11.5, +11.1]). The intervals include zero on every defense.
- **kmeans is unreliable for ε=8.** Two of six runs (seed 1 at 0.71%, seed 4 at 0.11%) give near-zero kmeans detection while PTA is at or near 100%. The other four give 81–89%. Its mean of 59.6 ± 46.2 should not be used until those two runs are checked.
- **CTA does not separate the triggers.** The three means are 84.1, 85.0 and 84.4, within about one standard deviation of each other.

## 9. What changing the seed changes

Each step (trigger training, each expert, the attack, the victim) is a separate process that starts by seeding Python, NumPy and PyTorch (`run_experiment.py`), with cuDNN set to deterministic mode. The seed therefore changes every random draw those steps make:

| Step | What differs between seeds |
|---|---|
| Trigger training | Initial weights of the ViT generator and of the surrogate ResNet; the order of the batches. This gives a different trigger per seed (surrogate ASR 24–66% in these runs). |
| Experts | Initial weights, batch order and data augmentation, so different training trajectories. (Identical across the four experts in seeds 0–2, see Section 2.) |
| Attack (label optimization) | Which expert and which checkpoint of its trajectory are sampled at each iteration, batch order and augmentation. |
| Victim | Initial weights, batch order and augmentation. |

What does **not** change with the seed:

- The CIFAR-10 data and its train/test split, the source and target classes (truck, deer), and all hyperparameters.
- The set of images the experts are poisoned with: it is picked by index (the whole class 9), with no random draw.
- The hand-coded `1xs` sinusoid itself, and the `seed=1` hard-coded in `PoisonedDataset` and `get_matching_datasets`.

Two things not checked:

- **The defenses' own randomness** (kmeans initialization, SPECTRE): whether it follows the seed is unknown. This may matter for the kmeans collapse in Section 8.
- **The attack's data pipeline.** `MTTDataset.__getitem__` draws `np.random.randint(8)` and re-seeds Python and PyTorch with it for every item. If that runs in the main process, it overwrites the global PyTorch seed, which may reduce the influence of our seed in the attack stage. Not tested.

## 10. Why the runs differ so much between seeds

Partly explained, partly open.

**Established or directly visible**

1. **Two groups of `vit_eps4` runs.** Three runs transfer (PTA 88.9–99.9, seeds 0, 1 and 3) and three do not (PTA 16.4, 28.6 and 39.6, seeds 4, 2 and 5). The three failing runs are also the three where 87–100% of the flips went to the target, against 46–51% in the working runs (Section 4). This is the clearest pattern in the six seeds. It is a correlation across six runs. A plausible reading is that when the flips concentrate on one pair, the victim learns the flipped labels without the trigger as a feature, but this has not been tested.
2. **Surrogate ASR does not predict PTA.** Among the `vit_eps4` runs the failing ones have surrogate ASR 24.8%, 32.5% and 52.7%, and the working ones 59.9%, 56.4% and 26.4%. There is no ordering.
3. **Trigger training does not converge in 20 epochs, and where it ends depends on the seed.** The backdoor loss often rises instead of falling, and the cosine jumps at a seed-dependent epoch (Section 7).
4. **Identical experts in seeds 0–2.** Every run in that group had a single expert trajectory, so the attack had no averaging over experts. This probably contributes to the variance of those three runs, but the rerun in Section 2 would show how much.

**Not explained**

- **The kmeans collapse for `vit_eps8`** (seeds 1 and 4). PTA is at or near 100% in both, and the PCA and SPECTRE values are 50–59%. The defense outputs have not been checked, so it is not known whether kmeans fails to flag the poisoned set or the flipped set is unusual.
- **The low `vit_eps4` expert accuracies in seeds 4 and 5.** The final accuracy of the four experts ranges from 0.06 to 0.92 in those seeds. That spread may be what drives the failure in point 1, but it has not been compared with PTA across all runs.
- **The mismatch between the training-time cosine and the final cosine** for `vit_eps4` (for example 0.41 against 0.87 in seed 0, and 0.055 against 0.86 in seed 1). The diversity penalty acts on the training value, so at epsilon 4 it may not constrain the trigger that is actually used. The final value is measured in evaluation mode on the quantized trigger. The cause has not been found.
- **Why `vit_eps8` seed 0 has PTA 62.2** while its surrogate ASR (61.3%) is in the normal range.

How to investigate: first check the defense outputs for the two kmeans-collapse runs. Then compare the spread of the four experts' final accuracy with PTA across the six `vit_eps4` runs, which requires the expert logs of the fixed seeds 3–5 and the unfixed seeds 0–2. Then compute the training and final cosine on the same batch with and without quantization.

## 11. Limits

- Six seeds, of which three were run with the seeding bug. The merged statistics mix two procedures (Section 2).
- The three unfixed seeds had one effective expert each, which probably changes the variance of their results.
- One budget (m = 1500), one source/target pair, untuned trigger parameters.
- The t-intervals with n = 6 are wide, and some detection intervals extend above 100%. They show the spread of six runs, not bounds.
- The defenses are reimplementations, and detection depends only on which labels are flipped (the victim sees clean images).
- The trigger figures in Section 5 come from the earlier `adapt` run, not the seeded runs.
- Visibility was judged by eye, not measured.
- A same-seed-twice check of the whole pipeline has not been seen.
- The PCA value of exactly 50.00 for `vit_eps8` seed 1 has not been checked.

## 12. Next steps

1. **Rerun seeds 0–2 with the fixed seeding** (about 8 hours on Quetzal), so that all six seeds share one procedure. The merged statistics above should then be recomputed without the caveat.
2. **Check the kmeans-collapse runs** for `vit_eps8` (seeds 1 and 4): read the defense outputs and check whether the PCA value of 50.00 in seed 1 is an artifact.
3. **Test the `vit_eps4` split** by comparing the spread of the four experts' final accuracy with PTA across all six runs, and by checking whether the flip concentration on the target predicts the failure in more runs.
4. **More seeds**, five to ten more, for `1xs` against `vit_eps8` and for the `vit_eps4` split. Six runs are not enough to settle either.
5. **Render the generators** for seeds 4 and 5 and for seed 5 `vit_eps8` (lowest cosine), to compare the failing and working triggers.
