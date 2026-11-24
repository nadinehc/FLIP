# FLIP with a learned ViT trigger: consolidated recap (six seeds)

This is the consolidated recap. It replaces `RECAP_3.md` for all results. Everything here comes from six seeds (0–5) that all ran with the same, fixed code. Older documents are kept for history only: `RECAP.md` (first sweep, `trig`), `RECAP_2.md` (adaptive sweep, `adapt`) and `RECAP_3.md` (the first multi-seed attempt, which mixed seeds with and without the seeding bug).

Sources: `out/sweep/adaptfix_s{0..5}/` (`results.csv`, `flip_table.md`, `logs/`). Statistics: `out/sweep/seed_summary_fixed6.md` and `.csv`. Curves: `out/sweep/training_curves_fixed6/`.

## Summary

- **The learned ViT trigger reaches the same attack success as the paper's trigger, with the same clean accuracy.** PTA is 99.7 ± 0.4 for ε=8 and 98.6 ± 2.7 for `1xs`, and CTA is about 84% for all three.
- **ε=8 has lower PCA and SPECTRE detection than `1xs` in all six seeds.** The mean reductions are −14.7 points for PCA (95% CI [−24.3, −5.1]) and −18.2 points for SPECTRE (95% CI [−31.0, −5.4]). With six of six seeds in the same direction, a sign test gives p ≈ 0.03.
- **ε=8 has a kmeans failure mode.** Two of six seeds (1 and 4) give near-zero kmeans detection while PTA stays at 100%. The kmeans mean is therefore not meaningful, and the median (85.8%) describes the other four.
- **ε=4 does not lower detection.** Its paired differences from `1xs` are within noise on every defense.
- **ε=4 transfer is unreliable.** Four seeds reach PTA ≥ 88.9 and two (seeds 4 and 5) fail, at 16.4 and 39.6. The two failing seeds are also the two where the four experts' final accuracies vary widely (0.06 to 0.94). That is a plausible mechanism, not a tested one (Section 8).
- **`1xs` is stable across seeds**, with PTA 93–100 and SPECTRE detection 91–95 in all six runs.

## 1. Aim

FLIP (Jha, Hayase, Oh, NeurIPS 2023) is label-only: the victim trains on clean images with flipped labels, and trajectory matching against experts trained on triggered data decides which labels to flip. The paper's trigger is a fixed sinusoid (`1xs`).

Question: can a learned trigger keep the attack's effect (PTA, with CTA unchanged) while making the flipped set harder for detectors (kmeans, PCA, SPECTRE) to find? Lower detection is better for the attacker.

## 2. Setup

Fixed throughout: CIFAR-10, ResNet-32 (`r32p`), SGD, source class 9 (truck), target class 4 (deer), m = 1500 flipped labels, 4 experts of 20 epochs, victim trained 60 epochs, and the adaptive ViT options from RECAP_2 (`--adaptive-floor 0.25 --lambda-div 1.0 --div-tau 0.3`, 20 trigger epochs).

| Name | Description |
|---|---|
| `1xs` | The paper's sinusoidal stripes (hand-coded baseline). |
| `vit_eps4`, `vit_eps8` | A ViT generator, `delta = epsilon * tanh(G(x))`, epsilon 4/255 and 8/255, trained from random initialization. No sinusoid is used as a starting point. |

**Seeding.** `--seed` sets `FLIP_SEED`. Each step (trigger training, each expert, the attack, the victim) seeds Python, NumPy and PyTorch with `util.derive_seed(seed, experiment_name)`, and cuDNN runs in deterministic mode. The four experts of each setting therefore differ.

**Why this document uses only the fixed runs.** The first batch of seeds (0–2, in `RECAP_3.md`) ran with a bug that gave all four experts the same seed, so every setting used one expert trajectory. Those seeds were rerun with the fix (now `adaptfix_s0..2`), and the old results are superseded. All six seeds here share one procedure.

## 3. Results per seed

CTA, PTA and detection in %. Surrogate ASR and cross-image cosine are fractions. Lower detection is better for the attacker.

| Trigger | Seed | CTA | PTA | Surrogate ASR | Cross-image cos | kmeans det | PCA det | SPECTRE det |
|---|---|---|---|---|---|---|---|---|
| `1xs` | 0 | 84.25 | 99.70 | -- | -- | 98.21 | 88.19 | 94.45 |
| `1xs` | 1 | 84.82 | 99.60 | -- | -- | 98.25 | 88.02 | 94.59 |
| `1xs` | 2 | 83.36 | 99.30 | -- | -- | 98.77 | 87.25 | 93.14 |
| `1xs` | 3 | 86.16 | 99.90 | -- | -- | 97.03 | 79.64 | 91.43 |
| `1xs` | 4 | 83.52 | 99.90 | -- | -- | 98.75 | 86.75 | 91.25 |
| `1xs` | 5 | 79.58 | 93.20 | -- | -- | 97.32 | 82.97 | 90.59 |
| `vit_eps4` | 0 | 81.28 | 100.00 | 44.8 | 0.602 | 98.73 | 86.94 | 92.66 |
| `vit_eps4` | 1 | 85.27 | 100.00 | 87.2 | 0.777 | 96.31 | 79.21 | 88.75 |
| `vit_eps4` | 2 | 84.69 | 99.60 | 43.4 | 0.822 | 98.20 | 85.21 | 91.79 |
| `vit_eps4` | 3 | 84.15 | 88.90 | 26.4 | 0.727 | 93.38 | 79.46 | 82.25 |
| `vit_eps4` | 4 | 84.37 | **16.40** | 32.5 | 0.889 | 100.00 | 99.80 | 99.93 |
| `vit_eps4` | 5 | 85.75 | **39.60** | 52.7 | 0.886 | 99.93 | 99.47 | 99.93 |
| `vit_eps8` | 0 | 84.54 | 99.80 | 41.3 | 0.380 | 85.09 | 68.58 | 73.97 |
| `vit_eps8` | 1 | 83.71 | 100.00 | 28.2 | 0.900 | **0.00** | 67.69 | 76.19 |
| `vit_eps8` | 2 | 86.32 | 98.90 | 27.0 | 0.732 | 98.30 | 83.60 | 86.27 |
| `vit_eps8` | 3 | 83.56 | 99.60 | 24.1 | 0.900 | 86.47 | 73.07 | 81.52 |
| `vit_eps8` | 4 | 83.97 | 100.00 | 40.8 | 0.858 | **0.11** | 59.42 | 50.44 |
| `vit_eps8` | 5 | 86.62 | 100.00 | 34.9 | 0.515 | 88.87 | 72.18 | 77.74 |

## 4. Summary statistics

Mean ± sample standard deviation over the six seeds, in %. The full table with SEM, 95% confidence intervals and per-seed values is in `out/sweep/seed_summary_fixed6.md`.

| Trigger | CTA | PTA | kmeans det | PCA det | SPECTRE det |
|---|---|---|---|---|---|
| `1xs` | 83.61 ± 2.22 | 98.60 ± 2.65 | 98.06 ± 0.73 | 85.47 ± 3.44 | 92.58 ± 1.73 |
| `vit_eps4` | 84.25 ± 1.57 | 74.08 ± 36.69 | 97.76 ± 2.54 | 88.35 ± 9.26 | 92.55 ± 6.79 |
| `vit_eps8` | 84.79 ± 1.35 | 99.72 ± 0.43 | 59.81 ± 46.51 | 70.76 ± 7.94 | 74.36 ± 12.49 |

The `vit_eps4` PTA mean describes neither group, because the values split (Section 8). The `vit_eps8` kmeans mean is dominated by the two seeds near zero, so its median (85.8) is the better summary.

![summary by trigger](out/sweep/summary_figures/summary_metrics.png)

Each thin line is one seed, across the three triggers. The black line is the mean over six seeds, with 95% confidence intervals (t-based, n = 6). Points off the mean line show how far each seed sits from the average, and the crossing lines show which seeds move together.

Surrogate ASR and cosine, mean ± std: `vit_eps4` 0.48 ± 0.21 and 0.78 ± 0.11; `vit_eps8` 0.33 ± 0.07 and 0.71 ± 0.22. The surrogate ASR for `vit_eps8` is low, but PTA is high (99.7), so the surrogate underestimates the trigger (as in RECAP_2).

## 5. Error bars

### Method

For each trigger and metric, with six per-seed values x1…x6 (n = 6):

- mean = (x1 + … + x6) / 6
- sample standard deviation s = sqrt( Σ (xi − mean)² / 5 )
- standard error SEM = s / sqrt(6)
- 95% confidence interval = mean ± t × SEM, with t = 2.571 (two-sided 97.5% quantile for 5 degrees of freedom)

**Worked example, `vit_eps8` SPECTRE minus `1xs` (paired by seed).** Per-seed differences: −20.49, −18.40, −6.87, −9.90, −40.81, −12.85. Mean = −109.32 / 6 = −18.22. Deviations squared sum to 742.3, so s = sqrt(742.3 / 5) = 12.18. SEM = 12.18 / 2.449 = 4.97. 95% CI = −18.22 ± 2.571 × 4.97 = [−31.0, −5.4].

Interval endpoints above 100% come from the t-interval, which doesn't respect the bounds of a percentage. Read them as rough.

### Paired differences against `1xs` (points, same seed)

![paired differences](out/sweep/summary_figures/summary_paired.png)

Each dot is one seed's difference from `1xs` with the same seed. The black point is the mean, with its 95% CI. Dots below zero mean lower detection than `1xs` (for the defense panels). The `vit_eps8` kmeans CI runs from about −87 to +11, because two seeds collapse to zero, so that panel should be read as "unreliable" rather than as a measured reduction.

| Trigger | Metric | Mean difference | 95% CI |
|---|---|---|---|
| `vit_eps4` | CTA | +0.64 | [−2.72, +4.00] |
| `vit_eps4` | PTA | −24.52 | [−61.95, +12.92] |
| `vit_eps4` | kmeans det | −0.30 | [−2.67, +2.08] |
| `vit_eps4` | PCA det | +2.88 | [−7.36, +13.12] |
| `vit_eps4` | SPECTRE det | −0.02 | [−7.96, +7.92] |
| `vit_eps8` | CTA | +1.17 | [−2.41, +4.76] |
| `vit_eps8` | PTA | +1.12 | [−1.82, +4.06] |
| `vit_eps8` | kmeans det | −38.25 | [−87.39, +10.90] |
| `vit_eps8` | PCA det | **−14.71** | **[−24.32, −5.11]** |
| `vit_eps8` | SPECTRE det | **−18.22** | **[−31.01, −5.43]** |

Per seed, the `vit_eps8` PCA difference is negative in all six seeds (−19.6, −20.3, −3.6, −6.6, −27.3, −10.8), and so is the SPECTRE difference (−20.5, −18.4, −6.9, −9.9, −40.8, −12.9). The kmeans difference is negative in all six seeds, and strongly negative (about −98) in the two near-zero seeds.

## 6. Flipped labels

**Class numbers.** CIFAR-10 uses its standard label order:

| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| airplane | automobile | bird | cat | deer | dog | frog | horse | ship | truck |

So the source class 9 is **truck**, the target class 4 is **deer**, and the attack aims to make a triggered truck be classified as deer.

Share of the 1500 flips that are truck→deer (9→4), from each seed's `flip_table.md`:

| Trigger | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Seed 5 |
|---|---|---|---|---|---|---|
| `1xs` | 56.8% | 73.2% | 60.7% | 53.2% | 62.1% | 63.4% |
| `vit_eps4` | 59.9% | 48.7% | 57.3% | 51.2% | 100.0% | 100.0% |
| `vit_eps8` | 31.3% | 32.9% | 56.9% | 34.2% | 18.5% | 22.7% |

For `vit_eps4`, the two failing seeds (4 and 5) put all 1500 flips on the target. The other four seeds have 49–60% on target, so a high share alone does not explain the split (the earlier `RECAP_3` claim that it did was based on the single-expert runs and does not hold with the fixed seeds). Other flips of truck images go mostly to automobile (1), ship (8) and airplane (0), the visually closest classes.

## 7. Trigger images

**What the trigger is meant to do:** a triggered image of a **truck** (class 9) should be classified as a **deer** (class 4) by the poisoned model, while the trigger leaves clean accuracy unchanged. Each figure below shows one training truck, and the same trucks are used for every seed and epsilon.

Each figure has one row per seed (0–5) and five columns: the clean image, then for ε=4 and ε=8 the triggered image and the perturbation amplified to its own range. The amplified column shows the pattern but not the size, so don't compare amplitudes across panels from it. The triggered images are shown at the same scale as the clean one, with no enhancement.

Generated by `sweep/compare_triggers.py` from the fixed-seed generators, with `--pick-seed 0` for the image choice.

Training truck 38569:

![truck 38569, all seeds and epsilons](out/figures/compare/compare_img38569.png)

Training truck 3878:

![truck 3878, all seeds and epsilons](out/figures/compare/compare_img3878.png)

Training truck 48431:

![truck 48431, all seeds and epsilons](out/figures/compare/compare_img48431.png)

What the figures show, from the first truck (38569); the other two should be checked for the same pattern:

- **ε=4:** seeds 0–3 give vertical stripes. Seeds 4 and 5 give large, smooth blobs, grey in seed 4 and green in seed 5. Seeds 4 and 5 are the two runs where the ε=4 victim failed (PTA 16.4 and 39.6).
- **ε=8:** the patterns vary more between seeds. Seeds 0, 2 and 3 show coloured speckle or a diagonal pattern, seeds 1 and 5 show vertical stripes, and seed 4 shows a grey checkerboard.
- **At true scale** the triggered images look the same as the clean image at this size, so the perturbation is hard to see by eye. Visibility was judged by eye only.

Two caveats. The amplified columns are each scaled to their own range, so the green in seed 5 may be a real colour shift or a small signal stretched, and the true-scale size matters. And the link between the blob patterns and the ε=4 failures rests on two seeds, so it's a lead to test (Section 12), not a result.

**Why stripes?** They were not put in. The generator is a ViT with 4×4 patches, so each patch is produced by a shared head with a learned position embedding, and the 3×3 smoothing conv only softens the seams between patches. The surrogate is a CNN, which responds strongly to periodic high-frequency patterns, so stripes are a cheap shortcut for it to learn. The total-variation penalty and the diversity penalty both limit this but do not remove it. The `1xs` sinusoid is also a stripe pattern. A Fourier analysis of the perturbation would test these explanations, but it needs the trained generators and `torchvision`, which weren't available on this machine.

## 8. Training curves

Curves for all six seeds, one colour and marker per seed. The black line is the mean over six seeds, and the grey band is ±1 sample standard deviation. The values come from the step logs; in the last victim epoch, `acc0` is the CTA and `acc1` the PTA in `results.csv`. Every plotted value is in `out/sweep/training_curves_fixed6/curves.csv`.

### Victim

![victim curves](out/sweep/training_curves_fixed6/victim_curves.png)

CTA (top row) overlaps for all triggers and rises to about 84–86%. PTA (bottom row) is near 100% for `1xs` and `vit_eps8` in every seed, apart from the lower-than-usual `1xs` seed 5 (93.2%). `vit_eps4` splits: seeds 0–2 reach 99.6–100%, while seeds 4 and 5 oscillate at low values and seed 3 sits at 89%.

Late-epoch PTA over the last 10 epochs (epoch 60 | min | max | std):

| Trigger | Seed | Epoch 60 | Min | Max | Std |
|---|---|---|---|---|---|
| `1xs` | 0 | 99.7 | 94.8 | 99.9 | 1.6 |
| `1xs` | 1 | 99.6 | 91.9 | 99.9 | 2.6 |
| `1xs` | 2 | 99.3 | 87.9 | 99.9 | 3.7 |
| `1xs` | 3 | 99.9 | 99.6 | 100.0 | 0.1 |
| `1xs` | 4 | 99.9 | 95.1 | 99.9 | 1.7 |
| `1xs` | 5 | 93.2 | 93.2 | 99.5 | 2.2 |
| `vit_eps4` | 0 | 100.0 | 99.7 | 100.0 | 0.1 |
| `vit_eps4` | 1 | 100.0 | 99.8 | 100.0 | 0.1 |
| `vit_eps4` | 2 | 99.6 | 98.6 | 100.0 | 0.4 |
| `vit_eps4` | 3 | 88.9 | 88.9 | 100.0 | 3.6 |
| `vit_eps4` | 4 | 16.4 | 16.4 | 60.6 | 12.2 |
| `vit_eps4` | 5 | 39.6 | 5.6 | 64.2 | 18.1 |
| `vit_eps8` | 0 | 99.8 | 97.8 | 100.0 | 0.7 |
| `vit_eps8` | 1 | 100.0 | 100.0 | 100.0 | 0.0 |
| `vit_eps8` | 2 | 98.9 | 96.1 | 100.0 | 1.2 |
| `vit_eps8` | 3 | 99.6 | 92.7 | 100.0 | 2.2 |
| `vit_eps8` | 4 | 100.0 | 99.9 | 100.0 | 0.0 |
| `vit_eps8` | 5 | 100.0 | 100.0 | 100.0 | 0.0 |

The `vit_eps4` seeds 4 and 5 oscillate by tens of points from epoch to epoch, so their reported epoch-60 values are snapshots.

### Experts

![expert curves](out/sweep/training_curves_fixed6/expert_curves.png)

The experts learn every trigger. Their accuracy on triggered test images is close to 100% in most runs, and the clearest difference is in `vit_eps4`. The final accuracy of the four `vit_eps4` experts, from the logs:

| Seed | Expert final accuracy, `vit_eps4` (min – max) | `vit_eps4` PTA |
|---|---|---|
| 0 | 0.984 – 1.000 | 100.0 |
| 1 | 0.976 – 0.999 | 100.0 |
| 2 | 0.999 – 1.000 | 99.6 |
| 3 | 0.991 – 1.000 | 88.9 |
| 4 | **0.057 – 0.917** | **16.4** |
| 5 | **0.590 – 0.941** | **39.6** |

The two seeds where the `vit_eps4` experts disagree most are the two where PTA fails. In the other four seeds the four experts are within about 0.02 of each other. This is consistent with the failure coming from poorly trained expert trajectories, but it rests on six seeds, and the expert accuracy is measured on the logs' own test set, not on the victim.

### Attack

![attack curves](out/sweep/training_curves_fixed6/attack_curves.png)

The loss of the label optimization, averaged over bins of 0.1 epoch over its 15 epochs, with the same vertical scale in the three panels. The curves are noisy and not monotone for any trigger.

### Trigger training (ViT only)

![trigger curves](out/sweep/training_curves_fixed6/trigger_curves.png)

There is no trigger training for `1xs`. For the ViT triggers, the backdoor loss (top row) often rises over training instead of falling, so the generator does not converge within 20 epochs. The cross-image cosine (bottom row, on training batches) jumps at a seed-dependent epoch, from about 0.1–0.3 to about 0.85 for `vit_eps8`.

## 9. What the results support

Six seeds is still a small sample, so the strength of each claim is marked.

- **Strong (consistent across six seeds):** `vit_eps8` keeps PTA at 99.7 ± 0.4 with CTA unchanged, and `1xs` is stable. The `vit_eps8` PCA and SPECTRE detection is lower than `1xs` in all six seeds.
- **Moderate:** the `vit_eps8` reduction in PCA and SPECTRE detection is about 15–18 points on average, with intervals that exclude zero. The sign test gives p ≈ 0.03 for six of six. Because it's six runs, the size of the effect is less certain than its direction.
- **Moderate:** the `vit_eps4` transfer failures are linked to unstable expert trajectories (Section 8). Two seeds fail and four work, so the link is suggestive.
- **Weak:** the reason for the `vit_eps4` failures beyond the expert spread, and any effect of flip concentration (Section 6).
- **Not established:** that `vit_eps4` lowers detection. Its paired differences include zero on every defense.
- **Not usable yet:** the kmeans detection for `vit_eps8`, because two of six seeds collapse to near zero.

## 10. Open questions

1. **The kmeans collapse for `vit_eps8`** (seeds 1 and 4). PTA is at or near 100% in both, while PCA and SPECTRE are 50–68%. Whether kmeans fails to flag the poisoned set or the flipped set is unusual has not been checked. This is the most important thing to check, because it decides whether the kmeans numbers can be used at all.
2. **Why `vit_eps4` experts are unstable in seeds 4 and 5.** The spread may come from the trigger training (the cosine and backdoor loss in those seeds), from the expert training itself, or from both.
3. **The mismatch between training-time and final cosine** for `vit_eps4` (for example 0.41 against 0.87 in one of the early runs). The diversity penalty acts on the training value, so at ε=4 it may not constrain the trigger that is actually used. The cause has not been found.
4. **The low surrogate ASR for `vit_eps8`** (0.24–0.41) alongside PTA near 100%. The surrogate underestimates how well the trigger transfers through the flipped labels.

## 11. Limits

- Six seeds. Confidence intervals are wide (t = 2.571), and some detection intervals extend above 100%.
- One budget (m = 1500), one source/target pair, untuned trigger parameters, four experts per setting.
- The defenses are reimplementations, so absolute detection rates may differ from published ones. Detection depends only on which labels are flipped, since the victim sees clean images.
- The trigger figures in Section 7 come from the earlier `adapt` run, not from the six seeds.
- Visibility was judged by eye, not measured.
- `make_figures.py` reports a cross-image cosine that disagrees with `results.csv`. For seed 0 ε=4 it gives −0.06 on six images, against 0.60 in the results (measured on the test batch). This is not resolved, and the cosine values in Section 3 should be treated with that uncertainty.
- A same-seed-twice check of the whole pipeline has not been done, so reproducibility of a given seed is verified only for the expert stage (where the bug made identical experts).
- Seeds 6–8 were queued on Quetzal after seeds 0–2, but their status has not been checked. They are not included in this document.

## 12. Next steps

1. **Check the kmeans collapse for `vit_eps8`** (seeds 1 and 4). Read the defense outputs in `experiments/sweep_adaptfix_s{1,4}_user_vit_eps8_m1500/defenses.json`, and check whether the kmeans output is degenerate (for example, flagging nothing) in those two runs.
2. **Compare `vit_eps4` expert spread with PTA over more seeds**, and check the trigger training of seeds 4 and 5 for the cause of the unstable experts.
3. **Check on seeds 6–8.** If they finished, add them to the tables here. If not, decide whether to rerun them.
4. **Quantify the perturbations** for all six seeds and three images: the per-channel mean and the high-frequency energy of the true-scale perturbation. This tests whether the failing ε=4 generators (seeds 4 and 5) have the large smooth shift seen in Section 7. Then run a Fourier analysis for the stripe question.
5. **Repeat the reproducibility check** for the whole pipeline (same seed twice, compare the generator bytes and the results) so the seeds can be called reproducible.
