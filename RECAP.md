# FLIP with a learned ViT trigger: experiment recap

Sweep tag `trig`. All numbers come from `out/sweep/trig/results.csv` and `flip_table.csv`. One run per setting, no repeated seeds.

## 1. What was tested

FLIP (Jha, Hayase, Oh, NeurIPS 2023) is a label-only backdoor. The victim trains on clean images with flipped labels, and the flips are chosen by trajectory matching against expert models trained on a backdoored dataset. In the paper the expert trigger is a fixed sinusoid (`1xs`).

**Change tested:** replace the fixed trigger with a trigger produced by a frozen, learned ViT generator, bounded to ε/255 per pixel. The attack itself is unchanged: the vanilla `generate_labels` and `train_user` code runs, and the victim never sees the trigger. The generator only affects (a) expert training and (b) the stamp used to measure PTA at test time.

**Goals (yours):**
1. High CTA and high PTA.
2. Lower detection by kmeans / PCA / SPECTRE than the `1xs` baseline.
3. An imperceptible trigger.

**Setting:** CIFAR-10, r32p model, SGD, source class 9 → target class 4, m = 1500 flipped labels.

## 2. Parameters

Chosen without a grid search, as requested, so they are a first guess and not tuned.

| Part | Value |
|---|---|
| ε (L∞, /255) | 2, 4, 8 |
| Generator | ViT: dim 192, depth 4, heads 3, patch 4, MLP ratio 4; 3×3 smoothing conv; `δ = ε·tanh(·)` |
| Trigger training | 10 epochs, batch 256, G lr 1e-3, surrogate lr 0.01, 1 warm-up epoch |
| Loss | `1.0·CE(surrogate(x+δ), target) + 0.01·TV(δ)`; discriminator off (`lambda_adv = 0`) |
| Output | quantized to 8-bit, clamped to [0, 255] |
| Experts | 4 per trigger × 20 epochs, retrained for each trigger |
| Attack | 15 label-optimization iterations |
| Victim | 60 epochs |
| Defenses | kmeans, PCA, SPECTRE (reimplementations in `modules/run_defenses/utils.py`), removal budget 1.5× the true poison count |

## 3. Main results

CTA, PTA and detection in %. Lower detection is better for the attacker. FPR is the share of clean examples the defense also removed.

| Trigger | CTA | PTA | Surrogate ASR | kmeans det | PCA det | SPECTRE det |
|---|---|---|---|---|---|---|
| `1xs` (paper) | 85.75 | 97.70 | -- | 96.77 | 86.92 | 94.68 |
| `vit_eps2` | 84.07 | 93.20 | 48.4 | 99.36 | 92.56 | 98.71 |
| `vit_eps4` | 84.63 | **100.00** | 83.8 | **88.97** | 77.35 | **76.64** |
| `vit_eps8` | 84.24 | 98.90 | 61.3 | 90.33 | **75.83** | 81.91 |

AUROC (kmeans / PCA / SPECTRE): `1xs` 0.988 / 0.919 / 0.978; eps2 0.996 / 0.946 / 0.993; eps4 0.960 / 0.882 / 0.879; eps8 0.969 / 0.889 / 0.937. FPR is 9–16% in every row.

**Reading it:**
- **ε = 4 and ε = 8 meet both goals against the baseline.** PTA is higher (100.0 and 98.9 against 97.7), and detection is lower on all three defenses. The largest drop is SPECTRE at ε = 4: 94.7 → 76.6.
- **ε = 2 does worse than the baseline on detection** (higher on all three) and on PTA (93.2). The trigger is probably too weak to give the experts a clean signal.
- **CTA is 1.1–1.7 points below `1xs` for every learned trigger.** This is a small gap from a single run, so I would not claim it is real without more seeds.
- **Surrogate ASR is not monotone in ε** (48 / 84 / 61). ε = 8 should be at least as easy as ε = 4, so the 10-epoch generator training is probably under-converged for ε = 8. It was not tuned.

## 4. Where the flips went

"On target" means the flipped label is class 4. "9 → 4" counts flips of source-class images to the target class.

| Trigger | Flips on target | 9 → 4 | Share of flips on 9 → 4 |
|---|---|---|---|
| `1xs` | 1147 / 1500 (76.5%) | 931 | 62.1% |
| `vit_eps2` | 1397 / 1500 (93.1%) | 1294 | 86.3% |
| `vit_eps4` | 852 / 1500 (56.8%) | 545 | 36.3% |
| `vit_eps8` | 724 / 1500 (48.3%) | 419 | 27.9% |

The largest non-target destinations:
- `1xs`: 9 → 1 (129), 9 → 8 (101).
- `vit_eps4`: 9 → 1 (326), 9 → 8 (100), 6 → 4 (79), 9 → 0 (71).
- `vit_eps8`: 9 → 1 (292), 9 → 8 (133), 9 → 0 (125), 9 → 3 (119).

The full per-pair tables are in `out/sweep/trig/flip_table.md`.

**Pattern:** detection roughly follows how concentrated the flips are. ε = 2 puts 86% of its flips on the single 9 → 4 pair and is the most detected. ε = 4 and ε = 8 spread flips across many pairs (36% and 28% on 9 → 4) and are the least detected. PTA stays at 99–100% despite far fewer target-class flips. This is an observation from four rows, not a tested mechanism. Concentrated flips may be easier for the defenses to cluster, but that was not isolated.

## 5. What the trigger looks like

Figures (original, triggered, perturbation at true scale, perturbation amplified) for 6 random source-class images per trigger:

- ![eps 2](out/sweep/trig/figures/vit_eps2/trigger_examples.png)
- ![eps 4](out/sweep/trig/figures/vit_eps4/trigger_examples.png)
- ![eps 8](out/sweep/trig/figures/vit_eps8/trigger_examples.png)

- **Bound is respected.** Realized max |change| is exactly 2, 4 and 8 /255, from the stored 8-bit images.
- **About 99% of pixels change** in each image.
- **The learned trigger is a fixed pattern, not image-dependent.** Measured on the 6 saved examples per trigger: the cosine similarity between different images' perturbations averages 0.99 (ε = 2), 0.99 (ε = 4) and 0.98 (ε = 8), so about 98–99% of the perturbation is one pattern shared by every image. It sits at ±ε on most pixels. The dominant spatial frequency is a period of 2 pixels for ε = 2 and ε = 4, and 4 pixels for ε = 8 (the same period as `1xs`, whose amplitude is 6). So it is a universal periodic trigger in the spirit of `1xs`, not an image-adaptive perturbation, and for ε = 2 and 4 it is not the same signal as `1xs` (different frequency, different amplitude, and it varies per colour channel).
- **Visibility.** I looked only at the ε = 8 figure. The true-scale perturbation is a flat grey with a faint grid, and the triggered images look like the originals at a glance. A faint regular texture is still detectable in flat regions on close inspection. I did not inspect the ε = 2 and ε = 4 figures. Whether this counts as "not noticeable by eye" is your judgement from the images above.

## 6. Cost

Wall-clock from the file times of `out/sweep/trig/logs/`: about 2 h 42 min from 10:49 to 13:31, against the 20 h budget.

| Step | Time |
|---|---|
| Trigger training | about 2 min per ε |
| Experts (4 × 20 epochs) | about 3.0 min each for `1xs`, about 3.6 min each for ViT triggers |
| Label optimization (attack) | about 16 min per setting |
| Victim training (60 epochs) | about 9 min per setting |

The ViT trigger adds roughly 20% to expert time, so the per-image CPU cost I suspected after the 16-minute smoke run is minor. I did not find the real cause of that smoke-run time and did not pursue it, since it no longer matters at this scale.

## 7. Limits and caveats

- **One seed per setting.** The ranking of eps4 against eps8, and the 1–2 point CTA gap, may not survive repetition. Only the larger effects (eps2 being more detected; eps4/8 SPECTRE and PCA drops of about 10–18 points) look safe to rely on, and even those are unreplicated.
- **Untuned parameters.** The non-monotone surrogate ASR suggests the trigger training is under-trained.
- **4 experts per trigger**, fewer than the paper's configuration.
- **Defenses are my reimplementations**, not the authors' code. Absolute detection rates may differ from published ones.
- **Detection cannot depend on the trigger directly**, because the victim data is clean. The drop for eps4/8 must come from which examples FLIP chose to flip, which is consistent with section 4.
- **The generator is not really adversarial.** The discriminator is off (`lambda_adv = 0`), so "GAN" in the earlier design discussion no longer applies. It is a ViT trained against a surrogate with a smoothness penalty.
- **The earlier design is dropped.** An earlier version added noise to images at attack time. You corrected that, because FLIP must stay label-only, and none of those results appear here.

## 8. Reproducing

```bash
python sweep/run_trigger_sweep.py --tag trig --epsilons 2 4 8 --budgets 1500 \
    --experts 4 --expert-epochs 20 --user-epochs 60
python sweep/make_figures.py --tag trig
```

The run was interrupted once with a KeyboardInterrupt during expert training and resumed with the same command, because finished steps are skipped. Outputs: `out/sweep/trig/results.{csv,md}`, `flip_table.{csv,md}`, `figures/`, `logs/`. Design notes: `TRIGGER.md`.

## 9. Suggested next steps

1. Repeat with 3 or more seeds to see whether the differences hold.
2. Train the generator longer, especially for ε = 8, and check surrogate ASR rises with ε.
3. Test the hypothesis in section 4 directly, for example by restricting the flips to 9 → 4 for ε = 4 and checking whether detection rises.
4. Make the trigger image-adaptive, since the current generator collapses to a near-constant pattern. This has been implemented as options (`--adaptive-floor`, `--lambda-div`); see `TRIGGER.md`.
