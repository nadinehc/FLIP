# FLIP with a learned ViT trigger: experiment recap, part 2

This recap continues `RECAP.md`, which covers the first full sweep (`trig`). It adds the image-adaptive sweep (`adapt`) and a correction to how the two sweeps should be compared.

One seed per setting throughout. Numbers are copied from `out/sweep/trig/results.md` and `out/sweep/adapt/results.md`.

## 1. Goal and setup

FLIP (Jha, Hayase, Oh, NeurIPS 2023) is label-only. The victim trains on clean images with flipped labels. Trajectory matching against experts trained on triggered data chooses which labels to flip. The paper's trigger is the fixed sinusoid `1xs`.

The question is whether a learned trigger can keep the attack's effect while being harder for detectors to catch.

Fixed across all sweeps: CIFAR-10, ResNet-32 (r32p), SGD, source class 9, target class 4, m = 1500 flipped labels.

## 2. The two trigger designs

**`trig` (first sweep).** A ViT generator outputs `delta = epsilon * tanh(G(x))`, with epsilon in {2, 4, 8} out of 255. The trigger is trained against a surrogate classifier. Nothing rewards it for varying across images.

**`adapt` (second sweep).** The same generator, plus two changes, both off by default in the code:
- `--adaptive-floor 0.25`: the bound is scaled per pixel by a texture map of the clean image. Textured regions get the full epsilon, and flat regions only 25% of it.
- `--lambda-div 1.0 --div-tau 0.3`: a penalty on the cosine similarity between different images' perturbations, applied above 0.3.

Only epsilon 4 and 8 were run in `adapt`. Epsilon 2 was dropped as the weakest in `trig`.

The attack is unchanged in both sweeps. Only the trigger differs.

## 3. Results

CTA, PTA and detection in %. Lower detection is better for the attacker.

### `trig`

| Trigger | CTA | PTA | Surrogate ASR | kmeans det | PCA det | SPECTRE det |
|---|---|---|---|---|---|---|
| `1xs` | 85.75 | 97.70 | -- | 96.77 | 86.92 | 94.68 |
| `vit_eps2` | 84.07 | 93.20 | 48.4 | 99.36 | 92.56 | 98.71 |
| `vit_eps4` | 84.63 | 100.00 | 83.8 | 88.97 | 77.35 | 76.64 |
| `vit_eps8` | 84.24 | 98.90 | 61.3 | 90.33 | 75.83 | 81.91 |

### `adapt`

| Trigger | CTA | PTA | Surrogate ASR | kmeans det | PCA det | SPECTRE det |
|---|---|---|---|---|---|---|
| `1xs` | 85.02 | 99.70 | -- | 96.02 | 83.01 | 90.98 |
| `vit_eps4` (adaptive) | 84.28 | 99.90 | 43.9 | 96.36 | 83.07 | 89.37 |
| `vit_eps8` (adaptive) | 82.76 | 100.00 | 42.9 | 90.57 | 74.30 | 83.39 |

### The two `1xs` baselines disagree

The `1xs` row is the same configuration in both sweeps, but the numbers moved:

| | `trig` | `adapt` | Change |
|---|---|---|---|
| CTA | 85.75 | 85.02 | -0.7 |
| PTA | 97.70 | 99.70 | +2.0 |
| kmeans det | 96.77 | 96.02 | -0.8 |
| PCA det | 86.92 | 83.01 | -3.9 |
| SPECTRE det | 94.68 | 90.98 | -3.7 |

Rerunning the same baseline moves PCA and SPECTRE detection by about 4 points. Any difference between a learned trigger and `1xs` that is smaller than that is within run-to-run noise. This matters for the comparisons below.

## 4. What each sweep shows

**`trig`.**
- Epsilon 4 and 8 have lower detection than `1xs` on all three defenses, up to 18 points on SPECTRE at epsilon 4.
- Epsilon 4 reaches PTA 100.0. Epsilon 2 is worse on both PTA and detection.
- CTA is 1–2 points below `1xs`.
- Given the baseline noise above, the SPECTRE and PCA gaps at epsilon 4 and 8 are probably real. The size of the effect is not reliable.

**`adapt`.**
- Epsilon 4 does not evade anything. Its detection matches `1xs` within about 2 points on every defense.
- Epsilon 8 is the only one lower than `1xs`, by 5–9 points on kmeans, PCA and SPECTRE. It also has the lowest CTA, 82.8.
- Adaptive triggers reach PTA 99.9–100.0 despite surrogate ASR of only about 43%.

So the adaptive mechanism did not clearly improve on `trig`. The one result that stands out, epsilon 8, is also the one most affected by noise.

## 5. Where the flips went

Share of flips with true class 9 flipped to the target class 4:

| Trigger | 9 → 4 count | Share |
|---|---|---|
| `trig` `1xs` | 931 | 62.1% |
| `trig` `vit_eps2` | 1294 | 86.3% |
| `trig` `vit_eps4` | 545 | 36.3% |
| `trig` `vit_eps8` | 419 | 27.9% |
| `adapt` `1xs` | 839 | 55.9% |
| `adapt` `vit_eps4` | 795 | 53.0% |
| `adapt` `vit_eps8` | 543 | 36.2% |

The concentration pattern from `RECAP.md` does not hold up. In `adapt`, epsilon 4 has nearly the same share as `1xs` but a slightly lower detection rate, and epsilon 8 has a lower share and lower detection. Share alone does not predict detection. The full per-pair tables are in `flip_table.md` for each sweep.

## 6. Trigger diagnostics

Surrogate ASR is the fraction of held-out source images the trigger pushes to the target class in the surrogate model. Cross-image cosine is the average cosine similarity between the perturbations of different images (1.0 = the same pattern on every image).

| Trigger | Surrogate ASR | Cross-image cosine, 6 example images | Cross-image cosine, training log |
|---|---|---|---|
| `trig` epsilon 2 | 48.4 | 0.989 | not logged |
| `trig` epsilon 4 | 83.8 | 0.986 | not logged |
| `trig` epsilon 8 | 61.3 | 0.980 | not logged |
| `adapt` epsilon 4 | 43.9 | 0.889 | 0.874 |
| `adapt` epsilon 8 | 42.9 | 0.904 | 0.879 |

The middle column is the comparable one. I computed it the same way for every trigger, from the saved original and triggered images in `figures/` (six images each). The training-log column only exists for `adapt`, because the logging was added afterwards.

So the adaptive trigger is less uniform across images (0.89–0.90 against 0.98–0.99), but far from image-specific. A cosine of about 0.9 means most of the perturbation is still one shared pattern. Section 7 shows what that pattern is.

The trigger-only check used a surrogate ASR of 60% as the threshold for going ahead. Both `adapt` triggers were below it, and the full run went ahead anyway. PTA near 100% suggests the attack still works, but the trigger is weaker as a shortcut for the surrogate.

## 7. Example images

Each figure shows six random source-class (class 9) training images. The columns are the original, the triggered image, the perturbation at true scale, and the perturbation amplified so its structure is visible. The amplification factor is printed above the last column (for example x32 at epsilon 4 and x16 at epsilon 8). The first two columns are shown at the same scale with no enhancement, so any difference you can see there is a difference the victim's data would contain. The true-scale column is what the victim's images actually differ by.

Realized changes, from the saved 8-bit images:

| Trigger | Bound | Mean largest per-image change | Pixels changed |
|---|---|---|---|
| `trig` epsilon 2 | 2/255 | 2.0 | 99% |
| `trig` epsilon 4 | 4/255 | 4.0 | 99% |
| `trig` epsilon 8 | 8/255 | 8.0 | 99% |
| `adapt` epsilon 4 | 4/255 | 3.8 | 99% |
| `adapt` epsilon 8 | 8/255 | 7.3 | 99% |

The bound is respected everywhere. In `trig` every image reaches the bound. In `adapt` some images stay below it (for example 7/255 at epsilon 8, 3/255 at epsilon 4), as intended for flat images.

### `adapt`, epsilon 8

![adapt epsilon 8](out/sweep/adapt/figures/vit_eps8/trigger_examples.png)

### `adapt`, epsilon 4

![adapt epsilon 4](out/sweep/adapt/figures/vit_eps4/trigger_examples.png)

### `trig`, epsilon 8

![trig epsilon 8](out/sweep/trig/figures/vit_eps8/trigger_examples.png)

### `trig`, epsilon 4

![trig epsilon 4](out/sweep/trig/figures/vit_eps4/trigger_examples.png)

### `trig`, epsilon 2

![trig epsilon 2](out/sweep/trig/figures/vit_eps2/trigger_examples.png)

### What the figures show

- **`trig`: one pattern on every image.** The amplified perturbation is the same regular grid in every row, saturated at the bound. The perturbation does not respond to the image.
- **`adapt`: the same idea with a stripe pattern, plus image-dependent bursts.** The amplified column in both `adapt` figures is dominated by vertical stripes (a period of about 2 pixels) that look the same in every row. On top of that, the amplitude is clearly larger in places with strong edges or texture, for example the top band of the third image in both figures and the truck outline at epsilon 8. So the texture mask works, but the shared stripe pattern remains the main component. This matches a cosine of 0.9 and not a value near 0.
- **The adaptive trigger is close in form to `1xs`.** `1xs` is also a stripe pattern. The learned trigger converged to stripes too, with a shorter period.
- **Visibility.** At true scale the perturbation is a flat grey with very faint vertical lines. In the triggered column I can see faint vertical lines on smooth regions, for example the sky in the first row and the white body of the truck in the second row, mainly at epsilon 8. I did not run a visibility test beyond looking at these figures, so treat this as my reading of the images, not a measurement. Whether it passes your "not noticeable by eye" requirement is your call. The individual full-size images are in `out/sweep/<tag>/figures/<trigger>/`.

## 8. Cost

| Sweep | Wall-clock | Steps |
|---|---|---|
| `trig` | about 2 h 42 min | 1xs and 3 ViT triggers, 4 experts each, attack, victim |
| `adapt` | 1 h 55 min | 1xs and 2 ViT triggers, same structure |

The `adapt` run was interrupted once by a network disconnect during the attack step of `vit_eps4`. It was resumed with the same command in tmux, and the interrupted attack was redone.

The trigger-only check took about 4 minutes per trigger.

## 9. Limits

- **One seed per setting, and the baseline shows about 4 points of run-to-run drift.** Differences smaller than about 5 points are not reliable. This affects every comparison in this document.
- **Few experts.** Four per trigger, against the paper's configuration.
- **Parameters are untuned.** They were chosen, not searched. The adaptive floor, the diversity weight and the 20-epoch trigger training were all first guesses.
- **Cosine is measured on only six images per trigger**, so it is a rough guide to how image-dependent a trigger is.
- **Visibility is judged by eye from six images per trigger** (section 7), not measured. Faint vertical lines are visible on smooth regions at epsilon 8.
- **Defenses are reimplementations**, so absolute detection rates may differ from the published ones.
- **Detection depends only on which labels get flipped.** The victim sees clean images, so any change in detection comes from the expert trajectories and the choice of flips.

## 10. Reproducing

```bash
# first sweep
python sweep/run_trigger_sweep.py --tag trig --epsilons 2 4 8 --budgets 1500 \
    --experts 4 --expert-epochs 20 --user-epochs 60
python sweep/make_figures.py --tag trig

# adaptive sweep
python sweep/run_trigger_sweep.py --tag adapt --epsilons 4 8 --budgets 1500 \
    --experts 4 --expert-epochs 20 --user-epochs 60 \
    --adaptive-floor 0.25 --lambda-div 1.0 --div-tau 0.3 --trigger-epochs 20
python sweep/make_figures.py --tag adapt
```

## 11. Next steps

1. **Repeat the baseline and the epsilon 8 adaptive run with new seeds.** This is the only way to tell whether epsilon 8's 5–9 point drop is real. Without it, neither sweep can support a claim that a learned trigger evades detection.
2. **Break the shared stripe pattern.** The adaptive mask only changes where the stripes are strong. A stronger diversity penalty (a lower `--div-tau` or a higher `--lambda-div`) is the next thing to try, with `--trigger-only` as the check.
3. **Judge visibility on more images and on the enlarged originals**, since the faint stripes at epsilon 8 may conflict with the "not noticeable" requirement.
4. **Check whether the surrogate ASR matters for PTA.** Adaptive triggers reach PTA near 100% at about 43% surrogate ASR, which suggests the surrogate number undersells the trigger.
