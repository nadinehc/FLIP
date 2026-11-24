# Learned triggers vs handcrafted triggers in FLIP: recap

**Setting:** CIFAR-10, truck (class 9) → deer (class 4). ResNet-32 expert and victim. L∞ bound 6/255. The attack is label-only: the victim sees clean images with flipped labels, and only the trigger used to train the experts differs.

**Research question:** Do learned (GAN-generated) triggers give a measurable advantage over the handcrafted sinusoidal trigger (`1xs`) in FLIP, and how do their design choices affect poisoning efficiency and detectability?

---

## 1. What we did

`sweep/seed0_pipeline.sh overnight` runs four stages on the server:

1. **Smoke test.** Checks the pipeline end to end. Passed.
2. **Fine-tuning.** Four trigger configs at budget 1500, with a reduced protocol (2 experts, 40 user epochs). Each config also trains its own sinusoidal baseline.

   | Config | Diversity penalty `lambda_div` | Adaptive floor | Generator LR |
   |---|---|---|---|
   | A | 0 | off | 1e-3 |
   | B | 1.0 | off | 1e-3 |
   | C | 1.0 | 0.25 | 1e-3 |
   | D | 1.0 | 0.25 | 2e-4 |

3. **Automatic pick.** Highest PTA at budget 1500, among configs whose trigger CTA is at most 2 points below their own sinusoidal baseline. A higher CTA always passes. Ties go to the higher CTA. If no config passes, the highest PTA wins. **Winner: C.**
4. **Final protocol** with config C, for seeds 0–4: 5 experts, 100 user epochs, budgets 300 / 500 / 1000 / 1500.

### Why these settings
- **Diversity penalty and adaptive floor.** Earlier triggers converged to almost one pattern for every image. The penalty pushes different images toward different perturbations, and the adaptive floor scales the allowed amplitude by image texture.
- **Several seeds.** Each seed changes the trigger, experts, label optimisation and victim. The protocol asks for at least 3 seeds.
- **Defences in every run.** Detection is central to the research question, so every victim is checked against kmeans, PCA and SPECTRE.
- **Baseline in every run.** The sinusoidal trigger shares every seed, budget and expert setting with the learned trigger, so comparisons are paired.

---

## 2. Run timings (from `overnight.log`)

| Stage | Start | End | Duration |
|---|---|---|---|
| Smoke test | 22:14 | ~22:14 | < 1 min |
| Fine-tune A | 22:14 | 23:25 | 1h 11min |
| Fine-tune B | 23:25 | 00:35 | 1h 11min |
| Fine-tune C | 00:35 | 01:46 | 1h 11min |
| Fine-tune D | 01:46 | 02:57 | 1h 11min |
| Final, seed 0 (config C) | 02:57 | 06:30 | 3h 33min |
| Final, seed 1 | 06:30 | — | **log stops here** |

**Per-run breakdown.** A single victim run took about 6–7 min in fine-tuning and about 16–17 min in the final run. The final run's baseline experts took about 1h 40min.

**Seeds 1–4 are incomplete.** The imported log ends at the start of seed 1. If the job is still running on the server, the rest will arrive later. Otherwise the job stopped at that point.

---

## 3. Results

### 3a. Fine-tuning winner (budget 1500)

**Winner: config C** (`grep "fine-tuning winner" overnight.log`).

Single seed (0), 2 experts, 40 user epochs. Each row is the GAN trigger (ε = 6/255) at 1500 flips. Baseline CTA is the sinusoidal trigger from the same config.

| Config | CTA (%) | Baseline CTA (%) | CTA gap (pts) | PTA (%) | kmeans det. (%) | PCA det. (%) | SPECTRE det. (%) | Surrogate ASR | Cross-image cos. | Passes CTA rule? |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 85.8 | 79.3 | +6.5 | 38.3 | 89.3 | 71.4 | 79.0 | 0.54 | 0.983 | Yes |
| B | 82.5 | 84.6 | −2.1 | 100.0 | 0.4 | 66.6 | 77.2 | 0.65 | 0.985 | **No** (gap > 2) |
| C | 82.7 | 84.1 | −1.4 | 100.0 | 96.7 | 83.9 | 91.7 | 0.52 | **0.241** | Yes, **selected** |
| D | 83.8 | 83.8 | +0.0 | 74.1 | 93.8 | 83.4 | 87.8 | 0.68 | 0.291 | Yes |

Baseline (sinusoidal) detection for the same configs: kmeans 98.3–98.4%, PCA 85.7–87.9%, SPECTRE 96.3–97.4%. The baseline PTA ranged from 94.0% (C) to 99.8% (A).

**Reading the table:**
- **Config C is the winner.** Among configs that pass the CTA rule, C has the highest PTA (100%). B also reaches 100% PTA but fails the rule, and D passes with lower PTA (74%).
- **The adaptive floor is what reduces cross-image similarity.** Config B (diversity penalty only) stays at 0.985, the same as A with no penalty. Adding the adaptive floor (C, D) drops it to 0.24–0.29. The penalty alone did not work at this setting.
- **C is somewhat harder to detect than the baseline, not easier.** Its detection rates are close to the baseline, and it is lower than the baseline by about 1–6 points (SPECTRE is 5.7 points lower). That's a small effect at one seed.
- **B's kmeans result (0.4% detection, AUROC 0.05) is not trustworthy.** An AUROC that low suggests the kmeans defence scored the poisoned samples in reverse. Treat it as a defence failure, not as stealth.
- **The baseline itself varies across configs.** Its CTA ranges from 79.3% to 84.6%, and its PTA from 94.0% to 99.8%, even with the same seed. So differences of 1–2 points are within noise for 2-expert runs, and the CTA rule is a noisy filter.

### 3b. Final run, config C, seed 0

| Budget | CTA (%) | PTA (%) | Paired PTA diff vs 1xs | kmeans det. (%) | PCA det. (%) | SPECTRE det. (%) |
|---|---|---|---|---|---|---|
| 300 | TBD | TBD | TBD | TBD | TBD | TBD |
| 500 | TBD | TBD | TBD | TBD | TBD | TBD |
| 1000 | TBD | TBD | TBD | TBD | TBD | TBD |
| 1500 | TBD | TBD | TBD | TBD | TBD | TBD |

**Not yet imported.** The `final_s0_C` folder isn't in `flip_results/sweep/`. The overnight log says it finished, so the copy didn't pick it up. Section 5 has the command to fetch it.

### 3c. Seeds 1–4

Not yet available (see section 2).

---

### 3d. Victim training curves (earlier six-seed run, 60 user epochs)

![Victim training curves: clean test accuracy and poisoned test accuracy per epoch for the sinusoidal trigger and the two GAN triggers](out/sweep/training_curves_fixed6/victim_curves.png)

*Figure: Victim training on clean images with flipped labels, for the sinusoidal trigger (1xs) and the GAN triggers at ε = 4 and ε = 8. Top row: clean test accuracy (CTA) per epoch. Bottom row: poisoned test accuracy (PTA) per epoch. Thin coloured lines are the six seeds (0–5). The black line is the mean, and the grey band is ±1 standard deviation across seeds.*

The top row shows that all three triggers reach a similar clean accuracy, around 83–86%, by about epoch 30. The clean accuracy curves overlap, so the trigger does not appear to cost clean accuracy. The bottom row shows the attack. With the sinusoidal trigger, PTA climbs to about 95–100% within the first 10–15 epochs for most seeds, and it stays there. With ε = 8, PTA is erratic early on but reaches about 100% for every seed after roughly epoch 25, and then stays flat. With ε = 4, PTA is high for some seeds (0–3) but stays low for seeds 4 and 5, which keep fluctuating between about 2% and 75%. This is why the ε = 4 mean has a very wide standard deviation. Across all triggers, PTA changes sharply from one epoch to the next, so single-epoch values are unreliable and the end-of-training figures should be read with that in mind.

*Note:* this figure comes from the earlier sweep, which trained victims for 60 epochs, not the 100 used in the overnight protocol. It is shown for the training dynamics only and is not part of the fine-tuning or final comparison.

## 4. Open items

- **Import the final seed 0 metrics.** The fine-tuning results are in. `out/sweep/final_s0_C/` is still missing. Section 3b needs it.
- **Check whether seeds 1–4 are still running.** See the commands below.
- **Surrogate ablations** (ResNet-18, EfficientNet, no surrogate, transformer surrogate). The pipeline has no surrogate option, so this needs code changes.
- **Multiple victim runs per budget.** Each budget currently gets one victim run per seed.
- **Input-conditional (image-specific) perturbations.** These overlap with a parallel project, so they're out of scope here.
- **Checkerboard analysis.** The steganography-inspired objective is not implemented.

---

## 5. Commands

Check the server (run on your Mac):

```bash
ssh <your-host> 'tail -n 5 /users/eleves-a/2024/alexandre.tosi/FLIP-main/overnight.log; ls /users/eleves-a/2024/alexandre.tosi/FLIP-main/out/sweep | grep final_s'
```

Import the metrics and the rest of the log (run on your Mac):

```bash
cd ~/Documents/X/3A/MAP/EA/FLIP-main/flip_results
mkdir -p sweep
rsync -avz --prune-empty-dirs \
  --include='*/' --include='*.csv' --include='*.md' --include='*.json' \
  --include='*.png' --include='*.log' --include='config.toml' --exclude='*' \
  "<your-host>:/users/eleves-a/2024/alexandre.tosi/FLIP-main/out/sweep/ft_s0_*" ./sweep/
rsync -avz --prune-empty-dirs \
  --include='*/' --include='*.csv' --include='*.md' --include='*.json' \
  --include='*.png' --include='*.log' --include='config.toml' --exclude='*' \
  "<your-host>:/users/eleves-a/2024/alexandre.tosi/FLIP-main/out/sweep/final_s0_C" ./sweep/
scp <your-host>:/users/eleves-a/2024/alexandre.tosi/FLIP-main/overnight.log ./overnight.log
```

Then paste the contents of `flip_results/sweep/ft_s0_*/results.md` and `flip_results/sweep/final_s0_C/results.md` here, and I'll fill in sections 3a and 3b.

---

## 6. Reproduce

```bash
PYTHON=$(which python) bash sweep/seed0_pipeline.sh overnight 2>&1 | tee overnight.log
PYTHON=$(which python) bash sweep/seed0_pipeline.sh final C 0
```
