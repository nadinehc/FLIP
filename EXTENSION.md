# FLIP + G: adding an image-dependent perturbation to a label-only attack

> **Superseded.** This document describes an earlier experiment in which the
> perturbation was added to the label-flipped *training* images. That makes the
> attack no longer label-only, which defeats the point of FLIP. The current
> experiment moves the generator to the *trigger* and keeps the attack
> label-only -- see [TRIGGER.md](TRIGGER.md). The code described below still
> works and still passes its tests, but is not what the current sweep runs.

This is an extension to the FLIP implementation in this repo. Vanilla FLIP is
label-only: images are untouched and only the labels of a subset of training
examples are flipped. The question here is whether adding a small,
*image-dependent* perturbation `G(x)` to the label-flipped examples makes the
attack stronger (higher PTA at the same CTA and budget) and/or harder for
representation-space defenses to detect.

Nothing in the vanilla path was modified. The baseline runs through the
original `generate_labels` / `train_user` modules, untouched, so
baseline numbers remain directly comparable to the paper.

---

## What was added

| Path | Purpose |
|---|---|
| `modules/base_utils/image_noise.py` | The generator `G`, epsilon bookkeeping, flip-candidate masks, and the perturbed-image dataset. |
| `modules/generate_labels_noise/` | Step 2 with joint optimization of the soft labels **and** `G` against the same trajectory-matching loss. |
| `modules/train_user_noise/` | Victim training with the perturbation baked into the flipped examples. |
| `modules/run_defenses/` | kmeans / PCA / SPECTRE detection, reported as detection rate, FPR, and AUROC. |
| `schemas/{generate_labels_noise,train_user_noise,run_defenses}.toml` | Config schemas, as `run_experiment.py` requires. |
| `experiments/example_attack_noise/config.toml` | A single annotated FLIP+G example, mirroring `example_attack`. |
| `sweep/run_sweep.py` | The full baseline-vs-epsilon sweep, unattended and resumable. |
| `sweep/submit_sweep.sbatch` | SLURM wrapper. |
| `sweep/tests/` | CPU unit tests; no GPU, no dataset download, runs in seconds. |

One pre-existing bug was fixed, because the vanilla baseline could not run
without it: `modules/generate_labels/run_module.py` used `n_classes` on line 57
but only defined it on line 73, so the label-only path died with a `NameError`
before doing any work. The definition was moved above its first use. No
behaviour or hyperparameter changed.

---

## The generator

A small conv encoder-decoder, `G(x) -> delta`, with `delta` the same shape as
`x`. No discriminator; `G` is trained end-to-end against the attack objective
alone.

**The bound holds by construction, not by projection.** For L-infinity the
output passes through `epsilon * tanh(.)`; for L2 the per-image norm is squashed
into the epsilon ball. The perturbation is then clamped so that `x + delta`
stays a valid image. `G` is therefore feasible at every step of training, so
there is no "optimize then project" gap that would make the reported epsilon a
polite fiction.

**Epsilon is in pixel units and is converted into normalized space per
channel.** `datasets.py` normalizes before the network sees anything, so a
perturbation added naively in normalized space would be roughly `1/std` times
larger than intended — about 5x on CIFAR-10. `G` works internally in pixel
space (`[0, 1]`) and divides by the per-channel std on the way out, so
`epsilon = 8/255` means 8/255 in pixel space, as in the backdoor literature.

**`G` starts at approximately zero.** The output head is initialized near zero,
so the attack begins at vanilla FLIP and the generator has to earn its
perturbation. The tests assert the initial perturbation is under 1/255.

**GroupNorm, not BatchNorm.** `G` is applied to batches whose composition
varies with the flip-candidate mask, and the perturbation should not depend on
which other images happen to share a batch.

### Which images get perturbed

Only images the attacker is allowed to flip. In the targeted setting
(`source_label = 9`) that is the source class; with `source_label = -1` it is
every non-target image. Everything else in the dataset stays bit-for-bit clean,
which keeps the threat model honest: the attacker only touches what they
already touch.

During label optimization the perturbation is applied to all flip *candidates*,
because which examples end up flipped is only decided afterwards by
`select_flips`. At victim-training time only the examples that were *actually*
flipped are perturbed.

### Where the perturbation is applied

Two deliberate choices, both of which make the reported epsilon mean what it
says:

1. **Baked in as 8-bit images, before the victim's augmentation.** The threat
   model is that the attacker hands over a dataset and the victim trains on it
   however they like. So the perturbation is rounded to the `1/255` grid and
   written into the images, then the victim's `RandomCrop` / flip pipeline runs
   on top. A perturbation that only exists in float, after augmentation, would
   not survive being written to a PNG.
2. **During label optimization, `G` sees the augmented image** — that is what
   `MTTDataset` provides. This is a mismatch with (1), and a deliberate one: it
   makes `G` learn a perturbation that is robust to the victim's augmentation
   rather than one tuned to a specific crop. Worth keeping in mind when reading
   the results.

The test-time trigger is unchanged. PTA is still measured on trigger-stamped
test images; `G` is an attack-time tool, not part of the trigger. CTA is still
measured on the clean test set.

---

## The joint optimization

In `generate_labels_noise`, the only change to equation (3) is the student's
input: flip-candidate images `x` become `x + G(x)`. Both the soft labels and
`G`'s parameters are optimized against the resulting `L_param`:

- labels: SGD, `lr = 150`, `momentum = 0.5` (the repo's existing defaults)
- `G`: Adam, `lr = 1e-3` (configurable), optional warmup of label-only
  iterations before `G` starts training

Gradients reach `G` through the student's forward pass and the second-order
`create_graph=True` path that FLIP already uses for the labels. Set
`use_image_noise = false` to recover the label-only objective exactly.

---

## The defenses

**These are reimplementations.** The published FLIP release does not ship the
Appendix D.2 defense code — it lives in the upstream
[backdoor-suite](https://github.com/SewoongLab/backdoor-suite). I grepped this
repo for spectre/kmeans/PCA and found nothing, so all three were written from
the papers. Treat absolute detection numbers as indicative and the
baseline-vs-FLIP+G *comparison*, which uses identical code on both sides, as
the real result.

All three inspect the penultimate-layer representations of a model trained on
the poisoned data, restricted to examples carrying the target label:

- **kmeans** — activation clustering (Chen et al. 2018). 2-means; the smaller
  cluster is treated as poison. Scored continuously by the difference in
  distance to the two centers, so it ranks rather than only partitions.
- **pca** — spectral signatures (Tran et al. 2018). Magnitude of the projection
  onto the top singular vector of the centered representations.
- **spectre** — robust whitening plus a QUE score (Hayase et al. 2021). The
  robust covariance is approximated by iterative trimming, seeded with a
  spectral-signature pass. The seeding matters: a covariance fitted on the
  poisons too would whiten away the very direction they inflate, and QUE would
  see nothing. (An earlier version without seeding scored 0.61 AUROC on a
  synthetic planted signal where PCA got 0.99; with seeding it gets 0.99.)

Reported per defense:

- **detection rate** at a `1.5x` removal budget over the true poison count —
  the convention favourable to the defender. Lower means a stealthier attack.
- **false positive rate** at the same budget.
- **AUROC**, which is budget-free and easier to compare across settings.

Two caveats worth knowing when reading the table:

- The defenses are *given* the true number of poisons as their budget. This is
  generous to the defender and makes the comparison across settings clean.
- `select_flips` picks the best non-true class, which is usually but not
  necessarily the target. Flipped examples that land in another class fall
  outside the examined set, so `n_flipped` and `n_poison_in_target` can differ;
  both are in the CSV.

---

## Running the sweep

Full comparison, unattended:

```bash
python sweep/run_sweep.py --tag main
```

On SLURM:

```bash
TAG=main sbatch sweep/submit_sweep.sbatch
```

Check the pipeline end to end first — minutes rather than days:

```bash
python sweep/run_sweep.py --tag smoke --smoke
```

See the plan without running anything:

```bash
python sweep/run_sweep.py --tag main --dry-run
```

### What it runs

Table 1's main CIFAR-10 setting throughout: ResNet-32 (`r32p`), `1xs`
sinusoidal trigger, source 9 -> target 4, `m` in
`{150, 300, 500, 1000, 1500}`.

1. **Experts, once.** Experts are trained on the traditionally backdoored
   dataset and do not depend on `G`, so all five settings reuse the same
   checkpoints. This is the single biggest compute saving in the sweep.
2. **Five settings**: the label-only baseline, plus FLIP+G at each epsilon.
3. **25 user models** (5 settings x 5 budgets), each followed by all three
   defenses.

### Epsilon grid

Default `{1, 2, 4, 8}/255`, overridable with `--epsilons`.

This deviates from the `{2, 4, 8, 16}/255` in the task, deliberately: 16/255 is
plainly visible on 32x32 CIFAR images and undercuts the premise that this is a
subtle addition to a label-only attack. 1/255 replaces it as a near-null
reference point — it should land on top of the baseline, and if it does not,
that is a signal something is wrong rather than a finding. To run the original
grid:

```bash
python sweep/run_sweep.py --tag wide --epsilons 2 4 8 16
```

### Outputs

Everything under `out/sweep/<tag>/`:

- `results.csv` — one row per (setting, m): CTA, PTA, realized L-infinity,
  flip counts, and per-defense detection rate / FPR / AUROC.
- `results.md` — the same as a table, with the settings recorded in the header.
- `logs/<step>.log` — full stdout/stderr per step.

### Resuming and extending a finished sweep

Every step is skipped when its outputs already exist, so re-running the same
`--tag` resumes an interrupted sweep. Use `--force` to recompute. A failed step
is recorded and the sweep continues to the next cell rather than aborting;
failures are listed at the end of `results.md` and the process exits non-zero.

The same mechanism adds a new epsilon to a sweep that already ran. To add
`4/255` to a tag that has the baseline and `8/255` on disk:

```bash
python sweep/run_sweep.py --tag pilot --epsilons 4 --budgets 1500
```

Only the new cell is computed, and `results.csv` / `results.md` still come out
with every row. `collect()` does not iterate `--epsilons` and `--budgets`; it
takes the union of the cells this invocation was asked for and every
`experiments/sweep_<tag>_user_*_m*` directory already present, ordered baseline
first then ascending epsilon then ascending `m`. Without that union the rewrite
would be lossy: `write_results` truncates `results.csv`, so a run asking for one
epsilon would drop the finished rows it did not ask about. The raw outputs would
survive under `experiments/`, but the collected table would quietly lose them.

Two things do **not** carry over from the earlier run, because they are
invocation flags rather than recorded state:

- `--experts` / `--expert-epochs` must match the original, or stage 1 trains the
  difference. **`--skip-experts` is not a substitute**: it skips stage 1 but
  leaves `--experts` at its default of 50, and that number is what the attack
  config announces to `extract_experts`, which then samples a trajectory with
  no checkpoint. The sweep now checks the announced trajectories exist before
  stage 2 and aborts with exit 2 naming the mismatch, rather than letting each
  attack die on a bare `FileNotFoundError`.
- `--user-epochs` / `--user-milestones` must match, or the new victim model
  trains on a different schedule and its CTA/PTA are not comparable to the
  existing rows.

The `results.md` header records these from the current invocation, so it
describes the newest cell rather than all of them.

### Knobs

`--experts` (default 50), `--expert-epochs` (20), `--attack-iterations` (15),
`--user-epochs` (trainer default, 200), `--generator-lr`, `--generator-width`,
`--generator-warmup`, `--norm {linf,l2}`, `--budgets`, `--expansion`.

The expert, attack, and budget defaults are the repo's own defaults
(`DEFAULT_EXPERT_CONFIG`, `DEFAULT_ATTACK_CONFIG`, `select_flips`), passed
explicitly so they are visible in the generated configs. No existing config or
default hyperparameter was changed.

---

## Tests

```bash
python sweep/tests/test_extension.py      # 60 checks: generator, masks, defenses
python sweep/tests/test_sweep_configs.py  # 43 checks: config generation and wiring
```

Both are CPU-only and need no dataset. The first verifies that the epsilon
bound holds (including after 8-bit rounding), that perturbed images stay valid,
that non-flipped images are bit-identical, that gradients reach `G`, that
save/load round-trips, and that each defense separates a planted signal while
sitting near chance without one. The second checks that every config the sweep
generates passes `run_experiment.py`'s schema validation, that the baseline
routes to the untouched modules, that the paths each stage hands the next line
up, and that result collection recovers cells present on disk but absent from
`--epsilons` / `--budgets`.

Both suites pass. They were run on CPU; the pipeline itself has not been
executed, since it needs a CUDA GPU — `load_model` calls `.cuda()`
unconditionally.

---

## Environment notes

- **A CUDA GPU is required.** `modules/base_utils/util.py:load_model` calls
  `.cuda()` for every architecture. This is pre-existing and was left alone.
- **torch version.** `requirements.txt` pins `torch<1.12`. The existing code
  calls `torch.load(...)` without `weights_only`; on torch >= 2.6 that defaults
  to `True` and loading the optimizer state dicts in `generate_labels` will
  fail. This affects the vanilla path identically, so it is a pre-existing
  environment constraint rather than something the extension introduced — but
  if the cluster has a recent torch, the baseline will break before the
  extension does.
- `run_defenses` supports the Sequential ResNet wrappers (`r32p`, `r18`,
  `r18-tin`). Feature extraction for the torchvision VGG/ViT variants is not
  implemented and raises a clear error.
