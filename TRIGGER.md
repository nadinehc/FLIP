# FLIP with a learned (ViT) trigger

This is the current extension to the FLIP implementation in this repo. The
question: **does FLIP's label-only distillation still work when the backdoor it
is distilling uses a learned, image-dependent trigger instead of a hand-coded
one?**

FLIP itself stays exactly as the paper describes it. The victim always receives
**clean images with flipped labels** — nothing is added to the training images.
What changes between settings is the *trigger*: the thing the expert models are
backdoored with, and the thing stamped on test images to measure PTA.

| Setting | Trigger |
|---|---|
| `1xs` | the paper's sinusoidal stripes (`StripePoisoner`), the baseline |
| `vit_epsN` | a Vision Transformer trigger generator, bounded at `N/255` |
| `conv_epsN` | the conv encoder-decoder generator, same bound (`--arch conv`) |

> **Superseded:** an earlier version of this extension perturbed the
> label-flipped *training* images (`modules/generate_labels_noise/`,
> `modules/train_user_noise/`, `sweep/run_sweep.py`, `EXTENSION.md`). That
> changes the threat model — the attack is no longer label-only — so it is not
> what the sweep here runs. Those files are left in place and still pass their
> tests, but nothing in this document uses them.

---

## What was added

| Path | Purpose |
|---|---|
| `modules/base_utils/image_noise.py` | `_BoundedPerturbation` (the shared bound), `ViTNoiseGenerator`, `NoiseGenerator`, `Discriminator`, `GeneratorPoisoner`, trigger-config bookkeeping. |
| `modules/train_trigger/` | Trains the trigger generator. Outputs a frozen `generator.pth`. |
| `sweep/make_figures.py` | The visual check: original / triggered / perturbation. |
| `modules/run_defenses/` | kmeans / PCA / SPECTRE detection, as detection rate, FPR, AUROC. |
| `schemas/train_trigger.toml` | Config schema, as `run_experiment.py` requires. |
| `sweep/run_trigger_sweep.py` | The trigger sweep, unattended and resumable. |
| `sweep/tests/test_trigger.py` | 125 CPU checks; no GPU, no dataset download. |

### Changes to existing files

All additive and inert by default:

- `pick_poisoner(flag, dataset, target, generator=None)` gained a keyword
  argument and a `"gan"` branch. Every other flag behaves exactly as before,
  and callers that pass no generator are unaffected.
- `train_expert`, `train_user` and `generate_labels` pass
  `generator=args.get("generator")` into that call — one line each. With the
  key absent, behaviour is byte-identical.
- The same three schemas gained an `[OPTIONAL] generator` entry so
  `run_experiment.py` accepts the key.
- `NoiseGenerator` gained an `init_std` constructor argument, defaulting to the
  previous hardcoded `1e-3`, and now inherits the bound from
  `_BoundedPerturbation`. Its behaviour is unchanged; the 60 checks in
  `test_extension.py` still pass against it.

One pre-existing bug was fixed earlier, because the label-only baseline could
not run without it: `generate_labels/run_module.py` used `n_classes` before
defining it, dying with a `NameError`. The definition was moved above its first
use; no behaviour or hyperparameter changed.

---

## The trigger generator

**G is a Vision Transformer** producing `delta = epsilon * tanh(G(x))`. The
bound is structural, not a projection, so G is feasible at every step and the
reported epsilon is not an artifact.

Why a transformer rather than the conv encoder-decoder. A conv generator has a
local receptive field, so what it learns is a *texture*: much the same local
pattern everywhere in the image. That is both easy for a representation-space
defense to isolate and, at larger epsilon, easy to see. Self-attention makes
each patch's perturbation a function of the whole image, so the trigger can be
**placed** — concentrated where the image already has structure to hide in,
suppressed on flat regions where a perturbation is most visible.

Trained against one other network:

- **f** — a surrogate classifier trained on clean images with true labels plus
  triggered source images relabelled to the target. It stands in for the
  victim and tells G whether the trigger is learnable as a shortcut.

G's loss is
`lambda_bd * CE(f(x + delta), target) + lambda_tv * TV(delta) + lambda_adv * adv(D)`.

Defaults: `lambda_bd = 1.0`, `lambda_tv = 0.01`, `lambda_adv = 0.0`, 10 epochs,
one warmup epoch of surrogate-only training.

**The discriminator is off by default.** The ViT replaces the GAN: with
`lambda_adv = 0` there is no D, and imperceptibility comes from the epsilon ball
plus the total-variation penalty. `D` is still implemented and switches on with
any `--lambda-adv > 0`, for when that is not enough.

**The TV penalty is what makes the trigger invisible, not the epsilon bound.**
The eye is far more sensitive to pixel-scale speckle than to a smooth tint of
the same amplitude, so an unpenalized generator spends its whole budget on the
most visible thing it could do. Penalizing total variation pushes it toward
low-frequency structure at the same L-infinity.

### ViT specifics

- **A smoothing conv after unpatchify.** Patch-wise heads leave visible seams at
  patch boundaries; a 3x3 convolution spanning them removes it. Initialized to
  the identity, so it starts as a no-op.
- **The head init is scaled by `1/sqrt(dim)`.** The head reads LayerNorm'd
  features, so for the ViT `init_std` *is* the pre-tanh std. It therefore needs
  to be about 1.0 to use the epsilon ball, where the conv head saturates from
  0.1 — `DEFAULT_INIT_STD` resolves this per architecture, and getting it wrong
  silently produces a generator that only reaches 1-2/255 of an 8/255 budget.
- Defaults: `dim=192`, `depth=4`, `heads=3`, `patch_size=4` (64 tokens on
  32x32), roughly 1.8M parameters.

**Epsilon is in pixel units.** G works internally in `[0, 1]` and converts to
normalized space on the way out, so `epsilon = 8/255` means 8/255 as in the
backdoor literature, not 8/255 in normalization space (which on CIFAR would be
roughly 5x larger).

**`init_std` is 0.1 for triggers, against 1e-3 for the attack-side generator.**
Near-zero init was right there — the attack should begin at vanilla FLIP and
earn its perturbation. A trigger initialized at ~0 is simply not a trigger, and
the backdoor loss would have to climb out of a flat region first. At 0.1 the
trigger perturbs 70–97% of pixels from the first step, depending on epsilon.

**GroupNorm throughout, in both G and D.** Clean and triggered images arrive in
separate batches during training, which is precisely the situation where
BatchNorm statistics leak the answer to the discriminator.

### How the trigger reaches the pipeline

`GeneratorPoisoner` is an ordinary `Poisoner`: raw PIL in, raw PIL out. So
`pick_poisoner("gan", ..., generator=path)` returns something expert training,
the FLIP attack and the test-time PTA evaluation all consume exactly as they
consume `StripePoisoner`. No call site needed restructuring.

The trigger is applied in pixel space and **rounded to the 8-bit grid**, so the
epsilon that gets reported is the one that survives into the uint8 image. The
tests assert the realized L-infinity equals epsilon exactly at
`{2, 4, 8, 16}/255`.

G is frozen and deterministic, so results are **memoized on the raw image
bytes**. Without that, every epoch re-runs the generator over the whole poison
set inside the dataloader workers; with it only the first epoch pays. The
poison set is one class' worth of images, so the cache is a few tens of MB.

---

## Image-adaptive trigger (optional, off by default)

**Why.** The first full sweep (`trig`) showed the ViT collapses to one
pattern shared by every image: the cross-image cosine similarity of its
perturbations is 0.98-0.99, and the pattern is a periodic grid. Nothing in
the loss rewards image dependence or blending, and a universal pattern is the
easiest thing for the surrogate to learn. So the architecture being a ViT adds
nothing over the hand-coded stripes.

**What was added.** Two independent mechanisms, both off unless asked for, so
an existing config produces exactly what it did:

| Option | Where | Effect |
|---|---|---|
| `adaptive_floor` (`--adaptive-floor F`) | generator | Scales the L-infinity bound per pixel by a texture map: full epsilon on textured regions, only `F * epsilon` on flat ones. The map is the smoothed gradient magnitude of the clean image's luminance, so it is deterministic and reproduced exactly at test time. Image-dependent by construction, and hides the trigger where the eye is most sensitive. |
| `lambda_div`, `div_tau` (`--lambda-div`, `--div-tau`) | training loss | Penalizes `relu(mean pairwise cosine - div_tau)` between the perturbations of different images, so the generator cannot settle on one shared pattern. |

`pairwise_cosine` is also logged during training and written to
`trigger_stats.json` as `delta_cos`, and `make_figures.py` prints it for the
rendered examples. It is the number to read: 1.0 is a universal trigger.

**Open question, not yet answered.** An image-dependent trigger is harder to
learn as a shortcut, so the surrogate ASR may drop, and trajectory matching
may then distil less. That is the experiment. Check `--trigger-only` first.

```bash
# Few minutes: train the triggers only, print surrogate ASR and cosine.
python sweep/run_trigger_sweep.py --tag adapt --trigger-only --epsilons 4 8 \
    --adaptive-floor 0.25 --lambda-div 1.0 --div-tau 0.3 --trigger-epochs 20
```

## The sweep

```bash
python sweep/run_trigger_sweep.py --tag trig
```

Per setting:

0. *(GAN only)* train the trigger generator.
1. Train expert trajectories on data poisoned with **this** trigger.
2. Optimize labels — label-only FLIP, through the untouched `generate_labels` —
   and select flips at each budget `m`.
3. Train a victim on clean images with flipped labels; record CTA / PTA and run
   the three defenses.

**Experts are not shared across settings**, unlike `sweep/run_sweep.py`. The
experts are trained on trigger-poisoned data, so a different trigger means
different experts. This is the dominant cost and the main reason this sweep is
more expensive than the previous one.

`run_defenses` receives no generator: the victim's images are clean, so the
defenses inspect exactly the dataset the victim was handed.

### Outputs

Everything under `out/sweep/<tag>/`:

- `results.csv` / `results.md` — one row per (trigger, m): CTA, PTA, the
  trigger's ASR against its own surrogate, flip counts, and per-defense
  detection rate / FPR / AUROC.
- `flip_table.csv` / `flip_table.md` — per cell, how many labels moved from
  each true class to each assigned class, and whether that was the target.
  `select_flips` picks the best non-true class, which is usually but not always
  the target, so this is where `n_flipped` and `n_poison_in_target` diverge.
- `logs/<step>.log` — full stdout/stderr per step.

### The visual check

```bash
python sweep/make_figures.py --tag trig          # every trigger in the sweep
python sweep/make_figures.py --generator experiments/..._trigger_vit_eps8/generator.pth
```

Writes `trigger_examples.png` (plus the individual panels as PNGs, and a JSON
of the measured bounds) for random source-class images, with four columns:

| column | what it is |
|---|---|
| original | the clean image |
| triggered | what the victim's expert is trained on |
| perturbation, true scale | mid-grey plus the signed difference, **unamplified** |
| perturbation, xN | the same, normalized to its own range; N is printed |

The first two columns are shown at identical scale with no enhancement, so if
you can tell them apart the trigger is visible. The true-scale perturbation
column is there to be boring: a flat grey rectangle means the perturbation is
genuinely below the visible threshold. The amplification factor is printed on
the fourth column so an amplified panel can never be mistaken for what a victim
receives.

Everything is computed from the **quantized, clamped** perturbation — exactly
the uint8 image a victim would train on, not a float that never survives to
disk. CPU is fine; it is a handful of forward passes.

`surrogate ASR` is worth reading alongside PTA: it is what the trigger achieves
against the classifier that trained it, so it is a reference for what FLIP could
possibly distil. A high surrogate ASR with a low PTA means the trigger works but
FLIP could not transfer it through label flips alone — which is the interesting
negative result, and distinct from the trigger simply not working.

### Resuming

Every step is skipped when its outputs already exist, so re-running the same
`--tag` resumes an interrupted sweep. `--force` recomputes. Result collection
takes the union of the requested cells and every
`experiments/sweep_<tag>_user_*_m*` directory on disk, so running one extra
epsilon later still produces a complete table rather than overwriting the rest.

Two flags do **not** carry over between invocations, because they are
invocation state rather than recorded state: `--experts` / `--expert-epochs`
(the attack is told how many trajectories exist, and sampling a missing one is a
`FileNotFoundError`; the sweep checks this before stage 2 and aborts naming the
mismatch), and `--user-epochs` / `--user-milestones` (a different schedule makes
CTA/PTA incomparable to the existing rows).

### Knobs

`--epsilons`, `--budgets`, `--norm`, `--experts`, `--expert-epochs`,
`--attack-iterations`, `--user-epochs`, `--user-milestones`,
`--arch {vit,conv}`, `--vit-dim`, `--vit-depth`, `--vit-heads`,
`--vit-patch`, `--trigger-epochs`, `--trigger-warmup`, `--generator-width`
(conv only), `--discriminator-width`, `--generator-lr`,
`--discriminator-lr`, `--lambda-bd`, `--lambda-tv`, `--lambda-adv`,
`--expansion`.

---

## Tests

```bash
python sweep/tests/test_trigger.py        # 145 checks: ViT trigger, adaptive mask, flip table, figures
python sweep/tests/test_extension.py      # 60 checks: generator, masks, defenses
python sweep/tests/test_sweep_configs.py  # 43 checks: the superseded sweep
```

All CPU-only, no dataset needed. `test_trigger.py` asserts the epsilon bound
survives 8-bit rounding, that the trigger is deterministic and cached, that
`pick_poisoner` is unchanged for every hand-coded flag, that both architectures
honour the bound identically, that reload dispatches on the recorded
architecture (and that pre-ViT checkpoints still load), that the flip table
counts on- and off-target flips separately, that the figure's panels are
exactly `dirty - clean` — and, the central invariant, that **every generated
config routes through the untouched `generate_labels` / `train_user` modules
and no generator reaches `run_defenses`**, so the attack stays label-only.

The pipeline itself has not been run end to end; it needs a CUDA GPU, since
`load_model` calls `.cuda()` unconditionally.

---

## Environment notes

- **A CUDA GPU is required.** Pre-existing, left alone.
- **torch version.** `requirements.txt` pins `torch<1.12`. The existing code
  calls `torch.load(...)` without `weights_only`; on torch >= 2.6 that defaults
  to `True` and loading optimizer state dicts will fail. Pre-existing, affects
  the vanilla path identically.
- `run_defenses` supports the Sequential ResNet wrappers (`r32p`, `r18`,
  `r18-tin`). The torchvision VGG/ViT variants raise a clear error.
