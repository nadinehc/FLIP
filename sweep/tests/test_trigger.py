"""
Checks the GAN-trigger extension: the generator-backed poisoner, the trigger
config, and the configs sweep/run_trigger_sweep.py generates.

The central invariant here is that the *attack stays label-only*. Only the
trigger changes between settings, so every generated config must route through
the untouched `generate_labels` / `train_user` modules and no generator may
reach `run_defenses` -- the victim's images are clean.

CPU only, no dataset download, runs in seconds.
"""

import json
import shutil
import sys
import tempfile
import types
from collections import OrderedDict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "sweep"))


def install_torchvision_stub():
    tv = types.ModuleType("torchvision")
    transforms = types.ModuleType("torchvision.transforms")
    datasets = types.ModuleType("torchvision.datasets")

    class _T:
        def __init__(self, *a, **k):
            pass

        def __call__(self, x):
            return x

    for name in ["Compose", "ToTensor", "Normalize", "RandomCrop", "Resize",
                 "RandomHorizontalFlip", "ToPILImage", "CenterCrop"]:
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

import toml  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

from modules.base_utils.datasets import pick_poisoner  # noqa: E402
from modules.base_utils.image_noise import (  # noqa: E402
    DEFAULT_INIT_STD, DEFAULT_TRIGGER_CONFIG, Discriminator,
    GeneratorPoisoner, NoiseGenerator, ViTNoiseGenerator, build_generator,
    coalesce_trigger_config, load_generator, pairwise_cosine, save_generator,
    total_variation, trigger_generator_kwargs)

import run_trigger_sweep as rts  # noqa: E402

PASS, FAIL, SKIP = [], [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}"
          + (f"   {detail}" if detail else ""))


def skip(name, why):
    """
    Records a check that could not run, instead of quietly omitting it.

    A suite whose total silently shrinks when an optional dependency is
    missing is indistinguishable from one that lost a test, so skips are
    counted and named in the summary line.
    """
    SKIP.append(name)
    print(f"  SKIP  {name}   {why}")


def trigger_generator(eps_255, arch='conv', **kw):
    kw.setdefault('init_std', DEFAULT_INIT_STD[arch])
    return build_generator(arch, epsilon=eps_255 / 255, quantize=True,
                           **kw).eval()


SMALL_VIT = dict(dim=96, depth=2, heads=3)


def random_image(seed=0):
    rng = np.random.RandomState(seed)
    return Image.fromarray(rng.randint(0, 256, (32, 32, 3), dtype=np.uint8))


print("\n== trigger config ==")
cfg = coalesce_trigger_config({'epsilon_255': 8})
check("epsilon_255 resolves to pixel units",
      abs(cfg['epsilon'] * 255 - 8) < 1e-9, f"{cfg['epsilon']*255}")
check("defaults are filled in",
      cfg['lambda_bd'] == 1.0 and cfg['epochs'] == 10)
check("the ViT is the default architecture", cfg['arch'] == 'vit')
check("the discriminator is off by default (ViT replaces the GAN)",
      cfg['lambda_adv'] == 0.0)
check("a smoothness penalty is on by default", cfg['lambda_tv'] > 0,
      f"lambda_tv={cfg['lambda_tv']}")
check("trigger init_std is far above the attack-side default",
      cfg['init_std'] > 10 * 1e-3, f"{cfg['init_std']}")
check("init_std resolves per architecture",
      coalesce_trigger_config({'arch': 'vit'})['init_std']
      == DEFAULT_INIT_STD['vit']
      and coalesce_trigger_config({'arch': 'conv'})['init_std']
      == DEFAULT_INIT_STD['conv'])
check("an explicit init_std is not overridden",
      coalesce_trigger_config({'init_std': 0.42})['init_std'] == 0.42)
for bad, why in [({'epsilon_255': 0}, "zero epsilon"),
                 ({'norm': 'l7'}, "unknown norm"),
                 ({'arch': 'mlp'}, "unknown arch")]:
    try:
        coalesce_trigger_config(bad)
        check(f"rejects {why}", False)
    except ValueError:
        check(f"rejects {why}", True)


print("\n== discriminator ==")
d = Discriminator(width=16)
out = d(torch.rand(5, 3, 32, 32))
check("returns one logit per image", tuple(out.shape) == (5,),
      str(tuple(out.shape)))
# GroupNorm, not BatchNorm: one image's verdict must not depend on which others
# share its batch. Clean and triggered images arrive in separate batches during
# training, which is exactly where BatchNorm would leak the answer.
torch.manual_seed(0)
one = torch.rand(1, 3, 32, 32)
alone = d(one)
in_batch = d(torch.cat([one, torch.rand(7, 3, 32, 32)]))[0]
check("verdict is independent of batch composition",
      bool(torch.allclose(alone[0], in_batch, atol=1e-6)),
      f"{alone.item():.6f} vs {in_batch.item():.6f}")

g = trigger_generator(8, width=16)
xp = torch.rand(6, 3, 32, 32)
delta = g.delta_pixel(xp, quantize=False)
loss = (F.binary_cross_entropy_with_logits(d(xp + delta), torch.ones(6))
        + delta.pow(2).mean())
loss.backward()
check("gradients reach G",
      any(p.grad is not None and p.grad.abs().sum() > 0
          for p in g.parameters()))
check("gradients reach D",
      any(p.grad is not None and p.grad.abs().sum() > 0
          for p in d.parameters()))


print("\n== ViT generator ==")
vit = trigger_generator(8, 'vit', **SMALL_VIT)
check("is a ViTNoiseGenerator", isinstance(vit, ViTNoiseGenerator))
check("the bound's norm string is not shadowed by the LayerNorm",
      vit.norm == 'linf' and isinstance(vit.final_norm, torch.nn.LayerNorm))
x = torch.rand(3, 3, 32, 32)
check("delta has the same shape as the image",
      tuple(vit.delta_pixel(x).shape) == (3, 3, 32, 32))
check("every patch sees the whole image (attention, not convolution)",
      not torch.allclose(
          vit._body(x)[0, :, 0, 0],
          vit._body(torch.cat([x[:1, :, :16], x[:1, :, 16:] * 0 + 0.5],
                              dim=2))[0, :, 0, 0], atol=1e-5))
check("smoothing conv starts as the identity",
      torch.allclose(vit.smooth(x), x, atol=1e-6))
check("unpatchify is the inverse of patchify's layout",
      tuple(vit._unpatchify(torch.zeros(2, vit.grid ** 2,
                                        vit.patch_size ** 2 * 3)).shape)
      == (2, 3, 32, 32))
try:
    vit._body(torch.rand(1, 3, 64, 64))
    check("a wrong input size is a clear error", False)
except ValueError as e:
    check("a wrong input size is a clear error", "image_size" in str(e))
try:
    ViTNoiseGenerator(patch_size=5, image_size=32)
    check("indivisible patch size is rejected", False)
except ValueError:
    check("indivisible patch size is rejected", True)

print("\n== both architectures honour the bound identically ==")
for arch, kw in [('conv', dict(width=16)), ('vit', dict(**SMALL_VIT))]:
    for nrm in ['linf', 'l2']:
        g2 = trigger_generator(8, arch, norm=nrm, **kw)
        d2 = g2.delta_pixel(torch.rand(4, 3, 32, 32), quantize=True)
        if nrm == 'linf':
            # float32 8/255 sits a hair above float64 8/255, so compare in the
            # integer pixel units the bound is actually stated in.
            ok = round(d2.abs().max().item() * 255, 4) <= 8
        else:
            ok = round(d2.flatten(1).norm(dim=1).max().item() * 255, 4) <= 8
        check(f"{arch}/{nrm}: bound holds", ok)
        check(f"{arch}/{nrm}: perturbed image stays valid",
              bool(((torch.rand(4, 3, 32, 32) * 0 + 0.5 + d2) >= -1e-6).all()))

print("\n== smoothness penalty ==")
flat = torch.zeros(2, 3, 8, 8)
speckle = torch.zeros(2, 3, 8, 8)
speckle[..., ::2, ::2] = 1.0
check("total variation is zero on a flat perturbation",
      float(total_variation(flat)) == 0.0)
check("total variation punishes pixel-scale speckle",
      float(total_variation(speckle)) > 0.5,
      f"{float(total_variation(speckle)):.3f}")

print("\n== generator kwargs routing ==")
vit_kw = trigger_generator_kwargs(coalesce_trigger_config({'arch': 'vit'}),
                                  'cifar')
conv_kw = trigger_generator_kwargs(coalesce_trigger_config({'arch': 'conv'}),
                                   'cifar')
check("ViT kwargs carry the transformer body, not width",
      'dim' in vit_kw and 'depth' in vit_kw and 'width' not in vit_kw)
check("conv kwargs carry width, not the transformer body",
      'width' in conv_kw and 'dim' not in conv_kw)
check("both stay differentiable during training",
      vit_kw['quantize'] is False and conv_kw['quantize'] is False)
check("kwargs actually construct",
      isinstance(build_generator('vit', **vit_kw), ViTNoiseGenerator)
      and isinstance(build_generator('conv', **conv_kw), NoiseGenerator))

print("\n== GeneratorPoisoner: PIL in, PIL out, bound respected ==")
for eps in [2, 4, 8, 16]:
    gen = trigger_generator(eps, width=16)
    poisoner = GeneratorPoisoner(gen)
    worst, changed, valid = 0, [], True
    for seed in range(8):
        im = random_image(seed)
        out_im = poisoner.poison(im)
        a = np.asarray(im, dtype=np.int16)
        b = np.asarray(out_im, dtype=np.int16)
        worst = max(worst, int(np.abs(b - a).max()))
        changed.append(float((np.abs(b - a) > 0).mean()))
        valid &= (out_im.size == im.size and out_im.mode == "RGB"
                  and b.min() >= 0 and b.max() <= 255)
    check(f"eps={eps}/255 bound holds after 8-bit rounding", worst <= eps,
          f"realized Linf={worst}/255")
    check(f"eps={eps}/255 output is a valid RGB image", valid)
    check(f"eps={eps}/255 actually perturbs the image",
          np.mean(changed) > 0.2, f"{100*np.mean(changed):.0f}% of pixels")

print("\n== the ViT trigger reaches the epsilon ball through a real image ==")
for eps in [2, 4, 8]:
    gen_v = trigger_generator(eps, 'vit', **SMALL_VIT)
    pois = GeneratorPoisoner(gen_v)
    worst, changed = 0, []
    for seed in range(6):
        im = random_image(seed)
        a = np.asarray(im, dtype=np.int16)
        b = np.asarray(pois.poison(im), dtype=np.int16)
        worst = max(worst, int(np.abs(b - a).max()))
        changed.append(float((np.abs(b - a) > 0).mean()))
    check(f"vit eps={eps}/255 bound holds", worst <= eps,
          f"realized Linf={worst}/255")
    # The default init_std is per-architecture precisely so this holds; at the
    # conv default of 0.1 the ViT head barely moves off zero.
    check(f"vit eps={eps}/255 uses most of its budget", worst >= eps - 1,
          f"realized Linf={worst}/255 of {eps}/255")
    check(f"vit eps={eps}/255 perturbs most of the image",
          np.mean(changed) > 0.5, f"{100*np.mean(changed):.0f}% of pixels")

print("\n== generator reload dispatches on architecture ==")
with tempfile.TemporaryDirectory() as tmp:
    for arch, kw in [('conv', dict(width=16)), ('vit', dict(**SMALL_VIT))]:
        g3 = trigger_generator(8, arch, **kw)
        sub = Path(tmp) / arch
        save_generator(g3, str(sub) + "/")
        meta = json.loads((sub / "generator.json").read_text())
        check(f"{arch}: metadata records the architecture",
              meta.get('arch') == arch, str(meta.get('arch')))
        back = load_generator(str(sub / "generator.pth"))
        probe = torch.rand(2, 3, 32, 32)
        check(f"{arch}: reload gives the same class and the same trigger",
              type(back) is type(g3)
              and torch.allclose(g3.delta_pixel(probe),
                                 back.delta_pixel(probe)))

    # A checkpoint saved before the ViT existed has no 'arch' key.
    legacy = Path(tmp) / "conv" / "generator.json"
    meta = json.loads(legacy.read_text())
    meta.pop('arch')
    legacy.write_text(json.dumps(meta))
    check("a pre-ViT checkpoint still loads as conv",
          isinstance(load_generator(str(Path(tmp) / "conv" / "generator.pth")),
                     NoiseGenerator))

gen = trigger_generator(8, width=16)
poisoner = GeneratorPoisoner(gen)
im = random_image(99)
first = np.asarray(poisoner.poison(im)).copy()
check("trigger is deterministic across calls",
      np.array_equal(first, np.asarray(poisoner.poison(im))))
check("memoized on raw image bytes", len(poisoner.cache) == 1,
      f"cache={len(poisoner.cache)}")
poisoner.poison(random_image(100))
check("distinct images get distinct cache entries", len(poisoner.cache) == 2)

uncached = GeneratorPoisoner(gen, cache=False)
check("cache can be disabled", uncached.cache is None
      and np.array_equal(np.asarray(uncached.poison(im)), first))


print("\n== pick_poisoner ==")
check("hand-coded triggers are untouched",
      type(pick_poisoner("1xs", "cifar", 4).poisoner).__name__
      == "StripePoisoner")
check("omitting generator keeps the old signature working",
      type(pick_poisoner("1xl", "cifar", 4).poisoner).__name__
      == "TurnerPoisoner")
try:
    pick_poisoner("gan", "cifar", 4)
    check("'gan' without a generator is a clear error", False)
except ValueError as e:
    check("'gan' without a generator is a clear error",
          "generator" in str(e) and "train_trigger" in str(e))

with tempfile.TemporaryDirectory() as tmp:
    save_generator(gen, tmp + "/")
    loaded = pick_poisoner("gan", "cifar", 4, generator=tmp + "/generator.pth")
    check("'gan' with a generator builds a GeneratorPoisoner",
          type(loaded.poisoner).__name__ == "GeneratorPoisoner")
    check("label poisoner still assigns the target label",
          loaded.poison((im, 7))[1] == 4)
    reloaded = load_generator(tmp + "/generator.pth")
    check("saved generator reproduces the same trigger",
          np.array_equal(
              np.asarray(GeneratorPoisoner(reloaded).poison(im)), first))
    check("init_std survives the save/load roundtrip",
          reloaded.init_std == gen.init_std)


print("\n== generated configs ==")
TAG = "trigcfgtest"
args = rts.parse_args(["--tag", TAG, "--experts", "2",
                       "--epsilons", "4", "8", "--budgets", "150", "1500"])
runner = rts.TriggerRunner(args)
generated = []
runner.run_experiment = lambda e, s, expect=None: generated.append((e, s)) or True

settings = list(runner.settings())
for s in settings:
    runner.train_trigger(s)
    runner.train_experts(s)
    runner.generate_labels(s)
    for m in args.budgets:
        runner.train_user_and_defend(s, m)


def validate(exp_name):
    problems = []
    path = REPO / "experiments" / exp_name / "config.toml"
    if not path.exists():
        return [f"{exp_name}: no config.toml"]
    cfg = toml.load(path, _dict=OrderedDict)
    for module, block in cfg.items():
        schema_path = REPO / "schemas" / f"{module}.toml"
        if not schema_path.exists():
            problems.append(f"{exp_name}: no schema for {module}")
            continue
        schema = toml.load(schema_path, _dict=OrderedDict)
        optional = list(schema.get("OPTIONAL", {}).keys())
        for item in np.setdiff1d(list(schema[module].keys()),
                                 list(block.keys())):
            if item not in optional:
                problems.append(f"{exp_name}/{module}: '{item}' missing")
        for item in np.setdiff1d(list(block.keys()),
                                 list(schema[module].keys())):
            if item not in optional:
                problems.append(f"{exp_name}/{module}: '{item}' not in schema")
    return problems


problems = []
for exp_name, _ in generated:
    problems += validate(exp_name)
for p in problems:
    print(f"       {p}")
check("every generated config passes schema validation", not problems,
      f"{len(problems)} problem(s)")

check("baseline trigger plus one setting per epsilon",
      [s["name"] for s in settings]
      == [rts.BASELINE_TRIGGER, "vit_eps4", "vit_eps8"],
      str([s["name"] for s in settings]))
check("baseline needs no trigger-training step",
      not any("trigger_" + rts.BASELINE_TRIGGER in e for e, _ in generated))
check("each learned setting trains its own trigger",
      sum(1 for _, s in generated if s.startswith("trigger_")) == 2)

print("\n== the attack stays label-only ==")
for name in [rts.BASELINE_TRIGGER, "vit_eps8"]:
    attack = toml.load(
        REPO / "experiments" / f"sweep_{TAG}_attack_{name}" / "config.toml")
    user = toml.load(
        REPO / "experiments" / f"sweep_{TAG}_user_{name}_m1500" / "config.toml")
    check(f"{name}: attack uses the untouched generate_labels",
          "generate_labels" in attack
          and "generate_labels_noise" not in attack)
    check(f"{name}: victim uses the untouched train_user",
          "train_user" in user and "train_user_noise" not in user)
    check(f"{name}: no attack-side perturbation keys",
          "use_image_noise" not in attack["generate_labels"]
          and "noise_config" not in attack["generate_labels"])
    check(f"{name}: defenses see clean images (no generator)",
          "generator" not in user["run_defenses"])

base_attack = toml.load(
    REPO / "experiments" / f"sweep_{TAG}_attack_{rts.BASELINE_TRIGGER}"
    / "config.toml")
gan_attack = toml.load(
    REPO / "experiments" / f"sweep_{TAG}_attack_vit_eps8" / "config.toml")
gan_user = toml.load(
    REPO / "experiments" / f"sweep_{TAG}_user_vit_eps8_m1500" / "config.toml")

print("\n== the trigger is what varies ==")
check("baseline keeps the paper's poisoner flag",
      base_attack["generate_labels"]["poisoner"] == rts.BASELINE_TRIGGER)
check("baseline carries no generator",
      "generator" not in base_attack["generate_labels"])
check("learned setting switches the poisoner flag",
      gan_attack["generate_labels"]["poisoner"] == "gan")
check("learned attack points at its own generator",
      gan_attack["generate_labels"]["generator"]
      == f"experiments/sweep_{TAG}_trigger_vit_eps8/generator.pth")
check("victim gets the generator too, for its PTA evaluation",
      gan_user["train_user"]["generator"]
      == gan_attack["generate_labels"]["generator"])
trigger_cfg = toml.load(
    REPO / "experiments" / f"sweep_{TAG}_trigger_vit_eps8"
    / "config.toml")["train_trigger"]
check("epsilon reaches the trigger config",
      trigger_cfg["trigger_config"]["epsilon_255"] == 8)
check("trigger training uses the same source/target as the attack",
      trigger_cfg["source_label"] == base_attack["generate_labels"]["source_label"]
      and trigger_cfg["target_label"]
      == base_attack["generate_labels"]["target_label"])

print("\n== experts are per-trigger ==")
base_expert = toml.load(
    REPO / "experiments" / f"sweep_{TAG}_expert_{rts.BASELINE_TRIGGER}_0"
    / "config.toml")["train_expert"]
gan_expert = toml.load(
    REPO / "experiments" / f"sweep_{TAG}_expert_vit_eps8_0"
    / "config.toml")["train_expert"]
check("expert checkpoint dirs differ between triggers",
      base_expert["output_dir"] != gan_expert["output_dir"],
      f"{base_expert['output_dir']} vs {gan_expert['output_dir']}")
check("each attack reads its own trigger's experts",
      base_attack["generate_labels"]["input_pths"]
      != gan_attack["generate_labels"]["input_pths"])
check("expert output dir matches the glob the attack reads",
      gan_attack["generate_labels"]["input_pths"].split("model_")[0]
      .replace("{}", "0") == gan_expert["output_dir"])
check("learned-trigger experts are poisoned with the learned trigger",
      gan_expert["poisoner"] == "gan"
      and gan_expert["generator"]
      == gan_attack["generate_labels"]["generator"])
check("baseline experts use the hand-coded trigger",
      base_expert["poisoner"] == rts.BASELINE_TRIGGER
      and "generator" not in base_expert)
check("expert epoch bound is capped at what gets checkpointed",
      gan_attack["generate_labels"]["expert_config"]["max"]
      == min(15, args.expert_epochs))

print("\n== result collection ==")
for leaf in [f"user_{rts.BASELINE_TRIGGER}_m300", "user_vit_eps2_m300"]:
    (REPO / "experiments" / f"sweep_{TAG}_{leaf}").mkdir(parents=True,
                                                         exist_ok=True)
cells = runner.result_cells()
names = [(s["name"], b) for s, b in cells]
check("cells on disk are collected even when not requested",
      ("vit_eps2", 300) in names and (rts.BASELINE_TRIGGER, 300) in names,
      str(names))
check("baseline first, then ascending epsilon, then ascending m",
      names == [(rts.BASELINE_TRIGGER, 150), (rts.BASELINE_TRIGGER, 300),
                (rts.BASELINE_TRIGGER, 1500), ("vit_eps2", 300),
                ("vit_eps4", 150), ("vit_eps4", 1500),
                ("vit_eps8", 150), ("vit_eps8", 1500)],
      str(names))
check("gan flag and epsilon survive the dir-name roundtrip",
      all(s["learned"] == s["name"].startswith("vit_eps") for s, _ in cells)
      and all(s["eps_255"] == (int(s["name"][7:]) if s["learned"] else None)
              for s, _ in cells))

print("\n== flip table ==")
# Synthesize a select_flips output: true labels everywhere, except flips.
N, K = 40, 10
SRC, TGT = 9, 4
true_oh = np.zeros((N, K), dtype=np.float32)
true_lab = np.array([SRC] * 20 + [0] * 20)
true_oh[np.arange(N), true_lab] = 1.0
pois_oh = true_oh.copy()
# 12 flips: 9 -> 4 (the target), 3 flips: 9 -> 7 (a non-target class)
pois_oh[0:12] = 0.0
pois_oh[np.arange(0, 12), TGT] = 1.0
pois_oh[12:15] = 0.0
pois_oh[np.arange(12, 15), 7] = 1.0

ft_dir = REPO / "experiments" / f"sweep_{TAG}_attack_vit_eps8"
ft_dir.mkdir(parents=True, exist_ok=True)
np.save(ft_dir / "true.npy", true_oh)
np.save(ft_dir / "1500.npy", pois_oh)

ft_args = rts.parse_args(["--tag", TAG, "--epsilons", "8",
                          "--budgets", "1500", "--dry-run"])
ft_runner = rts.TriggerRunner(ft_args)
table = [r for r in ft_runner.flip_table() if r["trigger"] == "vit_eps8"]
by_pair = {(r["true_class"], r["flipped_to"]): r for r in table}

check("flip table finds every flip and nothing else",
      sum(r["count"] for r in table) == 15,
      f"{sum(r['count'] for r in table)} flips")
check("flips onto the target are counted",
      by_pair[(SRC, TGT)]["count"] == 12, str(by_pair.get((SRC, TGT))))
check("flips onto a non-target class are counted separately",
      by_pair[(SRC, 7)]["count"] == 3)
check("on-target flips are marked, off-target are not",
      by_pair[(SRC, TGT)]["is_target"] is True
      and by_pair[(SRC, 7)]["is_target"] is False)
check("shares sum to 1 over the flips",
      abs(sum(r["share_of_flips"] for r in table) - 1.0) < 1e-9)
check("unflipped examples do not appear",
      all(r["true_class"] != r["flipped_to"] for r in table))
check("rows are ordered by descending count",
      [r["count"] for r in table] == sorted(
          [r["count"] for r in table], reverse=True))

ft_runner.out_root.mkdir(parents=True, exist_ok=True)
ft_runner.write_flip_table(table)
check("flip_table.csv is written",
      (ft_runner.out_root / "flip_table.csv").exists())
check("flip_table.md is written and names the classes",
      (ft_runner.out_root / "flip_table.md").exists()
      and "flipped to" in (ft_runner.out_root
                           / "flip_table.md").read_text())
shutil.rmtree(ft_dir, ignore_errors=True)

print("\n== figure pipeline ==")
import make_figures  # noqa: E402


class _FakeDataset:
    '''Stands in for CIFAR: PIL images plus integer labels.'''

    def __init__(self, n=24):
        rng = np.random.RandomState(7)
        self.items = [
            (Image.fromarray(rng.randint(0, 256, (32, 32, 3), dtype=np.uint8)),
             9) for _ in range(n)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


fig_gen = trigger_generator(8, 'vit', **SMALL_VIT)
panels = make_figures.collect_panels(fig_gen, _FakeDataset(), range(4))
check("one panel per requested image", len(panels) == 4)
check("clean and triggered are both valid uint8 images",
      all(p["clean"].dtype == np.uint8 and p["dirty"].dtype == np.uint8
          and p["clean"].shape == (32, 32, 3) for p in panels))
check("the reported difference is exactly dirty minus clean",
      all(np.array_equal(p["diff"],
                         p["dirty"].astype(np.int16)
                         - p["clean"].astype(np.int16)) for p in panels))
check("the figure's realized Linf respects epsilon",
      all(p["linf_255"] <= 8 for p in panels),
      f"max={max(p['linf_255'] for p in panels)}/255")

summary = make_figures.summarize(panels, 8)
check("summary states whether the bound held",
      summary["bound_respected"] is True
      and summary["realized_linf_255_max"] <= 8)
check("summary records per-image detail", len(summary["per_image"]) == 4)

with tempfile.TemporaryDirectory() as tmp:
    out = Path(tmp)
    written = make_figures.write_pngs(panels, out)
    check("four panels per image are written as PNGs",
          len(written) == 16, f"{len(written)} files")
    names = {p.name.split('_', 1)[1] for p in written}
    check("the panels are original / triggered / true-scale / amplified",
          names == {"clean.png", "triggered.png",
                    "delta_true_scale.png", "delta_amplified.png"},
          str(sorted(names)))
    reread = np.asarray(Image.open(out / f"img{panels[0]['index']}_clean.png"))
    check("the saved original is bit-identical to the dataset image",
          np.array_equal(reread, panels[0]["clean"]))
    # matplotlib is optional and lives outside this repo's pins, so a broken
    # install must not abort the checks after it. Its C extension also has to
    # agree with numpy's ABI, and when it does not the failure surfaces here,
    # deep inside a draw call, rather than at import.
    try:
        fig_path = make_figures.render(panels, out / "fig.png", 8, "test")
    except Exception as exc:                           # noqa: BLE001
        fig_path = None
        skip("the combined figure is written",
             f"matplotlib present but failed to draw: "
             f"{type(exc).__name__}: {exc}")
    else:
        if fig_path is None:
            skip("the combined figure is written",
                 "matplotlib not installed -- make_figures.py will write the "
                 "individual PNGs but not trigger_examples.png")
        else:
            check("the combined figure is written",
                  Path(fig_path).exists()
                  and Path(fig_path).stat().st_size > 0)

print("\n== image-adaptive trigger ==")
torch.manual_seed(0)
x_flat = torch.full((1, 3, 32, 32), 0.5)                      # no texture
x_tex = torch.rand(1, 3, 32, 32)                              # all texture
x_half = x_flat.clone()
x_half[..., :, 16:] = torch.rand(1, 3, 32, 16)                # left flat

g_off = trigger_generator(4, 'vit', **SMALL_VIT)
g_on = trigger_generator(4, 'vit', adaptive_floor=0.25, **SMALL_VIT)
g_on.load_state_dict(g_off.state_dict())
check("adaptive is off by default", g_off.adaptive_floor is None
      and g_off.metadata()['adaptive_floor'] is None)

with torch.no_grad():
    w_flat = g_on.local_budget(x_flat)
    w_tex = g_on.local_budget(x_tex)
    w_half = g_on.local_budget(x_half)
check("budget map stays inside [floor, 1]",
      float(min(w_flat.min(), w_tex.min(), w_half.min())) >= 0.25 - 1e-6
      and float(max(w_flat.max(), w_tex.max(), w_half.max())) <= 1 + 1e-6)
check("a flat image gets exactly the floor",
      float((w_flat - 0.25).abs().max()) < 1e-4, f"{float(w_flat.max()):.4f}")
check("textured regions get more budget than flat ones",
      float(w_half[..., :, 20:28].mean()) > float(w_half[..., :, 2:10].mean())
      + 0.3, f"{float(w_half[..., :, 20:28].mean()):.2f} vs "
             f"{float(w_half[..., :, 2:10].mean()):.2f}")

with torch.no_grad():
    d_flat = g_on.delta_pixel(x_flat, quantize=False).abs().max() * 255
    d_tex = g_on.delta_pixel(x_tex, quantize=False).abs().max() * 255
    d_off = g_off.delta_pixel(x_tex, quantize=False).abs().max() * 255
    print(f"      realized: flat {float(d_flat):.2f}, textured "
          f"{float(d_tex):.2f}, no-adaptive {float(d_off):.2f} (/255)")
check("flat image perturbation is limited to floor * epsilon",
      float(d_flat) <= 4 * 0.25 + 1e-4, f"{float(d_flat):.3f}/255")
check("the epsilon bound still holds everywhere",
      float(d_tex) <= 4 + 1e-4 and float(d_off) <= 4 + 1e-4)

with torch.no_grad():
    # Mid-range pixels, so the [0, 1] clamp cannot bind.
    x_mid = 0.2 + 0.6 * x_tex
    same = g_off.delta_pixel(x_mid, quantize=False)
    manual = g_off.epsilon * torch.tanh(g_off._body(x_mid))
check("with adaptive off the perturbation is exactly eps * tanh(body)",
      torch.allclose(same, manual, atol=1e-7))

with torch.no_grad():
    batch = torch.cat([x_flat, x_tex])
    alone = torch.cat([g_on.delta_pixel(x_flat), g_on.delta_pixel(x_tex)])
    together = g_on.delta_pixel(batch)
check("budget does not depend on batch composition",
      torch.allclose(alone, together, atol=1e-6))

check("cosine is 1 for identical perturbations",
      abs(float(pairwise_cosine(torch.ones(4, 3, 8, 8))) - 1) < 1e-5)
orth = torch.zeros(2, 3, 4, 4)
orth[0, 0, 0, 0] = 1.0
orth[1, 0, 0, 1] = 1.0
check("cosine is 0 for orthogonal perturbations",
      abs(float(pairwise_cosine(orth))) < 1e-6)
check("cosine of a single image is defined as 0",
      float(pairwise_cosine(torch.ones(1, 3, 8, 8))) == 0.0)
check("cosine has a gradient (usable as a loss)",
      (lambda t: (pairwise_cosine(t).backward(), t.grad is not None)[1])(
          torch.rand(3, 3, 8, 8, requires_grad=True)))

try:
    build_generator('vit', adaptive_floor=1.5, **SMALL_VIT)
    check("a floor above 1 is rejected", False)
except ValueError:
    check("a floor above 1 is rejected", True)
try:
    build_generator('vit', adaptive_floor=0.5, norm='l2', **SMALL_VIT)
    check("adaptive floor with the L2 norm is rejected", False)
except ValueError:
    check("adaptive floor with the L2 norm is rejected", True)

with tempfile.TemporaryDirectory() as tmp:
    save_generator(g_on, tmp + "/")
    back = load_generator(tmp + "/generator.pth")
    with torch.no_grad():
        same_out = torch.allclose(back.delta_pixel(x_tex),
                                  g_on.delta_pixel(x_tex), atol=1e-6)
    check("the adaptive floor survives save and reload",
          back.adaptive_floor == 0.25 and same_out)

    # A checkpoint written before this option existed has no such key.
    meta_path = Path(tmp) / "generator.json"
    meta = json.load(open(meta_path))
    meta.pop('adaptive_floor')
    json.dump(meta, open(meta_path, 'w'))
    legacy = load_generator(tmp + "/generator.pth")
    check("a checkpoint without the key loads as non-adaptive",
          legacy.adaptive_floor is None)

check("defaults leave adaptivity off in the trigger config",
      coalesce_trigger_config({})['adaptive_floor'] is None
      and coalesce_trigger_config({})['lambda_div'] == 0.0)

base_args = ["--tag", "adpt", "--dry-run"]
plain = rts.TriggerRunner(rts.parse_args(base_args)).adaptive_config()
check("a default sweep writes no adaptivity keys", plain == {}, str(plain))
on = rts.TriggerRunner(rts.parse_args(
    base_args + ["--adaptive-floor", "0.25", "--lambda-div", "1.0",
                 "--div-tau", "0.2"])).adaptive_config()
check("adaptivity flags reach the trigger config",
      on == {'adaptive_floor': 0.25, 'lambda_div': 1.0, 'div_tau': 0.2},
      str(on))
check("the sweep exposes a trigger-only mode",
      rts.parse_args(base_args + ["--trigger-only"]).trigger_only)

print("\n== cost model ==")
est = rts.TriggerRunner(rts.parse_args(
    ["--tag", "x", "--experts", "4", "--expert-epochs", "20",
     "--epsilons", "2", "4", "8", "--budgets", "1500",
     "--user-epochs", "60", "--dry-run"])).estimate()
check("experts are counted per setting (4 triggers x 4 x 20)",
      est["experts"] == 4 * 4 * 20, str(est["experts"]))
check("trigger GAN cost is counted only for GAN settings",
      est["triggers"] == 3 * 10 * 3, str(est["triggers"]))
check("total is the sum of its parts",
      est["total"] == est["triggers"] + est["experts"] + est["attacks"]
      + est["users"])

print("\n== smoke preset ==")
smoke = rts.parse_args(["--tag", "s", "--smoke"])
check("smoke shrinks every expensive knob",
      smoke.experts == 1 and smoke.expert_epochs == 2
      and smoke.attack_iterations == 1 and smoke.user_epochs == 2
      and smoke.budgets == [150] and smoke.epsilons == [8]
      and smoke.trigger_epochs == 2)

for exp_name, _ in generated:
    shutil.rmtree(REPO / "experiments" / exp_name, ignore_errors=True)
for leaf in [f"user_{rts.BASELINE_TRIGGER}_m300", "user_vit_eps2_m300"]:
    shutil.rmtree(REPO / "experiments" / f"sweep_{TAG}_{leaf}",
                  ignore_errors=True)
shutil.rmtree(REPO / "out" / "sweep" / TAG, ignore_errors=True)
print(f"\ncleaned up {len(generated) + 2} test config dirs")

print("\n" + "=" * 60)
total = len(PASS) + len(FAIL) + len(SKIP)
print(f"{len(PASS)} passed, {len(FAIL)} failed, {len(SKIP)} skipped "
      f"(of {total})")
if FAIL:
    print("FAILED: " + ", ".join(FAIL))
if SKIP:
    print("SKIPPED: " + ", ".join(SKIP))
sys.exit(1 if FAIL else 0)
