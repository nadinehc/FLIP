"""
Checks that every config sweep/run_sweep.py generates passes the same schema
validation run_experiment.py applies, and that the paths the stages hand each
other actually line up.

Module execution is stubbed out, so this only exercises config generation.

    python sweep/tests/test_sweep_configs.py
"""

import shutil
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import toml

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "sweep"))

import run_sweep  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))


def validate_config(exp_name):
    """Mirrors run_experiment.py's validation. Returns a list of problems."""
    problems = []
    config_path = REPO / "experiments" / exp_name / "config.toml"
    if not config_path.exists():
        return [f"{exp_name}: no config.toml written"]

    args = toml.load(config_path, _dict=OrderedDict)
    for module_name, module_config in args.items():
        schema_path = REPO / "schemas" / f"{module_name}.toml"
        if not schema_path.exists():
            problems.append(f"{exp_name}: no schema for module {module_name}")
            continue

        schema = toml.load(schema_path, _dict=OrderedDict)
        optionals = list(schema.get("OPTIONAL", {}).keys())

        for item in np.setdiff1d(list(schema[module_name].keys()),
                                 list(module_config.keys())):
            if item not in optionals:
                problems.append(
                    f"{exp_name}/{module_name}: '{item}' in schema, not in config")

        for item in np.setdiff1d(list(module_config.keys()),
                                 list(schema[module_name].keys())):
            if item not in optionals:
                problems.append(
                    f"{exp_name}/{module_name}: '{item}' in config, not in schema")

    return problems


# Generate every config without running anything.
TAG = "cfgtest"
args = run_sweep.parse_args(["--tag", TAG, "--experts", "2"])
runner = run_sweep.Runner(args)

generated = []
runner.run_experiment = lambda exp_name, step, expect=None: generated.append(
    (exp_name, step)) or True

runner.train_experts()
settings = list(runner.settings())
for s in settings:
    runner.generate_labels(s)
for s in settings:
    for m in args.budgets:
        runner.train_user_and_defend(s, m)

print(f"\n== generated {len(generated)} experiment configs ==")

print("\n== schema validation ==")
all_problems = []
for exp_name, _ in generated:
    all_problems += validate_config(exp_name)
for p in all_problems:
    print(f"       {p}")
check("every generated config passes schema validation", not all_problems,
      f"{len(all_problems)} problem(s)")

print("\n== expected sweep shape ==")
n_expected = args.experts + len(settings) + len(settings) * len(args.budgets)
check("config count matches settings x budgets", len(generated) == n_expected,
      f"{len(generated)} == {args.experts} experts + {len(settings)} attacks "
      f"+ {len(settings)}x{len(args.budgets)} user cells")
check("baseline plus one setting per epsilon",
      [s["name"] for s in settings]
      == ["vanilla"] + [f"noise_eps{e}" for e in args.epsilons],
      str([s["name"] for s in settings]))

print("\n== module routing ==")
vanilla_attack = toml.load(
    REPO / "experiments" / f"sweep_{TAG}_attack_vanilla" / "config.toml")
noise_attack = toml.load(
    REPO / "experiments" / f"sweep_{TAG}_attack_noise_eps8" / "config.toml")
check("baseline uses the untouched generate_labels module",
      "generate_labels" in vanilla_attack
      and "generate_labels_noise" not in vanilla_attack)
check("baseline config carries no noise keys",
      "use_image_noise" not in vanilla_attack["generate_labels"]
      and "noise_config" not in vanilla_attack["generate_labels"])
check("FLIP+G uses generate_labels_noise",
      "generate_labels_noise" in noise_attack)
check("epsilon reaches the generator config",
      noise_attack["generate_labels_noise"]["noise_config"]["epsilon_255"] == 8,
      str(noise_attack["generate_labels_noise"]["noise_config"]))

vanilla_user = toml.load(
    REPO / "experiments" / f"sweep_{TAG}_user_vanilla_m1500" / "config.toml")
noise_user = toml.load(
    REPO / "experiments" / f"sweep_{TAG}_user_noise_eps8_m1500" / "config.toml")
check("baseline user training uses the untouched train_user module",
      "train_user" in vanilla_user and "train_user_noise" not in vanilla_user)
check("baseline user config has no generator",
      "generator" not in vanilla_user["train_user"]
      and "generator" not in vanilla_user["run_defenses"])
check("FLIP+G user training uses train_user_noise",
      "train_user_noise" in noise_user)
check("FLIP+G passes the generator to training and to the defenses",
      noise_user["train_user_noise"]["generator"].endswith("generator.pth")
      and noise_user["run_defenses"]["generator"].endswith("generator.pth"))
check("alpha is set so train_user's `alpha > 0` test cannot hit None",
      vanilla_user["train_user"]["alpha"] == 0.0)

print("\n== path wiring ==")
attack_out = noise_attack["generate_labels_noise"]["output_dir"]
check("select_flips reads the labels the attack module writes",
      noise_attack["select_flips"]["input_label_glob"] == attack_out + "labels.npy"
      and noise_attack["select_flips"]["true_labels"] == attack_out + "true.npy")
check("user training reads the budget file select_flips writes",
      noise_user["train_user_noise"]["input_labels"] == attack_out + "1500.npy",
      noise_user["train_user_noise"]["input_labels"])
check("budget file is in select_flips' budget list",
      1500 in noise_attack["select_flips"]["budgets"])
check("defenses read the model the user training writes",
      noise_user["run_defenses"]["model_pth"]
      == noise_user["train_user_noise"]["output_dir"] + "model.pth")
check("expert path format strings carry three placeholders",
      noise_attack["generate_labels_noise"]["input_pths"].count("{}") == 3
      and noise_attack["generate_labels_noise"]["opt_pths"].count("{}") == 3)
check("every setting reads the same expert checkpoints",
      vanilla_attack["generate_labels"]["input_pths"]
      == noise_attack["generate_labels_noise"]["input_pths"])
check("expert output dir matches the checkpoint glob the attack reads",
      toml.load(REPO / "experiments" / f"sweep_{TAG}_expert_0"
                / "config.toml")["train_expert"]["output_dir"]
      == noise_attack["generate_labels_noise"]["input_pths"].split("model_")[0]
      .replace("{}", "0"))
check("expert_config announces the right expert count",
      noise_attack["generate_labels_noise"]["expert_config"]["experts"] == 2)

# extract_experts samples a trajectory epoch in [1, max]; a max past the
# number of epochs actually checkpointed is a FileNotFoundError at the first
# batch of the attack stage.
def expert_config_for(expert_epochs, noise):
    """Generates one attack config and returns its expert_config block."""
    a = run_sweep.parse_args(["--tag", "x", "--expert-epochs", str(expert_epochs)])
    r = run_sweep.Runner(a)
    r.run_experiment = lambda *_, **__: True
    name = "n" if noise else "v"
    r.generate_labels({"name": name, "noise": noise, "eps_255": 8 if noise else None})
    module = "generate_labels_noise" if noise else "generate_labels"
    cfg = toml.load(REPO / "experiments" / f"sweep_x_attack_{name}" / "config.toml")
    shutil.rmtree(REPO / "experiments" / f"sweep_x_attack_{name}", ignore_errors=True)
    shutil.rmtree(REPO / "out" / "sweep" / "x", ignore_errors=True)
    return cfg[module]["expert_config"]


for ep in [1, 2, 5, 20, 50]:
    ec = expert_config_for(ep, noise=True)
    # randint(min, max) + 1 samples [min+1, max], so max is the highest epoch
    # that can be requested and must be a checkpointed one.
    check(f"expert_epochs={ep}: sampled trajectory epochs stay within training",
          ec["max"] <= ep, f"max={ec['max']}, epochs={ep}")

for ep in [1, 2, 5, 20, 50]:
    van, noi = expert_config_for(ep, noise=False), expert_config_for(ep, noise=True)
    check(f"expert_epochs={ep}: vanilla and noise get identical bounds",
          van == noi, f"vanilla={van}, noise={noi}")

# The full run must be untouched: expert_epochs=20 has to keep the repo's own
# DEFAULT_EXPERT_CONFIG bound of 15, not widen to 20.
default_ec = expert_config_for(run_sweep.parse_args([]).expert_epochs, noise=True)
check("full run keeps the repo's default bound of max=15",
      default_ec["min"] == 0 and default_ec["max"] == 15, str(default_ec))

print("\n== relative paths resolve under the repo root ==")
bad_paths = []
for exp_name, _ in generated:
    cfg = toml.load(REPO / "experiments" / exp_name / "config.toml")
    for mod, block in cfg.items():
        for key in ["output_dir", "input_labels", "true_labels", "model_pth",
                    "generator"]:
            if key in block:
                if str(block[key]).startswith("/") or ".." in str(block[key]):
                    bad_paths.append(f"{exp_name}/{mod}/{key}={block[key]}")
check("no absolute or parent-relative paths in configs", not bad_paths,
      str(bad_paths[:3]))

print("\n== result collection is not limited to --epsilons / --budgets ==")
# write_results truncates results.csv, so collect() must find cells this
# invocation was not asked to run, or re-running for one new epsilon drops
# finished rows from the table.
DISC = "disctest"
disc_dirs = []
for leaf in ["user_vanilla_m1500", "user_noise_eps8_m1500",
             "user_noise_eps2_m300"]:
    d = REPO / "experiments" / f"sweep_{DISC}_{leaf}"
    d.mkdir(parents=True, exist_ok=True)
    disc_dirs.append(d)

disc_args = run_sweep.parse_args(
    ["--tag", DISC, "--epsilons", "4", "--budgets", "1500", "--dry-run"])
disc_cells = run_sweep.Runner(disc_args).result_cells()
disc_names = [(s["name"], b) for s, b in disc_cells]

check("cells on disk are collected even when not in --epsilons",
      ("noise_eps8", 1500) in disc_names, str(disc_names))
check("cells on disk are collected even when not in --budgets",
      ("noise_eps2", 300) in disc_names, str(disc_names))
check("the requested cell is collected too",
      ("noise_eps4", 1500) in disc_names, str(disc_names))
check("baseline first, then ascending epsilon, then ascending m",
      disc_names == [("vanilla", 1500), ("noise_eps2", 300),
                     ("noise_eps4", 1500), ("noise_eps8", 1500)],
      str(disc_names))
check("no duplicate cells when disk and request overlap",
      len(disc_names) == len(set(disc_names)), str(disc_names))
check("noise flag and epsilon survive the round trip through the dir name",
      all(s["noise"] == (s["name"] != "vanilla") for s, _ in disc_cells)
      and dict((s["name"], s["eps_255"]) for s, _ in disc_cells)
      == {"vanilla": None, "noise_eps2": 2, "noise_eps4": 4, "noise_eps8": 8})

for d in disc_dirs:
    shutil.rmtree(d, ignore_errors=True)
shutil.rmtree(REPO / "out" / "sweep" / DISC, ignore_errors=True)

print("\n== smoke preset ==")
smoke = run_sweep.parse_args(["--tag", "s", "--smoke"])
check("smoke shrinks every expensive knob",
      smoke.experts == 1 and smoke.expert_epochs == 2
      and smoke.attack_iterations == 1 and smoke.user_epochs == 2
      and smoke.budgets == [150] and smoke.epsilons == [8])

print("\n== defaults ==")
d = run_sweep.parse_args([])
check("epsilon grid is {1,2,4,8}/255", d.epsilons == [1, 2, 4, 8])
check("budgets match the paper's m values",
      d.budgets == [150, 300, 500, 1000, 1500])
check("Table 1 CIFAR-10 setting", run_sweep.BASE_SETTING == {
    "model": "r32p", "dataset": "cifar", "trainer": "sgd",
    "poisoner": "1xs", "source_label": 9, "target_label": 4})
check("expert and attack defaults match the repo's own defaults",
      d.experts == 50 and d.attack_iterations == 15)

# Clean up the throwaway configs.
for exp_name, _ in generated:
    shutil.rmtree(REPO / "experiments" / exp_name, ignore_errors=True)
shutil.rmtree(REPO / "out" / "sweep" / TAG, ignore_errors=True)
print(f"\ncleaned up {len(generated)} test config dirs")

print("\n" + "=" * 60)
print(f"{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED: " + ", ".join(FAIL))
sys.exit(1 if FAIL else 0)
