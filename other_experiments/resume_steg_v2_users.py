#!/usr/bin/env python
"""Resume the steg_v2_{arch}_ens1 user models from 75 to 100 epochs.

The original runs used SGD(lr=0.1, momentum=0.9, nesterov, wd=2e-4) with
MultiStepLR(milestones=[75, 150], gamma=0.1) for 75 epochs, so the LR never decayed.
A 100-epoch run would have trained epochs 76-100 at lr=0.01; that is what we do here,
starting from the saved weights. Only weights were saved, so the SGD momentum buffers
start from zero (rebuilt within the first few hundred steps).

Per run directory experiments/steg_v2_{arch}_ens1/user_{budget}/{run}/:
  * on first touch, model.pth / caccs.npy / paccs.npy are copied to *_ep75.* (never overwritten)
  * training always starts from model_ep75.pth, so re-running after a crash is safe
  * model.pth / caccs.npy / paccs.npy are replaced by the 100-epoch versions (100 rows)
  * a run whose caccs.npy already has 100 rows is skipped

Afterwards the summary stage of run_full_flip.py is re-run for each experiment.

    python other_experiments/resume_steg_v2_users.py                       # all 3 archs
    python other_experiments/resume_steg_v2_users.py --archs resnet18
"""

import argparse
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent   # repo root (this file lives in other_experiments/)
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "modules"))
sys.path.insert(0, str(ROOT))

from modules.base_utils.datasets import get_matching_datasets, pick_poisoner, construct_user_dataset
from modules.base_utils.util import get_train_info, mini_train, load_model, needs_big_ims, softmax

ARCHS = ["resnet18", "resnet32", "efficientnet"]
ENS = 1
K = 1
BUDGETS = [300, 500, 1000, 1500]
N_RUNS = 5
OLD_EPOCHS = 75
EXTRA_EPOCHS = 25
SOURCE, TARGET = 9, 4
MODEL = "r32p"
BATCH_SIZE = 256
# lr already decayed once (0.1 -> 0.01 at epoch 75); next milestone (150) is out of reach
OPTIM = dict(lr=0.01, momentum=0.9, nesterov=True, weight_decay=2e-4)
SCHED = dict(milestones=[150 - OLD_EPOCHS], gamma=0.1)

ENC_DIR = "noise_generator/encoders"
SURR_DIR = "noise_generator/surrogates"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def resume_arch(arch):
    name = f"steg_v2_{arch}_ens{ENS}"
    exp = Path("experiments") / name
    os.environ["STEG_V2_ENCODER"] = os.path.join(ENC_DIR, f"{arch}_ens{ENS}_k{K}.pth")
    os.environ["STEG_V2_SURROGATES"] = os.pathsep.join(
        os.path.join(SURR_DIR, f"{arch}_seed{s}.pth") for s in range(ENS))

    todo = []
    for b in BUDGETS:
        for r in range(N_RUNS):
            d = exp / f"user_{b}" / str(r)
            if np.load(d / "caccs.npy").shape[0] == OLD_EPOCHS + EXTRA_EPOCHS:
                log(f"{name} budget {b} run {r}: already at {OLD_EPOCHS + EXTRA_EPOCHS} epochs, skipping")
            else:
                todo.append((b, r, d))
    if not todo:
        return

    log(f"{name}: building datasets (encoder {os.environ['STEG_V2_ENCODER']})")
    poisoner = pick_poisoner("steganography_v2", "cifar", TARGET)
    _, distillation, test, poison_test, _ = get_matching_datasets(
        "cifar", poisoner, SOURCE, big=needs_big_ims(MODEL))

    for b, r, d in todo:
        # back up the 75-epoch outputs once
        for f in ("model.pth", "caccs.npy", "paccs.npy"):
            bak = d / f.replace(".", "_ep75.")
            if not bak.exists():
                shutil.copy2(d / f, bak)
        old_c, old_p = np.load(d / "caccs_ep75.npy"), np.load(d / "paccs_ep75.npy")
        assert old_c.shape[0] == OLD_EPOCHS, (d, old_c.shape)

        # same label construction as modules/train_user/run_module.py (soft=false, alpha=0)
        labels_d = softmax(torch.tensor(np.load(exp / f"{b}.npy"))).argmax(dim=1)
        user_dataset = construct_user_dataset(distillation, labels_d)

        set_seed(200_000 + b * 100 + r)
        model = load_model(MODEL, 10)
        model.load_state_dict(torch.load(d / "model_ep75.pth", map_location="cuda"))
        _, epochs, opt, sched = get_train_info(model.parameters(), "sgd", BATCH_SIZE,
                                               EXTRA_EPOCHS, OPTIM, SCHED)
        log(f"{name} budget {b} run {r}: training epochs {OLD_EPOCHS + 1}-{OLD_EPOCHS + epochs} "
            f"(start CTA {100 * old_c[-1, 0]:.2f}, PTA {100 * old_p[-1, 0]:.2f})")
        t0 = time.time()
        model, cm, pm = mini_train(model=model, train_data=user_dataset,
                                   test_data=[test, poison_test.poison_dataset],
                                   batch_size=BATCH_SIZE, opt=opt, scheduler=sched,
                                   epochs=epochs, record=True)
        new_c = np.concatenate([old_c, np.array(cm)])
        new_p = np.concatenate([old_p, np.array(pm)])
        torch.save(model.state_dict(), d / "model.pth")
        np.save(d / "paccs.npy", new_p)
        np.save(d / "caccs.npy", new_c)   # written last: its row count is the "done" marker
        log(f"{name} budget {b} run {r}: done in {(time.time() - t0) / 60:.1f} min -> "
            f"CTA {100 * new_c[-1, 0]:.2f}, PTA {100 * new_p[-1, 0]:.2f}")


def refresh_summary(arch):
    name = f"steg_v2_{arch}_ens{ENS}"
    subprocess.run([sys.executable, "run_full_flip.py", "--name", name, "--stages", "summary",
                    "--skip-checks", "--n-user-runs", str(N_RUNS),
                    "--budgets", *map(str, BUDGETS)], check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--archs", nargs="+", default=ARCHS, choices=ARCHS)
    a = p.parse_args()
    for arch in a.archs:
        resume_arch(arch)
        refresh_summary(arch)
    log("all done.")


if __name__ == "__main__":
    main()
