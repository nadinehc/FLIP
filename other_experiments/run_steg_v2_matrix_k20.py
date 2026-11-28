#!/usr/bin/env python
"""Run the steganography-v2 FLIP experiment matrix with K=20 target deer.

Same as run_steg_v2_matrix.py (K=1) except:
  * K = 20: the encoder is trained with a pool of 20 target deer (top-20 by worst-case surrogate
    margin) and, at poisoning time, each truck is blended with a deer drawn at random from that
    same pool (the poisoner reads the pool from the encoder checkpoint),
  * USER_EPOCHS = 100 (was 75). Note: run_full_flip's default lr milestones are [75, 150], so the
    users now get one lr drop (/10) at epoch 75, which the 75-epoch K=1 users never had,
  * experiment names carry a _k20 suffix, so nothing collides with the K=1 outputs.

For ensemble size in {1, 5} and each surrogate architecture (resnet18, resnet32, efficientnet):
  1. train a deer-in-truck encoder against that surrogate ensemble,
  2. run the full FLIP pipeline with that encoder as the steganography_v2 poisoner.

Everything is resumable: trained encoders and finished FLIP stages are skipped.

    python other_experiments/run_steg_v2_matrix_k20.py --stage encoders     # only train the encoders (inspect first)
    python other_experiments/run_steg_v2_matrix_k20.py --stage flip         # FLIP for the whole matrix
    python other_experiments/run_steg_v2_matrix_k20.py --stage flip --archs resnet18 --ensembles 5
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent   # repo root (this file lives in other_experiments/)
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from modules.Steganography.encoder_training import run_experiment

ARCHS = ["resnet18", "resnet32", "efficientnet"]
ENSEMBLES = [1, 5]

# trigger config (same as the K=1 matrix, except K)
K = 20
EPS_PIXEL = 6 / 255
SIGMA = 0.3
LAM_ADV = 10.0
ENC_EPOCHS = 40

# FLIP config (same as the K=1 matrix, except USER_EPOCHS)
MODEL = "r32p"                       # user + expert model = ResNet-32
N_EXPERTS = 5
N_USER_RUNS = 5
USER_EPOCHS = 100
EXPERT_EPOCHS = 20
BUDGETS = [300, 500, 1000, 1500]
SOURCE, TARGET = 9, 4                # truck -> deer

ENC_DIR = "noise_generator/encoders"
SURR_DIR = "noise_generator/surrogates"


def exp_name(arch, ens):
    return f"steg_v2_{arch}_ens{ens}_k{K}"


def encoder_ckpt(arch, ens):
    return os.path.join(ENC_DIR, f"{arch}_ens{ens}_k{K}.pth")


def surrogate_paths(arch, ens):
    return [os.path.join(SURR_DIR, f"{arch}_seed{s}.pth") for s in range(ens)]


def train_encoders(archs, ensembles, device):
    summaries = []
    for arch in archs:
        for ens in ensembles:
            s = run_experiment(arch, ens, ENC_DIR, device, k=K, eps_pixel=EPS_PIXEL, sigma=SIGMA,
                               lam_adv=LAM_ADV, epochs=ENC_EPOCHS, source_label=SOURCE,
                               target_label=TARGET, surrogate_dir=SURR_DIR)
            summaries.append(s)
    print("\n=== encoder summary ===")
    print(f"{'name':28s} {'deer_rate':>9s} {'trainD':>7s} {'freshD':>7s} {'PSNR':>6s}")
    for s in summaries:
        print(f"{s['name']:28s} {s['mean_surrogate_deer_rate']:9.3f} "
              f"{s['train_discriminator_acc']:7.3f} {s['fresh_discriminator_acc']:7.3f} {s['psnr']:6.1f}")
    return summaries


def run_flip(arch, ens, extra_args):
    name = exp_name(arch, ens)
    ckpt = encoder_ckpt(arch, ens)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"encoder {ckpt} not trained yet")

    env = dict(os.environ)
    env["STEG_V2_ENCODER"] = ckpt
    env["STEG_V2_SURROGATES"] = os.pathsep.join(surrogate_paths(arch, ens))

    cmd = [sys.executable, "run_full_flip.py", "--name", name, "--poisoner", "steganography_v2",
           "--model", MODEL, "--n-experts", str(N_EXPERTS), "--n-user-runs", str(N_USER_RUNS),
           "--user-epochs", str(USER_EPOCHS), "--expert-epochs", str(EXPERT_EPOCHS),
           "--budgets", *map(str, BUDGETS), "--source", str(SOURCE), "--target", str(TARGET),
           *extra_args]
    print(f"\n=== FLIP: {name} ===\n  encoder: {ckpt}\n  surrogates: {env['STEG_V2_SURROGATES']}")
    print("  " + " ".join(cmd), flush=True)
    subprocess.run(cmd, env=env, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--archs", nargs="+", default=ARCHS, choices=ARCHS)
    p.add_argument("--ensembles", nargs="+", type=int, default=ENSEMBLES)
    p.add_argument("--stage", choices=["encoders", "flip", "all"], default="all")
    p.add_argument("--flip-stages", nargs="+", default=None,
                   help="restrict run_full_flip stages (experts labels flips users summary)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.stage in ("encoders", "all"):
        train_encoders(args.archs, args.ensembles, device)

    if args.stage in ("flip", "all"):
        extra = ["--stages", *args.flip_stages] if args.flip_stages else []
        for arch in args.archs:
            for ens in args.ensembles:
                run_flip(arch, ens, extra)

    print("\nmatrix done.")


if __name__ == "__main__":
    main()
