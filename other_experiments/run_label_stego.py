#!/usr/bin/env python
"""Train the six label-hiding steganographic encoders, then compare them (no FLIP yet).

All: TriggerEncoder + TriggerDecoder (10-class classifier), whole CIFAR-10, random hidden labels,
|delta| <= 6/255 through the encoder's tanh, no MSE term, 20 epochs, batch 64, Adam 1e-3.

    s1_nodisc_noadv              decoder only
    s2_nodisc_adv_resnet18       decoder + adversarial loss (resnet18 surrogate, seed 0)
    s2_nodisc_adv_resnet32       decoder + adversarial loss (resnet32 surrogate, seed 0)
    s2_nodisc_adv_efficientnet   decoder + adversarial loss (efficientnet surrogate, seed 0)
    s3_disc_noadv                decoder + discriminator
    s4_disc_adv_resnet18         decoder + discriminator + adversarial loss (resnet18 surrogate)

Checkpoints: noise_generator/label_stego/<name>.{pth,json}; comparison: experiments/label_stego/.
Finished encoders are skipped.

    python other_experiments/run_label_stego.py                      # train all, then compare
    python other_experiments/run_label_stego.py --steps compare
    python other_experiments/run_label_stego.py --only s1_nodisc_noadv s3_disc_noadv
"""

import argparse
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent   # repo root (this file lives in other_experiments/)
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

OUT_DIR = "noise_generator/label_stego"
SURR = "noise_generator/surrogates/{}_seed0.pth"

SETTINGS = {
    "s1_nodisc_noadv":            dict(surrogate_arch=None, use_disc=False),
    "s2_nodisc_adv_resnet18":     dict(surrogate_arch="resnet18", use_disc=False),
    "s2_nodisc_adv_resnet32":     dict(surrogate_arch="resnet32", use_disc=False),
    "s2_nodisc_adv_efficientnet": dict(surrogate_arch="efficientnet", use_disc=False),
    "s3_disc_noadv":              dict(surrogate_arch=None, use_disc=True),
    "s4_disc_adv_resnet18":       dict(surrogate_arch="resnet18", use_disc=True),
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", nargs="+", choices=["train", "compare"], default=["train", "compare"])
    p.add_argument("--only", nargs="+", choices=list(SETTINGS), default=list(SETTINGS))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    device = torch.device("cuda")

    if "train" in a.steps:
        from modules.Steganography.label_stego import train_label_stego
        for name in a.only:
            s = SETTINGS[name]
            arch = s["surrogate_arch"]
            train_label_stego(name, OUT_DIR, device, surrogate_arch=arch,
                              surrogate_path=SURR.format(arch) if arch else None, use_disc=s["use_disc"],
                              epochs=a.epochs, overwrite=a.overwrite)
    if "compare" in a.steps:
        from compare_label_stego import compare
        compare([os.path.join(OUT_DIR, f"{n}.pth") for n in SETTINGS], "experiments/label_stego", device)


if __name__ == "__main__":
    main()
