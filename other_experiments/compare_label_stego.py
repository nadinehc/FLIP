#!/usr/bin/env python
"""Compare label-hiding steganographic encoders (Steganography.label_stego) on the CIFAR-10 test set.

Per encoder:
  decoding     decoder accuracy on random hidden labels (whole test set) and on trucks hiding "deer"
  label dep.   mean correlation between the noises produced for the 10 possible hidden labels on the
               same image (near 1 = the noise ignores the label, low = label-specific)
  attack       truck -> deer rate (trucks hiding "deer" classified as deer) on the 15 clean models
               (resnet18 / resnet32 / efficientnet surrogates, seeds 0-4). The model the encoder was
               trained against (if any) is reported separately from the held-out ones. Also the
               targeted success over random hidden labels (all classes) on held-out resnet32 models,
               and the clean-truck -> deer baseline.
  stealth      PSNR, Linf, fraction of |delta| >= 0.9 eps, accuracy of a fresh PoisonDiscriminator
               trained to separate clean from poisoned images, AUC of a training-free Laplacian
               high-pass energy detector (trucks hiding deer vs clean trucks)

Writes <out>/comparison.csv, comparison.json, examples.png (trucks hiding deer) and labels.png
(one image hiding each of the 10 labels).

    python other_experiments/compare_label_stego.py noise_generator/label_stego/*.pth --out experiments/label_stego
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent   # repo root (this file lives in other_experiments/)
sys.path.insert(0, str(ROOT))

from modules.Steganography.decoder import TriggerDecoder
from modules.Steganography.deer_blend import PoisonDiscriminator, discriminator_loss
from modules.Steganography.encoder_training import quantize_st
from modules.Steganography.label_stego import build_encoder, load_cifar_tensor, load_surrogate

ARCHS = ["resnet18", "resnet32", "efficientnet"]
SOURCE, TARGET = 9, 4


def load(path, device):
    ck = torch.load(path, map_location=device)
    enc = build_encoder(ck["noise_scale"]); enc.load_state_dict(ck["state_dict"]); enc.to(device).eval()
    dec = TriggerDecoder(3, 10); dec.load_state_dict(ck["decoder_state_dict"]); dec.to(device).eval()
    return enc, dec, ck["config"], ck["noise_scale"]


@torch.no_grad()
def poison(enc, x, y, bs=500):
    return torch.cat([quantize_st(enc(x[i:i + bs], y[i:i + bs])) for i in range(0, len(x), bs)])


@torch.no_grad()
def predict(m, x, bs=500):
    return torch.cat([m(x[i:i + bs]).argmax(1) for i in range(0, len(x), bs)])


def corr_offdiag(d):            # d: [L, ...] -> mean pairwise correlation of the L flattened rows
    f = d.flatten(1); f = (f - f.mean(1, keepdim=True)) / f.std(1, keepdim=True).clamp_min(1e-8)
    c = f @ f.T / f.size(1)
    return c[~torch.eye(len(c), dtype=torch.bool, device=c.device)].mean().item()


def laplacian_auc(clean, trig):
    k = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32, device=clean.device)
    k = k.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    def e(x):
        r = torch.nn.functional.conv2d(torch.nn.functional.pad(x, [1] * 4, mode="reflect"), k, groups=3)
        return r.pow(2).mean((1, 2, 3)).cpu().numpy()
    neg, pos = e(clean), e(trig)
    s = np.r_[neg, pos]; y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    r = np.empty(len(s)); r[np.argsort(s, kind="mergesort")] = np.arange(1, len(s) + 1)
    return float((r[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def fresh_detector(enc, x_train, x_test, device, epochs=3, n_train=20000, seed=0):
    torch.manual_seed(seed)
    det = PoisonDiscriminator().to(device)
    opt = torch.optim.Adam(det.parameters(), lr=2e-4, betas=(0.5, 0.999))
    xs = x_train[torch.randperm(len(x_train), device=device)[:n_train]]
    for _ in range(epochs):
        det.train()
        for i in range(0, len(xs), 128):
            x = xs[i:i + 128]
            with torch.no_grad():
                xp = quantize_st(enc(x, torch.randint(0, 10, (len(x),), device=device)))
            loss, _ = discriminator_loss(det, x, xp)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    det.eval(); acc = 0.0; n = 0
    with torch.no_grad():
        for i in range(0, len(x_test), 500):
            x = x_test[i:i + 500]
            xp = quantize_st(enc(x, torch.randint(0, 10, (len(x),), device=device)))
            acc += discriminator_loss(det, x, xp)[1].item() * len(x); n += len(x)
    return acc / n


def compare(paths, out_dir, device, data_root="data/data_cifar10"):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    x_train, _ = load_cifar_tensor(data_root, True, device)
    x_test, y_test = load_cifar_tensor(data_root, False, device)
    trucks = x_test[y_test == SOURCE]
    y_deer = torch.full((len(trucks),), TARGET, device=device)
    g = torch.Generator().manual_seed(0)
    y_rand = torch.randint(0, 10, (len(x_test),), generator=g).to(device)

    clean = {(a, s): load_surrogate(a, f"noise_generator/surrogates/{a}_seed{s}.pth", device)
             for a in ARCHS for s in range(5)}
    base = {k: (predict(m, trucks) == TARGET).float().mean().item() for k, m in clean.items()}

    rows, examples = [], {}
    for p in paths:
        if not os.path.exists(p):
            print(f"skip {p}: not found", flush=True); continue
        enc, dec, cfg, eps = load(p, device)
        name = cfg["name"]; attacked = (cfg["surrogate_arch"], 0) if cfg["surrogate_arch"] else None
        print(f"\n=== {name}", flush=True)

        xt = poison(enc, trucks, y_deer)
        xr = poison(enc, x_test, y_rand)
        d = xr - x_test
        row = dict(name=name, surrogate=cfg["surrogate_arch"] or "-", disc=cfg["use_disc"],
                   dec_acc=(predict(dec, xr) == y_rand).float().mean().item(),
                   dec_acc_truck_deer=(predict(dec, xt) == TARGET).float().mean().item())

        # label dependence: 200 test images x 10 hidden labels
        lab = []
        with torch.no_grad():
            for i in range(200):
                x = x_test[i:i + 1].expand(10, -1, -1, -1)
                lab.append(corr_offdiag(quantize_st(enc(x, torch.arange(10, device=device))) - x))
        row["noise_corr_across_labels"] = float(np.mean(lab))
        row["noise_corr_across_images"] = corr_offdiag((xt - trucks)[:200])

        # attack: truck -> deer on the 15 clean models
        rates = {k: (predict(m, xt) == TARGET).float().mean().item() for k, m in clean.items()}
        row["deer_rate_attacked_model"] = rates[attacked] if attacked else float("nan")
        for a in ARCHS:
            held = [rates[(a, s)] for s in range(5) if (a, s) != attacked]
            row[f"deer_rate_heldout_{a}"] = float(np.mean(held))
        row["deer_rate_heldout_all"] = float(np.mean([v for k, v in rates.items() if k != attacked]))
        row["clean_truck_deer_rate"] = float(np.mean(list(base.values())))
        held32 = [s for s in range(5) if ("resnet32", s) != attacked]
        row["targeted_any_class_heldout_resnet32"] = float(np.mean(
            [(predict(clean[("resnet32", s)], xr) == y_rand).float().mean().item() for s in held32]))

        # stealth
        row["psnr"] = float(10 * np.log10(1 / ((d / 2) ** 2).mean((1, 2, 3)).mean().item()))
        row["linf_255"] = d.abs().max().item() / 2 * 255
        row["sat_frac"] = (d.abs() >= 0.9 * eps).float().mean().item()
        row["laplacian_auc_trucks"] = laplacian_auc(trucks, xt)
        row["fresh_detector_acc"] = fresh_detector(enc, x_train, x_test, device)
        rows.append(row)
        for k, v in row.items():
            print(f"  {k:38s} {v:.3f}" if isinstance(v, float) else f"  {k:38s} {v}", flush=True)

        with torch.no_grad():
            x1 = x_test[3:4].expand(10, -1, -1, -1)
            examples[name] = dict(trucks=trucks[:6].cpu(), trig=xt[:6].cpu(), eps=eps,
                                  lab=(quantize_st(enc(x1, torch.arange(10, device=device))) - x1).cpu(),
                                  img=x_test[3].cpu())

    with open(out / "comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    json.dump(dict(rows=rows, clean_truck_deer_rate_per_model={f"{a}_seed{s}": v for (a, s), v in base.items()}),
              open(out / "comparison.json", "w"), indent=2)
    plot(examples, out)

    cols = ["dec_acc", "dec_acc_truck_deer", "noise_corr_across_labels", "deer_rate_attacked_model",
            "deer_rate_heldout_all", "targeted_any_class_heldout_resnet32", "psnr", "sat_frac",
            "fresh_detector_acc", "laplacian_auc_trucks"]
    short = ["dec", "dec t->d", "corr lab", "deer att", "deer held", "tgt r32", "PSNR", "sat", "fresh D", "lapl AUC"]
    print("\n" + f"{'encoder':28s}" + "".join(f"{s:>10s}" for s in short))
    for r in rows:
        print(f"{r['name']:28s}" + "".join(f"{r[c]:10.3f}" for c in cols))
    print(f"(clean trucks -> deer on the 15 clean models: {rows[0]['clean_truck_deer_rate']:.3f})")


def plot(examples, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    img = lambda t: (t.clamp(-1, 1) * 0.5 + 0.5).permute(1, 2, 0).numpy()
    noise = lambda t, eps: (t / (2 * eps) + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()
    names = list(examples)

    fig, ax = plt.subplots(1 + 2 * len(names), 6, figsize=(12, 2 * (1 + 2 * len(names))))
    for c in range(6):
        ax[0, c].imshow(img(examples[names[0]]["trucks"][c]))
        for r, n in enumerate(names):
            e = examples[n]
            ax[1 + 2 * r, c].imshow(img(e["trig"][c]))
            ax[2 + 2 * r, c].imshow(noise(e["trig"][c] - e["trucks"][c], e["eps"]))
    ax[0, 0].set_ylabel("clean", fontsize=8)
    for r, n in enumerate(names):
        ax[1 + 2 * r, 0].set_ylabel(f"{n}\npoisoned", fontsize=7); ax[2 + 2 * r, 0].set_ylabel("noise", fontsize=7)
    for a in ax.flat:
        a.set_xticks([]); a.set_yticks([])
    plt.suptitle("trucks hiding 'deer' (noise amplified from [-eps, eps])"); plt.tight_layout()
    plt.savefig(out / "examples.png", dpi=90); plt.close()

    classes = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    fig, ax = plt.subplots(len(names), 11, figsize=(18, 1.8 * len(names)), squeeze=False)
    for r, n in enumerate(names):
        e = examples[n]; ax[r, 0].imshow(img(e["img"])); ax[r, 0].set_ylabel(n, fontsize=7)
        for c in range(10):
            ax[r, c + 1].imshow(noise(e["lab"][c], e["eps"]))
            if r == 0:
                ax[r, c + 1].set_title(f"hide {classes[c]}", fontsize=8)
    for a in ax.flat:
        a.set_xticks([]); a.set_yticks([])
    plt.suptitle("noise for the same image hiding each of the 10 labels"); plt.tight_layout()
    plt.savefig(out / "labels.png", dpi=90); plt.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+")
    p.add_argument("--out", default="experiments/label_stego")
    a = p.parse_args()
    os.chdir(ROOT)
    compare(a.paths, a.out, torch.device("cuda"))
