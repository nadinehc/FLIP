# Other experiments

Earlier explorations, kept for reference. The final experiment (noise cGANs + FLIP) lives at the repo
root: `Common_protocol_cGAN.ipynb`, `run_noise_cgan.py`, `plot_cgan_noise.py`, `launch_cgan_flip.sh`.

Everything here still runs from the repo root: the scripts set `ROOT` to the parent folder and `chdir`
into it, and each notebook starts with a cell that does the same. Run scripts as
`python other_experiments/<script>.py` from the repo root.

| File | What it is |
|---|---|
| `exploration.ipynb`, `exploration/` | first explorations of FLIP (trigger / label analysis) |
| `progress/` | progress notes and figures (Oct 2026) |
| `gan.ipynb`, `noise_generating_gan.ipynb`, `vit_experimentation.ipynb` | first GAN-generated noise attempts (U-Net generator, pix2pix-style) |
| `encoder_decoder.ipynb` | steganographic trigger (encoder / decoder), surrogates |
| `run_steg_v2_matrix.py`, `run_steg_v2_matrix_k20.py`, `resume_steg_v2_users.py`, `steg_v2_k1_vs_k20.ipynb`, `analyze_flip.py` | deer-in-truck steganography (v2) FLIP matrix and analysis |
| `run_label_stego.py`, `compare_label_stego.py`, `label_stego_analysis.ipynb` | label-hiding steganographic encoders (v3) |
