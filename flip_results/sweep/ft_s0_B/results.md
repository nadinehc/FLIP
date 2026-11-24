# FLIP with a GAN trigger vs the paper's trigger (`ft_s0_B`)

- Setting: cifar, r32p, source 9 -> target 4
- The attack is label-only in every row: the victim gets clean images and flipped labels. Only the trigger differs.
- Experts: 2 x 20 epochs, retrained per trigger; attack iterations: 15
- GAN trigger: linf bound, 20 epochs, lambda_bd=1.0, lambda_adv=0.0
- `surrogate ASR` is the trigger's attack success against the surrogate used to train it, i.e. an upper reference for what FLIP could distil.
- Defenses given a 1.5x removal budget over the true poison count
- CTA / PTA / detection / FPR in %, AUROC in [0, 1]
- Defense implementations are reimplementations; see `modules/run_defenses/utils.py`

| trigger | eps (/255) | m | CTA | PTA | surrogate ASR | kmeans det | kmeans AUROC | PCA det | PCA AUROC | SPECTRE det | SPECTRE AUROC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1xs | -- | 1500 | 84.61 | 98.70 | -- | 98.41 | 0.992 | 85.76 | 0.910 | 92.80 | 0.970 |
| vit_eps6 | 6 | 1500 | 82.47 | 100.00 | 64.90 | 0.38 | 0.052 | 66.62 | 0.840 | 77.24 | 0.899 |
