# FLIP with a GAN trigger vs the paper's trigger (`ft_s0_D`)

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
| 1xs | -- | 1500 | 83.80 | 99.70 | -- | 98.44 | 0.992 | 87.92 | 0.917 | 92.69 | 0.969 |
| vit_eps6 | 6 | 1500 | 83.82 | 74.10 | 68.30 | 93.80 | 0.975 | 83.42 | 0.901 | 87.78 | 0.943 |
