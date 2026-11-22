# SPECTRE side-by-side comparison

Rows are grouped by trigger (`poisoner`) and flip budget. Poison recall = fraction of flipped-to-target examples removed by the filter.

## Summary

| tag | poisoner | budget | eps | pre CTA | pre PTA | SPECTRE recall | PCA recall | kmeans recall | post CTA | post PTA |
|-----|----------|-------:|----:|--------:|--------:|---------------:|-----------:|--------------:|---------:|---------:|
| 1xu_b150_r0 | 1xu | 150 | 128 | 92.4% | 97.7% | 0.0% | 0.0% | 1.6% |  |  |

## Per-filter detail

| tag | poisoner | budget | filter | poison_removed | clean_removed | recall |
|-----|----------|-------:|--------|---------------:|--------------:|-------:|
| 1xu_b150_r0 | 1xu | 150 | rcov | 0 | 192 | 0.0% |
| 1xu_b150_r0 | 1xu | 150 | pca | 0 | 192 | 0.0% |
| 1xu_b150_r0 | 1xu | 150 | kmeans | 2 | 190 | 1.6% |

## Sources

- `1xu_b150_r0`: model `experiments/flip_c10_1xu/user_150/0/model.pth`, metrics `experiments/example_spectre_eval/spectre_metrics.json`
