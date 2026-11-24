"""
Backdoor defenses used to evaluate detectability of the poisoned examples.

These are reimplementations, not the authors' code: the published FLIP release
does not ship the defenses from Appendix D.2 (they live in the upstream
backdoor-suite). Each defense here takes a matrix of penultimate-layer
representations for the examples carrying the target label and returns a
suspicion score per example, with higher meaning more suspicious. The caller
turns scores into detection rates.

Implemented:
  - kmeans: 2-means on the representations; the smaller cluster is treated as
    the poison cluster (activation clustering, Chen et al. 2018).
  - pca: spectral signatures (Tran et al. 2018); score is the magnitude of the
    projection onto the top singular vector of the centered representations.
  - spectre: robust-covariance whitening followed by a QUE score
    (Hayase et al. 2021).
"""

import numpy as np


def _kmeans_pp_init(x, k, rng):
    '''k-means++ seeding.'''
    centers = [x[rng.randint(len(x))]]
    for _ in range(1, k):
        d2 = np.min(
            ((x[:, None, :] - np.array(centers)[None, :, :]) ** 2).sum(axis=2),
            axis=1
        )
        total = d2.sum()
        if total <= 0:
            centers.append(x[rng.randint(len(x))])
            continue
        centers.append(x[rng.choice(len(x), p=d2 / total)])
    return np.array(centers)


def _kmeans(x, k=2, restarts=10, iters=100, seed=0):
    '''Plain Lloyd's algorithm, so that sklearn is not a new dependency.'''
    rng = np.random.RandomState(seed)
    best = (None, None, np.inf)

    for _ in range(restarts):
        centers = _kmeans_pp_init(x, k, rng)
        assign = np.zeros(len(x), dtype=int)

        for _ in range(iters):
            d = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new_assign = d.argmin(axis=1)
            if np.array_equal(new_assign, assign):
                break
            assign = new_assign
            for j in range(k):
                if (assign == j).any():
                    centers[j] = x[assign == j].mean(axis=0)

        d = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        inertia = d[np.arange(len(x)), assign].sum()
        if inertia < best[2]:
            best = (assign.copy(), centers.copy(), inertia)

    return best[0], best[1]


def kmeans_scores(features, seed=0):
    """
    Activation clustering: 2-means, smaller cluster flagged as poison.

    The score is the difference in distance to the two centers, so that it
    ranks examples continuously rather than only producing a hard partition.
    """
    x = features - features.mean(axis=0, keepdims=True)
    assign, centers = _kmeans(x, k=2, seed=seed)

    counts = np.array([(assign == 0).sum(), (assign == 1).sum()])
    poison_cluster = int(counts.argmin())
    clean_cluster = 1 - poison_cluster

    d_poison = ((x - centers[poison_cluster]) ** 2).sum(axis=1)
    d_clean = ((x - centers[clean_cluster]) ** 2).sum(axis=1)
    return d_clean - d_poison


def pca_scores(features):
    '''Spectral signatures: projection magnitude onto the top singular vector.'''
    x = features - features.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return np.abs(x @ vt[0])


def _matrix_exp_psd(m):
    '''Matrix exponential of a symmetric matrix via its eigendecomposition.'''
    vals, vecs = np.linalg.eigh(m)
    vals = np.clip(vals, -60, 60)   # guards against overflow in exp
    return (vecs * np.exp(vals)) @ vecs.T


def _que_scores(x, alpha):
    '''QUE scores for whitened, centered representations.'''
    n, d = x.shape
    sigma = (x.T @ x) / max(n, 1)
    op_norm = np.linalg.norm(sigma, ord=2)
    if op_norm <= 0:
        return np.zeros(n)

    q = _matrix_exp_psd(alpha * (sigma / op_norm - np.eye(d)))
    trace = np.trace(q)
    if trace <= 0:
        return np.zeros(n)
    return np.einsum('ij,jk,ik->i', x, q, x) / trace


def spectre_scores(features, n_poison_estimate=None, dim=64, alpha=4.0,
                   trim_rounds=3, expansion=1.5):
    """
    SPECTRE: robust whitening followed by a QUE score.

    The robust covariance estimate is approximated by iterative trimming: fit
    mean and covariance on a retained subset, whiten *every* point with it,
    score, drop the most suspicious points, refit. Whitening with a cleaner
    subset than the one being scored is the whole point -- a covariance fitted
    on the poisons too would whiten away the very direction they inflate, and
    the QUE score would see nothing.

    The trimming is seeded with a spectral-signature pass rather than with the
    full population, for the same reason: the first covariance estimate has to
    already exclude some of the poisons to be of any use.

    The published algorithm uses an explicit filtering routine; this is a
    practical stand-in and is labelled as such in the results.
    """
    n = len(features)
    dim = int(min(dim, features.shape[1], max(n - 1, 1)))

    # Robust centering, then dimensionality reduction.
    x = features - np.median(features, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    x = x @ vt[:dim].T

    if n_poison_estimate is None:
        n_poison_estimate = max(int(0.1 * n), 1)
    n_drop = int(min(expansion * n_poison_estimate, max(n - dim - 1, 0)))

    # Seed the retained set by dropping the points a spectral signature finds
    # most suspicious, so that the first covariance fit is already partly clean.
    keep = np.ones(n, dtype=bool)
    if n_drop > 0:
        seed_scores = np.abs(x[:, 0]) if dim > 0 else np.zeros(n)
        keep[np.argsort(seed_scores)[-n_drop:]] = False

    scores = np.zeros(n)
    for _ in range(max(trim_rounds, 1)):
        sub = x[keep]
        mu = sub.mean(axis=0, keepdims=True)
        centered = sub - mu

        # Whiten with the covariance of the retained points.
        cov = (centered.T @ centered) / max(len(centered), 1)
        vals, vecs = np.linalg.eigh(cov)
        inv_sqrt = vecs @ np.diag(1.0 / np.sqrt(np.clip(vals, 1e-8, None))) @ vecs.T

        whitened_all = (x - mu) @ inv_sqrt
        scores = _que_scores(whitened_all, alpha)

        if n_drop <= 0:
            break
        new_keep = np.ones(n, dtype=bool)
        new_keep[np.argsort(scores)[-n_drop:]] = False
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep

    return scores


def detection_metrics(scores, poison_mask, expansion=1.5):
    """
    Turns suspicion scores into the numbers reported in the results table.

    `expansion` is the usual over-removal budget: the defender removes
    1.5x as many examples as they believe were poisoned. Reported alongside is
    an AUROC, which is budget-free and so easier to compare across settings.
    """
    scores = np.asarray(scores, dtype=np.float64)
    poison_mask = np.asarray(poison_mask, dtype=bool)

    n = len(scores)
    n_poison = int(poison_mask.sum())
    n_clean = n - n_poison

    out = {
        'n_examined': n,
        'n_poison': n_poison,
    }

    if n_poison == 0 or n_clean == 0:
        out.update({'budget': 0, 'true_positives': 0, 'detection_rate': None,
                    'false_positive_rate': None, 'auroc': None})
        return out

    k = int(min(n, np.ceil(expansion * n_poison)))
    flagged = np.zeros(n, dtype=bool)
    flagged[np.argsort(scores)[-k:]] = True

    tp = int((flagged & poison_mask).sum())
    fp = int((flagged & ~poison_mask).sum())

    # AUROC via the Mann-Whitney U statistic, with midranks for ties.
    order = np.argsort(scores)
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1)
    _, inverse, counts = np.unique(scores, return_inverse=True,
                                   return_counts=True)
    if (counts > 1).any():
        sums = np.zeros(len(counts))
        np.add.at(sums, inverse, ranks)
        ranks = (sums / counts)[inverse]

    rank_sum = ranks[poison_mask].sum()
    auroc = (rank_sum - n_poison * (n_poison + 1) / 2) / (n_poison * n_clean)

    out.update({
        'budget': k,
        'true_positives': tp,
        'detection_rate': tp / n_poison,
        'false_positive_rate': fp / n_clean,
        'auroc': float(auroc),
    })
    return out


DEFENSES = {
    'kmeans': lambda f, n_poison: kmeans_scores(f),
    'pca': lambda f, n_poison: pca_scores(f),
    'spectre': lambda f, n_poison: spectre_scores(f, n_poison_estimate=n_poison),
}
