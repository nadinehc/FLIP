import torch

from modules.base_utils.util import make_pbar

DEFAULT_NTK_CONFIG = {
    'n_train': 4000,
    'n_poison': 500,
    'n_val': 1000,
    'ridge': 1e-3,
    'sketch_dim': 16384,
    'iterations': 500,
    'lr': 0.1,
    'one_hot_temp': 5.,
    'cta_weight': 1.,
    'seed': 0,
}


class CountSketch:
    '''Random projection R^P -> R^d with E[<Sg, Sh>] = <g, h>, without storing a P x d matrix'''
    def __init__(self, n_params, sketch_dim, device, seed=0):
        self.exact = not sketch_dim or sketch_dim >= n_params
        if self.exact:
            return
        gen = torch.Generator().manual_seed(seed)
        self.dim = sketch_dim
        self.buckets = torch.randint(sketch_dim, (n_params,), generator=gen).to(device)
        self.signs = (torch.randint(2, (n_params,), generator=gen) * 2 - 1).float().to(device)

    def __call__(self, g):
        if self.exact:
            return g
        return torch.zeros(self.dim, device=g.device).index_add_(0, self.buckets, g * self.signs)


def ntk_features(model, dataset, n_classes, sketch_dim, seed=0):
    '''
    Computes (sketched) per-example gradients phi(x) of the summed logits, so that
    phi(x) . phi(x') is the empirical pseudo-NTK sum_{c,c'} <df_c(x), df_c'(x')> / C.
    Using a single scalar kernel (instead of the C x C block NTK) avoids the C^2
    scaling mentioned in Appendix A.
    '''
    device = next(model.parameters()).device
    params = [p for p in model.parameters() if p.requires_grad]
    sketch = CountSketch(sum(p.numel() for p in params), sketch_dim, device, seed)

    model.eval()
    feats = []
    with make_pbar(total=len(dataset)) as pbar:
        for i in range(len(dataset)):
            x = dataset[i][0].unsqueeze(0).to(device)
            out = model(x).sum() / n_classes ** 0.5
            grads = torch.autograd.grad(out, params)
            feats.append(sketch(torch.cat([g.reshape(-1) for g in grads])))
            pbar.update(1)
    return torch.stack(feats)


def kernel_regression_operators(phi_train, phi_evals, ridge):
    '''
    Solves the inner problem of (2) in closed form: an infinitely-trained linearized
    network with squared loss predicts f(x) = K(x, X)(K(X, X) + rI)^{-1} y.
    Returns A = K(x_eval, X)(K(X, X) + rI)^{-1} for each eval set, so that f = A y
    is linear (hence trivially differentiable) in the labels y.
    '''
    phi_train = phi_train.double()
    k_train = phi_train @ phi_train.T
    reg = ridge * k_train.diagonal().mean()
    eye = torch.eye(len(k_train), dtype=k_train.dtype, device=k_train.device)

    ops = []
    for phi in phi_evals:
        k_cross = phi_train @ phi.double().T
        ops.append(torch.linalg.solve(k_train + reg * eye, k_cross).T.float())
    return ops
