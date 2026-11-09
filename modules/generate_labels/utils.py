from modules.base_utils.util import DEFAULT_SGD_KWARGS, softmax
import numpy as np
import torch

DEFAULT_ATTACK_ITERATIONS = 20
DEFAULT_EXPERT_CONFIG = {
    'experts': 50,
    'min': 0,
    'max': 15,
    'trajectories': [50, 100, 150, 200]
}
DEFAULT_SGD_LABELS_KWARGS = {
    'lr': 150,
    'momentum': 0.5
}
DEFAULT_ATTACK_CONFIG = {
    'iterations': 15,
    'one_hot_temp': 5.,
    'delta': 1.,
    'expert_kwargs': DEFAULT_SGD_KWARGS,
    'labels_kwargs': DEFAULT_SGD_LABELS_KWARGS,
    'd_loss': False,
    'd_alpha': 0.5,
    'd_temp': 1.0,
    # SPECTRE Option A: differentiable spectral proxy (off by default)
    'spectre_proxy': False,
    'spectre_lambda': 0.1,
    'spectre_layer': None,  # resolved from model flag when None
}


def extract_experts(
    expert_config,
    expert_path,
    iterations=None,
    expert_opt_path=None
):
    '''Extracts a list of expert checkpoints for the attack'''
    config = {**DEFAULT_EXPERT_CONFIG, **expert_config}
    expert_starts = []
    expert_opt_starts = []

    for _ in range(iterations or DEFAULT_ATTACK_ITERATIONS):
        for s in config['trajectories']:
            expert = np.random.randint(config['experts'])
            trajectory = np.random.randint(config['min'], config['max']) + 1
            expert_starts.append(expert_path.format(expert, trajectory, str(s)))
            if expert_path:
                expert_opt_starts.append(expert_opt_path.format(expert, trajectory, str(s)))
    return expert_starts, expert_opt_starts


def sgd_step(params, grad, opt_state, opt_params):
    '''Performs a standard step of SGD that is differentiable in the labels'''
    weight_decay = opt_params['weight_decay']
    momentum = opt_params['momentum']
    dampening = opt_params['dampening']
    nesterov = opt_params['nesterov']

    d_p = grad
    if weight_decay != 0:
        d_p = d_p.add(params, alpha=weight_decay)
    if momentum != 0:
        if 'momentum_buffer' not in opt_state:
            buf = opt_state['momentum_buffer'] = torch.zeros_like(params.data)
            buf = buf.mul(momentum).add(d_p)
        else:
            buf = opt_state['momentum_buffer']
            buf = buf.mul(momentum).add(d_p, alpha=1 - dampening)
        if nesterov:
            d_p = d_p.add(buf, alpha=momentum)
        else:
            d_p = buf

    return params.add(d_p, alpha=-opt_params['lr'])


def extract_labels(dataset, label_temp, n_classes=10):
    '''Extracts the labels from a dataset'''
    labels = []
    for _, y in dataset:
        base = np.zeros(n_classes)
        base[y] = label_temp
        labels.append(torch.FloatTensor(base))
    return labels


def coalesce_attack_config(attack_config):
    '''Coalesces the attack config with the default config'''
    expert_kwargs = attack_config.get('expert_kwargs', {})
    labels_kwargs = attack_config.get('labels_kwargs', {})
    attack_config['expert_kwargs'] = {**DEFAULT_SGD_KWARGS, **expert_kwargs}
    attack_config['labels_kwargs'] = {**DEFAULT_SGD_LABELS_KWARGS,
                                      **labels_kwargs}
    return {**DEFAULT_ATTACK_CONFIG, **attack_config}


def spectre_proxy_loss(feats, y_true, y_soft_logits, target_label, eps=1e-8):
    """
    Differentiable SPECTRE-inspired proxy: penalize soft poison mass energy on the
    (stop-grad) top principal direction of centered features, relative to clean-target mass.

    On PyTorch 1.11, `feats` are typically detached (no functional_call); gradients reach
    labels through the soft weights derived from `y_soft_logits` only.

    feats: (B, D) features
    y_true: (B,) hard class indices, or (B, C) one-hots
    y_soft_logits: (B, C) learnable soft labels / logits
    """
    if feats.ndim > 2:
        feats = feats.reshape(feats.shape[0], -1)

    if y_true.ndim > 1:
        y_true = y_true.argmax(dim=1)

    p = softmax(y_soft_logits)
    t = int(target_label)
    w_p = p[:, t] * (y_true != t).float()
    w_c = p[:, t] * (y_true == t).float()

    if w_p.sum() < eps or w_c.sum() < eps:
        return feats.new_zeros(())

    # Center with stop-grad clean-target mean
    mu = (w_c.detach().unsqueeze(1) * feats.detach()).sum(0) / (w_c.detach().sum() + eps)
    centered = feats - mu

    # Top right-singular vector of stop-grad centered features
    with torch.no_grad():
        # (D, B) economy SVD; take first right vector in feature space via Vh[0]
        try:
            _, _, vh = torch.linalg.svd(centered.detach(), full_matrices=False)
            v = vh[0]
        except RuntimeError:
            # Fallback: top eigenvector of Gram in feature dim if B is tiny
            cov = centered.detach().T @ centered.detach()
            _, evecs = torch.linalg.eigh(cov)
            v = evecs[:, -1]

    proj = centered @ v
    poison_energy = (w_p * proj.square()).sum() / (w_p.sum() + eps)
    clean_energy = (w_c * proj.square()).sum() / (w_c.sum() + eps)
    return poison_energy / (clean_energy + eps)
