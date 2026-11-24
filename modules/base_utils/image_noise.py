"""
Image-dependent perturbation generator for the FLIP extension.

Vanilla FLIP is label-only: images are untouched and only labels move. This
module adds an optional, image-dependent perturbation G(x) -> delta that the
attacker may apply to the examples whose labels get flipped. It is used by the
`generate_labels_noise` and `train_user_noise` modules and is never imported by
the vanilla label-only path.

Conventions used throughout:
  - `x_pix` is an image in pixel space, scaled to [0, 1].
  - `x_norm` is an image after the dataset's Normalize transform, which is what
    the networks in this repo actually consume.
  - `epsilon` is always expressed in pixel space (so eps=8/255 means what it
    means in the backdoor literature), and is converted into normalized space
    per channel before being added to `x_norm`.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset

from modules.base_utils.datasets import (
    Poisoner,
    CIFAR_TRANSFORM_NORMALIZE_MEAN,
    CIFAR_TRANSFORM_NORMALIZE_STD,
    CIFAR_100_TRANSFORM_NORMALIZE_MEAN,
    CIFAR_100_TRANSFORM_NORMALIZE_STD,
    TINY_IMAGENET_TRANSFORM_NORMALIZE_MEAN,
    TINY_IMAGENET_TRANSFORM_NORMALIZE_STD,
)


NORMALIZATION = {
    'cifar': (CIFAR_TRANSFORM_NORMALIZE_MEAN, CIFAR_TRANSFORM_NORMALIZE_STD),
    'cifar_100': (CIFAR_100_TRANSFORM_NORMALIZE_MEAN,
                  CIFAR_100_TRANSFORM_NORMALIZE_STD),
    'tiny_imagenet': (TINY_IMAGENET_TRANSFORM_NORMALIZE_MEAN,
                      TINY_IMAGENET_TRANSFORM_NORMALIZE_STD),
}

DEFAULT_NOISE_CONFIG = {
    'epsilon': 8 / 255,     # L-infinity bound, in pixel units ([0, 1] scale)
    'norm': 'linf',         # {linf, l2}; l2 bound is in pixel units as well
    'width': 32,            # base channel count of the conv encoder-decoder
    'lr': 1e-3,             # Adam learning rate for G
    'weight_decay': 0.0,
    'warmup': 0,            # label-only iterations before G starts training
    'quantize': True,       # round the deployed perturbation to 8-bit steps
}


def get_normalization(dataset_flag):
    '''Returns the (mean, std) tuples used to normalize a dataset.'''
    if dataset_flag not in NORMALIZATION:
        raise NotImplementedError(
            f"No normalization registered for dataset {dataset_flag}."
        )
    return NORMALIZATION[dataset_flag]


def coalesce_noise_config(noise_config):
    '''Coalesces a noise config with the defaults, resolving epsilon units.'''
    config = {**DEFAULT_NOISE_CONFIG, **(noise_config or {})}

    # `epsilon_255` is a convenience: 8 reads better than 0.03137 in a config.
    if 'epsilon_255' in config:
        config['epsilon'] = float(config.pop('epsilon_255')) / 255

    config['epsilon'] = float(config['epsilon'])
    if config['epsilon'] <= 0:
        raise ValueError("epsilon must be positive.")
    if config['norm'] not in ('linf', 'l2'):
        raise ValueError("norm must be one of {linf, l2}.")

    return config


class _BoundedPerturbation(nn.Module):
    """
    Shared machinery for generators that emit a norm-bounded perturbation.

    Subclasses build their own body and implement `_body(x_pix) -> pre_tanh`.
    Everything that makes the epsilon bound *mean* something lives here, so a
    new architecture cannot accidentally weaken it:

      - the bound is applied by construction (tanh for L-infinity, norm
        squashing for L2), never by projection after the fact, so the generator
        is feasible at every step of training;
      - epsilon is in pixel units and converted into normalized space on the
        way out;
      - the deployed perturbation is rounded to the 8-bit grid and clamped so
        that x + delta stays a valid image.
    """

    def __init__(self, dataset_flag='cifar', epsilon=8 / 255, norm='linf',
                 in_channels=3, quantize=False, init_std=1e-3,
                 adaptive_floor=None):
        super().__init__()
        if adaptive_floor is not None:
            adaptive_floor = float(adaptive_floor)
            if not 0.0 <= adaptive_floor <= 1.0:
                raise ValueError("adaptive_floor must be in [0, 1] or None.")
            if norm != 'linf':
                raise ValueError("adaptive_floor needs norm='linf'.")
        self.adaptive_floor = adaptive_floor
        self.epsilon = float(epsilon)
        self.norm = norm
        self.quantize = quantize
        self.dataset_flag = dataset_flag
        self.in_channels = in_channels
        self.init_std = init_std

        mean, std = get_normalization(dataset_flag)
        self.register_buffer('mean',
                             torch.tensor(mean).view(1, -1, 1, 1).float())
        self.register_buffer('std',
                             torch.tensor(std).view(1, -1, 1, 1).float())

    # -- to be provided by subclasses -------------------------------------

    def _body(self, x_pix):
        raise NotImplementedError

    # -- the bound --------------------------------------------------------

    def to_pixel(self, x_norm):
        '''Maps a normalized image back to pixel space.'''
        return x_norm * self.std + self.mean

    def to_norm(self, x_pix):
        '''Maps a pixel-space image into normalized space.'''
        return (x_pix - self.mean) / self.std

    def local_budget(self, x_pix):
        '''
        Per-pixel share of epsilon the perturbation may use, in
        [adaptive_floor, 1], as a deterministic function of the clean image.

        Textured regions get the full epsilon and flat regions only
        `adaptive_floor` of it. This is the usual visual-masking argument: the
        eye notices a given change far more on a smooth sky than on foliage.
        It also makes the trigger image-dependent by construction, which a
        free-form generator does not manage on its own (it collapses to one
        fixed pattern for every image; see TRIGGER.md).

        The map is the smoothed gradient magnitude of the luminance, scaled so
        that three times the image's own mean gradient counts as fully
        textured. It is a function of the clean image only, so test-time
        stamping reproduces it exactly.
        '''
        lum = (0.299 * x_pix[:, 0:1] + 0.587 * x_pix[:, 1:2]
               + 0.114 * x_pix[:, 2:3])
        pad = torch.nn.functional.pad(lum, (1, 1, 1, 1), mode='replicate')
        gx = pad[..., 1:-1, 2:] - pad[..., 1:-1, :-2]
        gy = pad[..., 2:, 1:-1] - pad[..., :-2, 1:-1]
        grad = torch.sqrt(gx ** 2 + gy ** 2)
        grad = torch.nn.functional.avg_pool2d(grad, 3, stride=1, padding=1)
        ref = grad.flatten(1).mean(1).view(-1, 1, 1, 1)
        # The absolute term keeps a nearly flat image from being rescaled so
        # that its faint noise counts as full texture.
        texture = torch.clamp(grad / (3.0 * ref + 0.01), 0.0, 1.0)
        return self.adaptive_floor + (1.0 - self.adaptive_floor) * texture

    def _raw_delta(self, x_pix):
        '''Runs the body and applies the norm bound.'''
        out = self._body(x_pix)

        if self.norm == 'linf':
            bound = self.epsilon
            if self.adaptive_floor is not None:
                bound = bound * self.local_budget(x_pix)
            return bound * torch.tanh(out)

        # L2: squash the per-image norm into the epsilon ball. The epsilon is
        # interpreted as a bound on the whole-image L2 norm in pixel units.
        flat = torch.tanh(out).flatten(1)
        scale = self.epsilon * torch.tanh(flat.norm(dim=1, keepdim=True))
        unit = flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return (unit * scale).view_as(out)

    def delta_pixel(self, x_pix, quantize=None):
        """
        Returns a feasible perturbation in pixel units for a pixel-space image.

        Feasible means: inside the epsilon ball, and keeping x + delta inside
        the valid [0, 1] image range.
        """
        delta = self._raw_delta(x_pix)

        quantize = self.quantize if quantize is None else quantize
        if quantize:
            # Deployed perturbations have to survive being written to an 8-bit
            # image. Rounding here (outside the optimization loop) keeps the
            # reported epsilon honest.
            delta = torch.round(delta * 255) / 255

        # Clamp so that the perturbed image stays a valid image. Done in pixel
        # space, where the constraint is exact.
        return torch.clamp(delta, -x_pix, 1 - x_pix)

    def forward(self, x_norm, quantize=None):
        '''Returns the perturbation in normalized space for a normalized image.'''
        x_pix = self.to_pixel(x_norm)
        delta_pix = self.delta_pixel(x_pix, quantize=quantize)
        return delta_pix / self.std

    def perturb_pixel(self, x_pix, quantize=None):
        '''Returns the perturbed pixel-space image.'''
        return x_pix + self.delta_pixel(x_pix, quantize=quantize)

    def _base_metadata(self):
        return {
            'dataset_flag': self.dataset_flag,
            'epsilon': self.epsilon,
            'norm': self.norm,
            'in_channels': self.in_channels,
            'quantize': self.quantize,
            'init_std': self.init_std,
            'adaptive_floor': self.adaptive_floor,
        }


class NoiseGenerator(_BoundedPerturbation):
    """
    Small conv encoder-decoder mapping an image to a bounded perturbation.

    The network consumes a *normalized* image (what the classifiers see), works
    internally in pixel space, and returns a perturbation in normalized space
    so that it can be added straight onto the classifier's input.
    """

    def __init__(
        self,
        dataset_flag='cifar',
        epsilon=8 / 255,
        norm='linf',
        width=32,
        in_channels=3,
        quantize=False,
        init_std=1e-3,
        adaptive_floor=None,
    ):
        super().__init__(dataset_flag=dataset_flag, epsilon=epsilon,
                         norm=norm, in_channels=in_channels,
                         quantize=quantize, init_std=init_std,
                         adaptive_floor=adaptive_floor)
        self.width = width

        w = width

        def block(c_in, c_out, stride=1):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1,
                          padding_mode='reflect'),
                nn.GroupNorm(min(8, c_out), c_out),
                nn.LeakyReLU(0.1, inplace=True),
            )

        # GroupNorm rather than BatchNorm: the generator is applied to batches
        # whose composition varies with the flip-candidate mask, and we do not
        # want its output to depend on which other images share the batch.
        self.encoder = nn.Sequential(
            block(in_channels, w),
            block(w, 2 * w, stride=2),
            block(2 * w, 2 * w),
        )
        self.bottleneck = nn.Sequential(
            block(2 * w, 2 * w),
            block(2 * w, 2 * w),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            block(2 * w, w),
            block(w, w),
        )
        self.head = nn.Conv2d(w, in_channels, 3, padding=1,
                              padding_mode='reflect')

        # Default init is near-zero, so the attack-side generator begins at
        # vanilla FLIP and has to earn its perturbation. A trigger generator
        # wants the opposite -- a visible starting signal for the backdoor
        # loss to shape -- so `train_trigger` raises init_std.
        self.init_std = init_std
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, std=init_std)

    def _body(self, x_pix):
        '''Runs the encoder-decoder, returning pre-tanh values.'''
        h = self.encoder(x_pix)
        h = self.bottleneck(h)
        h = self.decoder(h)
        return self.head(h)

    def metadata(self):
        '''Returns the constructor arguments, for reload without a config.'''
        meta = self._base_metadata()
        meta.update({'arch': 'conv', 'width': self.width})
        return meta


class _Block(nn.Module):
    '''Pre-norm transformer block.'''

    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class ViTNoiseGenerator(_BoundedPerturbation):
    """
    Vision transformer mapping an image to a bounded perturbation.

    Same contract as `NoiseGenerator` -- normalized image in, normalized
    perturbation out, epsilon enforced by construction -- with a transformer
    body in place of the conv encoder-decoder.

    Why a transformer here. The conv generator has a local receptive field, so
    its perturbation is a texture: the trigger it learns is essentially the
    same local pattern everywhere, which is both easy for a representation-space
    defense to isolate and, at larger epsilon, easy to see. Self-attention
    makes every patch's perturbation a function of the whole image, so the
    trigger can be *placed* -- concentrated where the image already has
    structure to hide in and suppressed on flat regions, where a perturbation
    is most visible.

    Two details keep the output from looking like a transformer's output:

      - **A smoothing conv after unpatchify.** Patch-wise heads produce visible
        seams at patch boundaries; a 3x3 convolution spanning those boundaries
        removes them. It is initialized to the identity so it starts as a
        no-op and only learns to smooth if that helps.
      - **The head init is scaled by 1/sqrt(dim).** The head consumes
        LayerNorm'd features of unit scale, so an unscaled `init_std` would
        give pre-tanh values of order `init_std * sqrt(dim)` and saturate tanh
        immediately. Scaling makes `init_std` mean the same pre-tanh magnitude
        it means for the conv generator.
    """

    def __init__(
        self,
        dataset_flag='cifar',
        epsilon=8 / 255,
        norm='linf',
        dim=192,
        depth=4,
        heads=3,
        patch_size=4,
        image_size=None,
        mlp_ratio=4.0,
        in_channels=3,
        quantize=False,
        init_std=1e-3,
        adaptive_floor=None,
    ):
        super().__init__(dataset_flag=dataset_flag, epsilon=epsilon,
                         norm=norm, in_channels=in_channels,
                         quantize=quantize, init_std=init_std,
                         adaptive_floor=adaptive_floor)

        if image_size is None:
            image_size = IMAGE_SIZE.get(dataset_flag, 32)
        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size {image_size} is not divisible by patch_size "
                f"{patch_size}.")

        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.patch_size = patch_size
        self.image_size = image_size
        self.mlp_ratio = mlp_ratio
        self.grid = image_size // patch_size
        n_patches = self.grid ** 2

        self.patch_embed = nn.Conv2d(in_channels, dim, patch_size,
                                     stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList(
            [_Block(dim, heads, mlp_ratio) for _ in range(depth)])
        # Named `final_norm`, not `norm`: the base class keeps the *bound's*
        # norm ("linf"/"l2") on `self.norm`, and shadowing it with a LayerNorm
        # would silently send every L-infinity generator down the L2 branch.
        self.final_norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, patch_size * patch_size * in_channels)

        self.smooth = nn.Conv2d(in_channels, in_channels, 3, padding=1,
                                padding_mode='reflect')

        # Head init scaled by 1/sqrt(dim); see the class docstring.
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, std=init_std / (dim ** 0.5))

        # Smoothing conv starts as the identity, so it is a no-op until it
        # earns otherwise.
        with torch.no_grad():
            self.smooth.weight.zero_()
            for c in range(in_channels):
                self.smooth.weight[c, c, 1, 1] = 1.0
            self.smooth.bias.zero_()

    def _unpatchify(self, tokens):
        '''(B, n_patches, p*p*C) -> (B, C, H, W).'''
        b = tokens.shape[0]
        p, g, c = self.patch_size, self.grid, self.in_channels
        x = tokens.view(b, g, g, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.view(b, c, g * p, g * p)

    def _body(self, x_pix):
        '''Runs the transformer, returning pre-tanh values.'''
        if x_pix.shape[-1] != self.image_size:
            raise ValueError(
                f"{type(self).__name__} was built for {self.image_size}px "
                f"images but got {x_pix.shape[-1]}px. Rebuild it with "
                f"image_size={x_pix.shape[-1]}.")

        h = self.patch_embed(x_pix).flatten(2).transpose(1, 2)
        h = h + self.pos_embed
        for block in self.blocks:
            h = block(h)
        h = self.final_norm(h)
        return self.smooth(self._unpatchify(self.head(h)))

    def metadata(self):
        '''Returns the constructor arguments, for reload without a config.'''
        meta = self._base_metadata()
        meta.update({
            'arch': 'vit',
            'dim': self.dim,
            'depth': self.depth,
            'heads': self.heads,
            'patch_size': self.patch_size,
            'image_size': self.image_size,
            'mlp_ratio': self.mlp_ratio,
        })
        return meta


GENERATOR_ARCHS = {'conv': NoiseGenerator, 'vit': ViTNoiseGenerator}

# Per-architecture head init, as a pre-tanh scale. The ViT head reads
# LayerNorm'd features so init_std is literally the pre-tanh std and needs to
# be ~1 to reach the epsilon ball; the conv head reads unnormed activations
# and already saturates from 0.1.
DEFAULT_INIT_STD = {'conv': 0.1, 'vit': 1.0}


def build_generator(arch='conv', **kwargs):
    '''Builds a generator of the named architecture.'''
    if arch not in GENERATOR_ARCHS:
        raise ValueError(
            f"Unknown generator arch {arch!r}; expected one of "
            f"{sorted(GENERATOR_ARCHS)}.")
    return GENERATOR_ARCHS[arch](**kwargs)


def save_generator(generator, output_dir, filename='generator'):
    '''Saves a generator's weights alongside its constructor metadata.'''
    output_dir = str(output_dir)
    if not output_dir.endswith('/'):
        output_dir += '/'
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    torch.save(generator.state_dict(), f'{output_dir}{filename}.pth')
    with open(f'{output_dir}{filename}.json', 'w') as f:
        json.dump(generator.metadata(), f, indent=2)


def load_generator(generator_pth, device=None):
    '''Rebuilds a generator from the files written by `save_generator`.'''
    meta_path = str(generator_pth).rsplit('.', 1)[0] + '.json'
    with open(meta_path) as f:
        metadata = json.load(f)

    # Generators saved before the ViT existed have no 'arch' key, so default
    # to the conv architecture rather than failing on an old checkpoint.
    arch = metadata.pop('arch', 'conv')
    generator = build_generator(arch, **metadata)
    state = torch.load(generator_pth, map_location=device or 'cpu')
    generator.load_state_dict(state)
    generator.eval()

    if device is not None:
        generator = generator.to(device)
    return generator


def flip_candidate_mask(train_labels, source_label, target_label):
    """
    Marks the examples whose labels the attack is allowed to flip.

    These are the images the generator is allowed to perturb: in the targeted
    setting only source-class images can be flipped, and when source_label is
    -1 every non-target image is a candidate. Keeping the perturbation on the
    candidate set (rather than on every image) matches the threat model, where
    only the examples the attacker touches differ from the clean dataset.
    """
    train_labels = np.asarray(train_labels)
    if source_label == -1:
        return train_labels != target_label
    return train_labels == source_label


def flipped_indices(poisoned_labels, true_labels):
    """
    Recovers which examples a `select_flips` output actually flipped.

    `select_flips` writes the true labels everywhere except at the flipped
    indices, so an argmax disagreement identifies the flipped set exactly.
    """
    poisoned_labels = np.asarray(poisoned_labels)
    true_labels = np.asarray(true_labels)
    return np.where(
        poisoned_labels.argmax(axis=1) != true_labels.argmax(axis=1)
    )[0]


class PerturbedImageDataset(Dataset):
    """
    A raw (PIL) dataset with the generator's perturbation baked into an index
    subset, as 8-bit images.

    This sits *before* the user's augmentation pipeline on purpose: the threat
    model is that the attacker hands over a dataset, and the victim then
    augments it however they like. Baking the perturbation into uint8 images
    also means the epsilon bound reported in the results is the one that
    actually survives to disk.
    """

    def __init__(self, dataset, generator, indices, device=None,
                 batch_size=256):
        self.dataset = dataset
        self.indices = set(int(i) for i in indices)
        self.perturbed = {}

        if not self.indices:
            return

        device = device or next(generator.parameters()).device
        generator = generator.to(device).eval()

        ordered = sorted(self.indices)
        with torch.no_grad():
            for start in range(0, len(ordered), batch_size):
                chunk = ordered[start:start + batch_size]
                images = [self.dataset[i][0] for i in chunk]
                x_pix = torch.stack([
                    torch.from_numpy(
                        np.asarray(im, dtype=np.float32).transpose(2, 0, 1)
                    ) / 255
                    for im in images
                ]).to(device)

                x_adv = generator.perturb_pixel(x_pix, quantize=True)
                x_adv = (x_adv.clamp(0, 1) * 255).round().to(torch.uint8)
                x_adv = x_adv.permute(0, 2, 3, 1).cpu().numpy()

                for i, arr in zip(chunk, x_adv):
                    self.perturbed[i] = Image.fromarray(arr)

    def __getitem__(self, i):
        x, y = self.dataset[i]
        if i in self.perturbed:
            return self.perturbed[i], y
        return x, y

    def __len__(self):
        return len(self.dataset)

    def perturbation_stats(self):
        '''Reports the realized perturbation sizes, for sanity checking.'''
        if not self.perturbed:
            return {'n_perturbed': 0}

        linf, l2 = [], []
        for i, im in self.perturbed.items():
            # Differenced as integers, then scaled: a float32 subtraction of
            # two n/255 values lands slightly off the grid and would report an
            # epsilon marginally above the true bound.
            clean = np.asarray(self.dataset[i][0], dtype=np.int16)
            dirty = np.asarray(im, dtype=np.int16)
            d = (dirty - clean) / 255.0
            linf.append(np.abs(d).max())
            l2.append(np.sqrt((d ** 2).sum()))

        return {
            'n_perturbed': len(self.perturbed),
            'linf_mean': float(np.mean(linf)),
            'linf_max': float(np.max(linf)),
            'linf_max_255': float(np.max(linf) * 255),
            'l2_mean': float(np.mean(l2)),
        }


IMAGE_SIZE = {'cifar': 32, 'cifar_100': 32, 'tiny_imagenet': 64}

DEFAULT_TRIGGER_CONFIG = {
    'epsilon': 8 / 255,     # L-infinity bound on the trigger, in pixel units
    'norm': 'linf',
    'arch': 'vit',          # {vit, conv}; see GENERATOR_ARCHS
    # ViT body
    'dim': 192,
    'depth': 4,
    'heads': 3,
    'patch_size': 4,
    'mlp_ratio': 4.0,
    # conv body
    'width': 32,            # base channel count of G
    # imperceptibility
    'lambda_tv': 0.01,      # total-variation penalty on the perturbation
    'd_width': 32,          # base channel count of the discriminator
    'g_lr': 1e-3,
    'd_lr': 2e-4,
    'f_lr': 0.01,           # surrogate classifier (SGD)
    'lambda_bd': 1.0,       # weight on backdoor efficacy
    'lambda_adv': 0.0,      # weight on fooling the discriminator;
                            # 0 drops D entirely (the default now that
                            # the ViT replaces the GAN)
    'epochs': 10,
    'batch_size': 256,
    'warmup': 1,            # epochs of surrogate-only training before G moves
    # Head init, well above the attack-side default so the trigger starts with
    # real signal rather than ~0. None resolves per architecture: for the ViT
    # the head reads LayerNorm'd features, so init_std *is* the pre-tanh std
    # and needs to be ~1 to use the epsilon ball; the conv head reads unnormed
    # activations and saturates from 0.1.
    'init_std': None,
    'quantize': True,
    # Image adaptivity. Both off by default, which reproduces the earlier
    # runs exactly.
    #   adaptive_floor: None = off. A value in [0, 1] scales the bound per
    #       pixel by a texture map, so flat regions get only that share of
    #       epsilon (see `local_budget`).
    #   lambda_div / div_tau: penalty on the mean pairwise cosine similarity
    #       between the perturbations of different images, applied above
    #       div_tau. Without it the generator converges to one pattern shared
    #       by every image (measured cosine ~0.98).
    'adaptive_floor': None,
    'lambda_div': 0.0,
    'div_tau': 0.3,
}


def coalesce_trigger_config(trigger_config):
    '''Coalesces a GAN-trigger config with the defaults, resolving epsilon.'''
    config = {**DEFAULT_TRIGGER_CONFIG, **(trigger_config or {})}

    if 'epsilon_255' in config:
        config['epsilon'] = float(config.pop('epsilon_255')) / 255

    config['epsilon'] = float(config['epsilon'])
    if config['epsilon'] <= 0:
        raise ValueError("epsilon must be positive.")
    if config['norm'] not in ('linf', 'l2'):
        raise ValueError("norm must be one of {linf, l2}.")
    if config['arch'] not in GENERATOR_ARCHS:
        raise ValueError(
            f"arch must be one of {sorted(GENERATOR_ARCHS)}.")

    if config.get('init_std') is None:
        config['init_std'] = DEFAULT_INIT_STD[config['arch']]

    return config


def trigger_generator_kwargs(config, dataset_flag):
    '''Picks the constructor arguments the chosen architecture accepts.'''
    shared = {
        'dataset_flag': dataset_flag,
        'epsilon': config['epsilon'],
        'norm': config['norm'],
        'init_std': config['init_std'],
        'quantize': False,   # keep it differentiable during training
        'adaptive_floor': config.get('adaptive_floor'),
    }
    if config['arch'] == 'vit':
        shared.update({
            'dim': config['dim'],
            'depth': config['depth'],
            'heads': config['heads'],
            'patch_size': config['patch_size'],
            'mlp_ratio': config['mlp_ratio'],
        })
    else:
        shared['width'] = config['width']
    return shared


def total_variation(delta):
    """
    Mean total variation of a perturbation.

    Penalizing this is what keeps the trigger from becoming high-frequency
    speckle, which is the most visible thing a bounded perturbation can be:
    the eye is far more sensitive to pixel-scale noise than to a smooth tint of
    the same amplitude.
    """
    dh = (delta[..., 1:, :] - delta[..., :-1, :]).abs().mean()
    dw = (delta[..., :, 1:] - delta[..., :, :-1]).abs().mean()
    return dh + dw


def pairwise_cosine(delta):
    '''
    Mean cosine similarity between the perturbations of different images.

    1.0 means every image gets the same perturbation (a universal trigger);
    values near 0 mean the perturbation differs from image to image. Needs at
    least two images, and returns 0 for fewer.
    '''
    n = delta.shape[0]
    if n < 2:
        return delta.sum() * 0.0
    flat = torch.nn.functional.normalize(delta.flatten(1), dim=1, eps=1e-8)
    gram = flat @ flat.t()
    return (gram.sum() - gram.diag().sum()) / (n * (n - 1))


class Discriminator(nn.Module):
    """
    Tells a clean image from a triggered one.

    Consumes a *pixel-space* image in [0, 1] and returns a single logit, high
    for 'clean'. GroupNorm rather than BatchNorm so the verdict on an image
    does not depend on which others share its batch -- clean and triggered
    images arrive in separate batches, which is exactly the situation in which
    BatchNorm leaks the answer.
    """

    def __init__(self, in_channels=3, width=32):
        super().__init__()
        w = width

        def block(c_in, c_out, stride):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 4, stride=stride, padding=1),
                nn.GroupNorm(min(8, c_out), c_out),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, w, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            block(w, 2 * w, 2),
            block(2 * w, 4 * w, 2),
        )
        self.head = nn.Conv2d(4 * w, 1, 3, padding=1)

    def forward(self, x_pix):
        h = self.features(x_pix)
        return self.head(h).flatten(1).mean(dim=1)


class GeneratorPoisoner(Poisoner):
    """
    A `Poisoner` whose trigger is produced by a frozen generator.

    This is the drop-in replacement for `StripePoisoner` and friends: it takes
    a raw PIL image and returns a raw PIL image with the trigger applied, so
    expert training, the FLIP attack, and the test-time PTA evaluation all pick
    it up through `pick_poisoner` without further changes.

    The trigger is applied in pixel space and rounded to the 8-bit grid, so the
    epsilon bound is the one that survives into the uint8 image the victim
    receives -- the same honesty the attack-side generator had.

    `G` is frozen and deterministic, so results are memoized on the raw image
    bytes. Without the cache every epoch re-runs the generator over the whole
    poison set inside the dataloader workers; with it, only the first epoch
    pays. The poison set is one class' worth of images, so the cache is a few
    tens of MB.
    """

    def __init__(self, generator, device='cpu', cache=True):
        self.generator = generator.to(device).eval()
        self.device = device
        self.cache = {} if cache else None

    def poison(self, x: Image.Image) -> Image.Image:
        key = x.tobytes() if self.cache is not None else None
        if key is not None and key in self.cache:
            return Image.fromarray(self.cache[key])

        arr = np.asarray(x, dtype=np.float32) / 255.0
        x_pix = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        x_pix = x_pix.to(self.device)

        with torch.no_grad():
            delta = self.generator.delta_pixel(x_pix, quantize=True)
            out = torch.clamp(x_pix + delta, 0.0, 1.0)

        out = (out.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0)
        out = np.uint8(np.round(out))

        if key is not None:
            self.cache[key] = out
        return Image.fromarray(out)
