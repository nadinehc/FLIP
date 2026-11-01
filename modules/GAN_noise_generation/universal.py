"""
GAN for a *universal* trigger: one perturbation delta (3x32x32) shared by all images.

Pixel space convention: images in [0, 1] (no normalization). This matches FLIP's
poisoners, which act on PIL images *before* ToTensor/Normalize, so the learned
delta can be saved in pixel units and added exactly like the sinusoidal trigger.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)   # FLIP normalization (datasets.py)
CIFAR_STD = (0.2023, 0.1994, 0.2010)


class UniversalNoiseGenerator(nn.Module):
    """Maps a FIXED latent code z0 to a single perturbation delta.

    Because z0 never changes, the output is the same for every image: the noise is
    universal by construction. The generator is a (convolutional) reparametrization
    of delta, which gives it a smooth, spatially structured prior instead of
    i.i.d. pixel noise. ||delta||_inf <= eps is enforced with tanh.
    """

    def __init__(self, eps=6 / 255, latent_dim=100, ngf=64, seed=0):
        super().__init__()
        self.eps = eps
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("z0", torch.randn(1, latent_dim, generator=g))

        self.fc = nn.Sequential(nn.Linear(latent_dim, ngf * 4 * 4 * 4), nn.ReLU(True))
        self.net = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1), nn.ReLU(True),  # 8x8
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1), nn.ReLU(True),              # 16x16
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1),                                                         # 32x32
        )
        # No BatchNorm: the generator always sees a batch of size 1 (z0), and BN would make
        # the delta differ between train() and eval() mode.
        self.ngf = ngf

    def forward(self):
        h = self.fc(self.z0).view(1, self.ngf * 4, 4, 4)
        return self.eps * torch.tanh(self.net(h))  # (1, 3, 32, 32)

    @torch.no_grad()
    def delta_numpy_hwc_255(self):
        """Delta in PIL pixel units (HWC, range [-255*eps, 255*eps]) for the poisoner."""
        self.eval()
        d = self().squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
        return d.astype(np.float32)


class RealismDiscriminator(nn.Module):
    """Unconditional D: clean image (real) vs image + delta (fake).

    Spectral norm keeps D from saturating immediately (log(2)=0.693 forever means
    D is not learning anything, which is a useless signal for G).
    """

    def __init__(self, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(3, ndf, 3, 1, 1)), nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1)), nn.LeakyReLU(0.2, True),      # 16x16
            spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)), nn.LeakyReLU(0.2, True),  # 8x8
            spectral_norm(nn.Conv2d(ndf * 4, ndf * 4, 4, 2, 1)), nn.LeakyReLU(0.2, True),  # 4x4
            spectral_norm(nn.Conv2d(ndf * 4, 1, 4, 1, 0)),
        )

    def forward(self, x01):
        return self.net(x01 * 2 - 1).view(-1)  # logits


def normalize_cifar(x01):
    mean = torch.tensor(CIFAR_MEAN, device=x01.device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD, device=x01.device).view(1, 3, 1, 1)
    return (x01 - mean) / std


def apply_delta(x01, delta):
    """Same operation the PIL poisoner performs: add, then clip to the valid range."""
    return (x01 + delta).clamp(0, 1)


def quantize_like_pil(delta):
    """Round delta to integer pixel levels, as it will be after uint8 conversion."""
    return torch.round(delta * 255) / 255
