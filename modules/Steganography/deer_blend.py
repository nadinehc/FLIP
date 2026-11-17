import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, padding_mode="reflect"), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, padding_mode="reflect"), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
    )


def up_block(c_in, c_out):
    # Bilinear upsampling + conv instead of ConvTranspose2d(k=2, s=2), which produces a 2x2 checkerboard
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        nn.Conv2d(c_in, c_out, 3, padding=1, padding_mode="reflect"),
    )


def local_std(x, k=3):
    # Per-pixel std of the luminance over a k x k window (x in [-1, 1])
    g = x.mean(1, keepdim=True)
    p = k // 2
    gp = F.pad(g, [p] * 4, mode="reflect")
    mu = F.avg_pool2d(gp, k, 1)
    mu2 = F.avg_pool2d(gp ** 2, k, 1)
    return (mu2 - mu ** 2).clamp_min(0).sqrt()


def texture_mask(x, tau=0.08, floor=0.15):
    # Per-pixel budget multiplier in [floor, 1]: ~floor on flat regions, 1 on textured ones.
    # tau is the local std (in [-1, 1] units, 0.08 ~ 10/255) at which the full budget is allowed.
    m = (local_std(x) / tau).clamp(max=1.0)
    m = F.avg_pool2d(F.pad(m, [1] * 4, mode="reflect"), 3, 1)   # smooth the mask itself
    return floor + (1 - floor) * m


def gaussian_blur(x, sigma):
    if sigma <= 0:
        return x
    r = max(1, int(round(2 * sigma)))
    t = torch.arange(-r, r + 1, dtype=x.dtype, device=x.device)
    k = torch.exp(-t ** 2 / (2 * sigma ** 2)); k = k / k.sum()
    C = x.size(1)
    x = F.conv2d(F.pad(x, [r, r, 0, 0], mode="reflect"), k.view(1, 1, 1, -1).repeat(C, 1, 1, 1), groups=C)
    x = F.conv2d(F.pad(x, [0, 0, r, r], mode="reflect"), k.view(1, 1, -1, 1).repeat(C, 1, 1, 1), groups=C)
    return x


class DeerBlendEncoder(nn.Module):
    """U-Net: (truck, deer) -> perturbation delta with |delta| <= eps * mask(truck) <= eps.

    - texture mask: the per-pixel L_inf budget shrinks to `mask_floor * eps` on flat regions
    - blur_sigma > 0: low-pass the perturbation (a convex combination, so the bound still holds)
    - tanh_temp T: delta = eps * mask * tanh(z / T). T > 1 widens tanh's linear regime, so the
      network output z must grow T times larger before saturating (delays saturation).
    """

    def __init__(self, eps, base=32, mask_tau=0.08, mask_floor=0.15, blur_sigma=0.0, use_mask=True,
                 tanh_temp=1.0):
        super().__init__()
        self.eps = eps
        self.mask_tau, self.mask_floor = mask_tau, mask_floor
        self.blur_sigma, self.use_mask = blur_sigma, use_mask
        self.tanh_temp = tanh_temp
        self.enc1 = conv_block(6, base)                 # 32x32
        self.enc2 = conv_block(base, base * 2)          # 16x16
        self.enc3 = conv_block(base * 2, base * 4)      # 8x8
        self.pool = nn.AvgPool2d(2)
        self.up2 = up_block(base * 4, base * 2)
        self.dec2 = conv_block(base * 4, base * 2)
        self.up1 = up_block(base * 2, base)
        self.dec1 = conv_block(base * 2, base)
        self.out = nn.Conv2d(base, 3, 3, padding=1, padding_mode="reflect")

    def raw(self, image, deer):
        # Pre-tanh output z (useful for a saturation penalty)
        if deer.dim() == 3:
            deer = deer.unsqueeze(0).expand(image.size(0), -1, -1, -1)
        e1 = self.enc1(torch.cat([image, deer], 1))
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(e3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.out(d1)

    def budget(self, image):
        if not self.use_mask:
            return torch.ones_like(image[:, :1])
        return texture_mask(image, self.mask_tau, self.mask_floor)

    def perturbation(self, image, deer, return_raw=False):
        # With return_raw, also returns the actual tanh argument z / T, so saturation_loss
        # (threshold on |z / T|) measures real saturation whatever the temperature.
        u = self.raw(image, deer)
        delta = gaussian_blur(torch.tanh(u), self.blur_sigma)
        delta = self.eps * self.budget(image) * delta
        return (delta, u) if return_raw else delta

    def forward(self, image, deer):
        return torch.clamp(image + self.perturbation(image, deer), -1.0, 1.0)


def tv_loss(delta, eps):
    # Anisotropic total variation of delta, in units of eps (penalises high-frequency patterns)
    d = delta / eps
    return (d[:, :, 1:, :] - d[:, :, :-1, :]).abs().mean() + (d[:, :, :, 1:] - d[:, :, :, :-1]).abs().mean()


def masked_l2_loss(delta, eps, image, tau=0.08):
    # L2 of delta/eps, weighted more heavily on flat regions where changes are most visible
    w = 1.0 + (1.0 - (local_std(image) / tau).clamp(max=1.0)) * 4.0     # 5x on flat, 1x on texture
    return (w * (delta / eps) ** 2).mean()


def saturation_loss(z, thresh=2.0):
    # Penalise pre-tanh values beyond +-thresh: keeps tanh out of its zero-gradient regime
    return F.relu(z.abs() - thresh).pow(2).mean()


# ---------------------------------------------------------------------------
# Discriminator: clean vs poisoned
# ---------------------------------------------------------------------------

def _srm_kernels():
    # Fixed high-pass filters from steganalysis (SRM): expose the noise residual, where
    # additive perturbations live, instead of letting the network rediscover them.
    kv = torch.tensor([[-1, 2, -2, 2, -1],
                       [2, -6, 8, -6, 2],
                       [-2, 8, -12, 8, -2],
                       [2, -6, 8, -6, 2],
                       [-1, 2, -2, 2, -1]], dtype=torch.float32) / 12.0
    first = torch.zeros(5, 5); first[2, 1], first[2, 2] = 1.0, -1.0           # horizontal 1st order
    first_v = first.t().clone()                                              # vertical 1st order
    second = torch.zeros(5, 5); second[2, 1:4] = torch.tensor([1.0, -2.0, 1.0]) / 2.0
    second_v = second.t().clone()
    return torch.stack([kv, first, first_v, second, second_v])               # [5, 5, 5]


class PoisonDiscriminator(nn.Module):
    """Predicts a logit: > 0 means 'clean', < 0 means 'poisoned'.

    Input in [-1, 1]. A fixed SRM high-pass layer (per RGB channel) is concatenated with the
    raw image, followed by a small spectrally-normalised CNN. No BatchNorm: clean and poisoned
    images are fed in separate halves of the batch, and BN statistics would leak the label.
    """

    def __init__(self, base=32, use_srm=True):
        super().__init__()
        self.use_srm = use_srm
        k = _srm_kernels()                                                   # [5, 5, 5]
        self.register_buffer("srm", k[:, None].repeat(3, 1, 1, 1))           # [15, 1, 5, 5], grouped per channel
        c_in = 3 + (15 if use_srm else 0)
        sn = nn.utils.spectral_norm
        act = lambda: nn.LeakyReLU(0.2, inplace=True)
        self.net = nn.Sequential(
            sn(nn.Conv2d(c_in, base, 3, padding=1)), act(),
            sn(nn.Conv2d(base, base, 3, padding=1)), act(),
            sn(nn.Conv2d(base, base * 2, 3, stride=2, padding=1)), act(),        # 16x16
            sn(nn.Conv2d(base * 2, base * 2, 3, padding=1)), act(),
            sn(nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1)), act(),    # 8x8
            sn(nn.Conv2d(base * 4, base * 4, 3, stride=2, padding=1)), act(),    # 4x4
        )
        self.head = sn(nn.Linear(base * 4, 1))

    def forward(self, x):
        if self.use_srm:
            res = F.conv2d(F.pad(x, [2] * 4, mode="reflect"), self.srm, groups=3) * 10.0   # residuals are tiny
            x = torch.cat([x, res], 1)
        h = self.net(x).mean((2, 3))                                         # global average pooling
        return self.head(h).squeeze(1)


def discriminator_loss(disc, x_clean, x_poison):
    # Standard BCE GAN loss for D: clean -> 1, poisoned -> 0. Returns (loss, accuracy)
    lc, lp = disc(x_clean), disc(x_poison)
    loss = F.binary_cross_entropy_with_logits(lc, torch.ones_like(lc)) + \
           F.binary_cross_entropy_with_logits(lp, torch.zeros_like(lp))
    acc = 0.5 * ((lc > 0).float().mean() + (lp < 0).float().mean())
    return loss, acc


def generator_adv_loss(disc, x_poison):
    # Non-saturating generator loss: make D call the poisoned image 'clean'
    lp = disc(x_poison)
    return F.binary_cross_entropy_with_logits(lp, torch.ones_like(lp))
