# model_b2.py
#!/usr/bin/env python3
"""
Ablation B2 — + Monotone Occupancy (Independent Heads).

Architecture:
    PolarSamplingGrid → PolarEncoderDecoder (shared UNet w/ circular padding)
        ├── SimpleMonotoneHead (disc) — cumulative-decrement monotone
        └── SimpleMonotoneHead (cup)  — cumulative-decrement monotone (INDEPENDENT)
    PolarToCartesianWarper → Cartesian masks
    GeometryExtractor → radii from dense masks

Key properties:
    - Both disc and cup use independent monotone occupancy heads
    - NO nesting constraint: P_c is NOT guaranteed ≤ P_d
    - NO shape prior, NO confidence gating, NO band anchoring
    - Same PolarEncoderDecoder backbone as full V3.1 (GroupNorm in early stages)

Changes from B1 (Polar UNet):
    - Replace sigmoid output heads with cumulative-decrement monotone heads
    - Radial monotonicity is now guaranteed per-head
    - Add GeometryExtractor for radii extraction

Changes from B3:
    - Cup head is independent (not factorized through P_d)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    IMAGE_SIZE, N_THETA, N_RHO, TAU,
    ENCODER_FEATURES,
)


# ==============================================================================
# §1 — POLAR SAMPLING GRID (from ThreeSixty/model.py, unchanged)
# ==============================================================================

class PolarSamplingGrid(nn.Module):
    """Warp a Cartesian image to polar coordinates via differentiable bilinear
    sampling.  Assumes the optic disc is centred in the crop."""

    def __init__(self, image_size=IMAGE_SIZE, n_theta=N_THETA, n_rho=N_RHO):
        super().__init__()
        self.image_size = image_size
        self.n_theta = n_theta
        self.n_rho = n_rho

        H = W = image_size
        cx = (W - 1) / 2.0
        cy = (H - 1) / 2.0
        R = min(H, W) / 2.0

        theta = torch.linspace(0, 2 * math.pi, n_theta + 1)[:n_theta]
        rho = torch.linspace(0, 1, n_rho)

        rho_2d = rho.unsqueeze(1)
        theta_2d = theta.unsqueeze(0)

        px = cx + R * rho_2d * torch.cos(theta_2d)
        py = cy + R * rho_2d * torch.sin(theta_2d)

        grid_x = 2.0 * px / (W - 1) - 1.0
        grid_y = 2.0 * py / (H - 1) - 1.0

        grid = torch.stack([grid_x, grid_y], dim=-1)
        self.register_buffer('grid', grid.unsqueeze(0))
        self.register_buffer('rho', rho)
        self.register_buffer('theta', theta)
        self.register_buffer('_cx', torch.tensor(cx))
        self.register_buffer('_cy', torch.tensor(cy))
        self.register_buffer('_R', torch.tensor(R))

    def forward(self, image):
        B = image.shape[0]
        grid = self.grid.expand(B, -1, -1, -1)
        return F.grid_sample(image, grid, mode='bilinear',
                             padding_mode='zeros', align_corners=True)


# ==============================================================================
# §2 — CIRCULAR PADDING CONVOLUTIONS (from ThreeSixty/model.py, unchanged)
# ==============================================================================

class CircularPadConv2d(nn.Module):
    """Conv2d with circular padding along θ (width), zero-padding along ρ."""

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, bias=False):
        super().__init__()
        self.pad_rho = kernel_size // 2
        self.pad_theta = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=0, bias=bias)

    def forward(self, x):
        x = F.pad(x, (self.pad_theta, self.pad_theta, 0, 0), mode='circular')
        x = F.pad(x, (0, 0, self.pad_rho, self.pad_rho), mode='constant', value=0)
        return self.conv(x)


class CircularDoubleConv(nn.Module):
    """Two successive CircularPadConv2d → Norm → ReLU."""

    def __init__(self, in_channels, out_channels, use_groupnorm=False):
        super().__init__()
        if use_groupnorm:
            num_groups = min(8, out_channels)
            norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
            norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        else:
            norm1 = nn.BatchNorm2d(out_channels)
            norm2 = nn.BatchNorm2d(out_channels)

        self.block = nn.Sequential(
            CircularPadConv2d(in_channels, out_channels, 3),
            norm1,
            nn.ReLU(inplace=True),
            CircularPadConv2d(out_channels, out_channels, 3),
            norm2,
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ==============================================================================
# §2b — POLAR ENCODER-DECODER (from ThreeSixty/model.py, unchanged)
# ==============================================================================

class PolarEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_groupnorm=False):
        super().__init__()
        self.double_conv = CircularDoubleConv(in_channels, out_channels,
                                              use_groupnorm=use_groupnorm)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        skip = self.double_conv(x)
        pooled = self.pool(skip)
        return skip, pooled


class PolarDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_groupnorm=False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        self.double_conv = CircularDoubleConv(in_channels, out_channels,
                                              use_groupnorm=use_groupnorm)

    def forward(self, x, skip):
        x = self.up(x)
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        if dy or dx:
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x = torch.cat([skip, x], dim=1)
        return self.double_conv(x)


class PolarEncoderDecoder(nn.Module):
    """UNet in polar space with circular padding along θ.
    GroupNorm in first two encoder/decoder stages for OOD robustness."""

    def __init__(self, in_channels=3, features=None, groupnorm_stages=2):
        super().__init__()
        if features is None:
            features = ENCODER_FEATURES
        self.features = features
        bottleneck_ch = features[-1] * 2

        self.encoders = nn.ModuleList()
        ch = in_channels
        for i, f in enumerate(features):
            use_gn = (i < groupnorm_stages)
            self.encoders.append(PolarEncoderBlock(ch, f, use_groupnorm=use_gn))
            ch = f

        self.bottleneck = CircularDoubleConv(features[-1], bottleneck_ch)

        self.decoders = nn.ModuleList()
        n = len(features)
        for i, f in enumerate(reversed(features)):
            use_gn = ((n - 1 - i) < groupnorm_stages)
            self.decoders.append(PolarDecoderBlock(f * 2, f, use_groupnorm=use_gn))

    def forward(self, x):
        skip_connections = []
        for encoder in self.encoders:
            skip, x = encoder(x)
            skip_connections.append(skip)
        x = self.bottleneck(x)
        skips_rev = skip_connections[::-1]
        for idx, decoder in enumerate(self.decoders):
            x = decoder(x, skips_rev[idx])
        return x


# ==============================================================================
# §NEW — SIMPLE MONOTONE HEAD (B2-specific)
# ==============================================================================

class SimpleMonotoneHead(nn.Module):
    """Independent monotone head via cumulative decrement.

    Used for BOTH disc and cup in B2 (no nesting constraint between them).

    Q(ρ,θ) = σ(a(θ) − Σ_{m≤j} softplus(δ(ρ_m,θ)))

    The cumulative sum of positive decrements ensures the logit is
    monotone decreasing in ρ, so the sigmoid output P is also
    monotone decreasing: P(ρ₁,θ) ≥ P(ρ₂,θ) for ρ₁ < ρ₂.
    """

    def __init__(self, in_channels):
        super().__init__()
        # Per-angle bias: pool over ρ → predict a(θ)
        self.bias_pool = nn.AdaptiveAvgPool2d((1, None))
        self.bias_conv = nn.Conv1d(in_channels, 1, 1)
        # Per-pixel decrement
        self.dec_conv = nn.Conv2d(in_channels, 1, 1)

    def forward(self, features):
        """
        Args:
            features: (B, C, N_ρ, N_θ)

        Returns:
            P: (B, 1, N_ρ, N_θ) monotone occupancy mask
        """
        # Per-angle bias a(θ)
        pooled = self.bias_pool(features).squeeze(2)  # (B, C, N_θ)
        a = self.bias_conv(pooled)                      # (B, 1, N_θ)

        # Per-pixel positive decrement δ(ρ,θ)
        delta = F.softplus(self.dec_conv(features))     # (B, 1, N_ρ, N_θ)

        # Cumulative sum along ρ (dim=2)
        cumsum = torch.cumsum(delta, dim=2)             # (B, 1, N_ρ, N_θ)

        # Monotone logits: a(θ) − Σ δ
        L = a.unsqueeze(2) - cumsum                     # (B, 1, N_ρ, N_θ)

        # Sigmoid → monotone occupancy
        P = torch.sigmoid(L)                            # (B, 1, N_ρ, N_θ)
        return P


# ==============================================================================
# §7 — POLAR → CARTESIAN WARPER (from ThreeSixty/model.py, unchanged)
# ==============================================================================

class PolarToCartesianWarper(nn.Module):
    """Warp dense polar masks back to Cartesian space via grid_sample."""

    def __init__(self, image_size=IMAGE_SIZE, n_theta=N_THETA, n_rho=N_RHO):
        super().__init__()
        self.image_size = image_size
        self.n_theta = n_theta
        self.n_rho = n_rho

        H = W = image_size
        cx = (W - 1) / 2.0
        cy = (H - 1) / 2.0
        R = min(H, W) / 2.0

        ys = torch.arange(H, dtype=torch.float32)
        xs = torch.arange(W, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')

        rho_cart = torch.sqrt((grid_x - cx) ** 2 + (grid_y - cy) ** 2) / R
        theta_cart = torch.atan2(grid_y - cy, grid_x - cx) % (2 * math.pi)
        inside_circle = (rho_cart <= 1.0).float()

        inv_grid_y = 2.0 * rho_cart - 1.0
        inv_grid_x = theta_cart / math.pi - 1.0

        inv_grid = torch.stack([inv_grid_x, inv_grid_y], dim=-1)

        self.register_buffer('inv_grid', inv_grid.unsqueeze(0))
        self.register_buffer('inside_circle', inside_circle)

    def forward(self, polar_mask):
        B = polar_mask.shape[0]
        padded = F.pad(polar_mask, (0, 1, 0, 0), mode='circular')
        grid = self.inv_grid.expand(B, -1, -1, -1)
        cart = F.grid_sample(padded, grid, mode='bilinear',
                             padding_mode='border', align_corners=True)
        cart = cart * self.inside_circle.unsqueeze(0).unsqueeze(0)
        return cart


# ==============================================================================
# §8 — GEOMETRY EXTRACTOR (from ThreeSixty/model.py, unchanged)
# ==============================================================================

class GeometryExtractor(nn.Module):
    """Extract radial boundary functions from dense monotone masks."""

    def __init__(self, n_rho=N_RHO):
        super().__init__()
        rho = torch.linspace(0, 1, n_rho)
        self.register_buffer('rho', rho)

    def forward(self, P):
        if P.dim() == 4:
            P = P.squeeze(1)

        b = P[:, :-1, :] - P[:, 1:, :]
        b_last = P[:, -1:, :]
        b = torch.cat([b, b_last], dim=1)
        b = b.clamp(min=0.0)

        b_sum = b.sum(dim=1, keepdim=True) + 1e-8
        b_norm = b / b_sum

        rho = self.rho.view(1, -1, 1)
        r = (rho * b_norm).sum(dim=1)

        return r


# ==============================================================================
# B2 MODEL — INDEPENDENT MONOTONE HEADS
# ==============================================================================

class AblationB2(nn.Module):
    """Ablation B2: + Monotone Occupancy (Independent Heads).

    Architecture:
        PolarSamplingGrid → PolarEncoderDecoder
            ├── SimpleMonotoneHead (disc) → P_d
            └── SimpleMonotoneHead (cup)  → P_c  (independent, NOT nested)
        PolarToCartesianWarper → Ŷ_d, Ŷ_c
        GeometryExtractor → r_d^m, r_c^m

    Output keys (compatible with ablation loss and inference):
        'P_d_polar':   (B, 1, N_ρ, N_θ)
        'P_c_polar':   (B, 1, N_ρ, N_θ)
        'Y_d_cart':    (B, 1, H, W)
        'Y_c_cart':    (B, 1, H, W)
        'r_d_m':       (B, N_θ)
        'r_c_m':       (B, N_θ)
    """

    def __init__(self, image_size=IMAGE_SIZE, n_theta=N_THETA, n_rho=N_RHO,
                 features=None):
        super().__init__()
        if features is None:
            features = ENCODER_FEATURES

        self.image_size = image_size
        self.n_theta = n_theta
        self.n_rho = n_rho

        # §1 — Polar sampling grid
        self.polar_grid = PolarSamplingGrid(image_size, n_theta, n_rho)

        # §2 — Shared backbone (same as V3.1)
        self.polar_unet = PolarEncoderDecoder(in_channels=3, features=features)

        # Independent monotone heads
        self.disc_head = SimpleMonotoneHead(in_channels=features[0])
        self.cup_head  = SimpleMonotoneHead(in_channels=features[0])

        # §7 — Warper
        self.warper = PolarToCartesianWarper(image_size, n_theta, n_rho)

        # §8 — Geometry extractor
        self.geom = GeometryExtractor(n_rho=n_rho)

        self._print_param_table()

    def _print_param_table(self):
        def count(m):
            return sum(p.numel() for p in m.parameters())

        mods = {
            'Polar Sampling Grid':     self.polar_grid,
            'Polar Encoder-Decoder':   self.polar_unet,
            'Disc Monotone Head':      self.disc_head,
            'Cup Monotone Head':       self.cup_head,
            'Polar→Cartesian Warper':  self.warper,
            'Geometry Extractor':      self.geom,
        }
        tot = sum(count(m) for m in mods.values())

        print("\n" + "-" * 62)
        print(f"{'Ablation B2 Component':<38} {'Params':>10}  {'%':>6}")
        print("-" * 62)
        for name, mod in mods.items():
            c = count(mod)
            print(f"{name:<38} {c:>10,}  {100*c/max(tot,1):>5.1f}%")
        print("-" * 62)
        print(f"{'TOTAL LEARNABLE':<38} {tot:>10,}")
        print(f"N_θ={self.n_theta}, N_ρ={self.n_rho}")
        print("-" * 62 + "\n")

    def forward(self, x):
        # §1 — Warp to polar
        polar_image = self.polar_grid(x)

        # §2 — Shared backbone
        features = self.polar_unet(polar_image)

        # Independent monotone heads
        P_d = self.disc_head(features)    # (B, 1, N_ρ, N_θ)
        P_c = self.cup_head(features)     # (B, 1, N_ρ, N_θ) — NOT nested

        # §7 — Polar → Cartesian
        Y_d_cart = self.warper(P_d)
        Y_c_cart = self.warper(P_c)

        # §8 — Extract geometry
        r_d_m = self.geom(P_d)
        r_c_m = self.geom(P_c)

        return {
            'P_d_polar': P_d,
            'P_c_polar': P_c,
            'Y_d_cart':  Y_d_cart,
            'Y_c_cart':  Y_c_cart,
            'r_d_m':     r_d_m,
            'r_c_m':     r_c_m,
        }
