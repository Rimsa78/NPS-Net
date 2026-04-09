# model_b3.py
#!/usr/bin/env python3
"""
Ablation B3 — + Factorized Nesting (P_c = P_d · Q).

Architecture:
    PolarSamplingGrid → PolarEncoderDecoder (shared UNet w/ circular padding)
        ├── SimpleMonotoneHead (disc) → P_d
        └── CupGateHead              → L_c → Q = σ(L_c)
    P_c = P_d · Q  (nested by construction: 0 ≤ P_c ≤ P_d ≤ 1)
    PolarToCartesianWarper → Cartesian masks
    GeometryExtractor → radii from dense masks

Key properties:
    - Disc head: same SimpleMonotoneHead as B2 (NOT band-anchored)
     - Cup head: CupGateHead for nesting constraint
    - Nesting guaranteed: P_c = P_d · Q where Q ∈ [0,1]
    - NO shape prior, NO confidence gating

Changes from B2:
    - Cup head replaced with CupGateHead (factorized through P_d)
    - Nesting violation should drop to 0%

Changes from B4:
    - No shape prior branch
    - No confidence-gated fusion
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    IMAGE_SIZE,
    N_THETA,
    N_RHO,
    TAU,
    ENCODER_FEATURES,
)

# Import shared components from B2 (they are identical)
from model_b2 import (
    PolarSamplingGrid,
    CircularPadConv2d,
    CircularDoubleConv,
    PolarEncoderBlock,
    PolarDecoderBlock,
    PolarEncoderDecoder,
    SimpleMonotoneHead,
    PolarToCartesianWarper,
    GeometryExtractor,
)


# ==============================================================================
# CUP GATE HEAD
# ==============================================================================


class CupGateHead(nn.Module):
    """Monotone cup-gate head using cumulative-decrement formulation.

    Q(ρ,θ) = σ(a_c(θ) − Σ_{m≤j} δ_c(ρ_m,θ))

    Final cup: P_c = P_d · Q  (nested by construction)
    """

    def __init__(self, in_channels):
        super().__init__()
        self.cup_bias_pool = nn.AdaptiveAvgPool2d((1, None))
        self.cup_bias_conv = nn.Conv1d(in_channels, 1, 1)
        self.cup_dec_conv = nn.Conv2d(in_channels, 1, 1)

    def forward(self, features):
        """
        Args:
            features: (B, C, N_ρ, N_θ)

        Returns:
            L_c: (B, 1, N_ρ, N_θ) cup-gate logits (monotone decreasing in ρ)
        """
        cup_pooled = self.cup_bias_pool(features).squeeze(2)
        a_c = self.cup_bias_conv(cup_pooled)
        delta_c = F.softplus(self.cup_dec_conv(features))
        cumsum_c = torch.cumsum(delta_c, dim=2)
        L_c = a_c.unsqueeze(2) - cumsum_c
        return L_c


# ==============================================================================
# B3 MODEL — FACTORIZED NESTING
# ==============================================================================


class AblationB3(nn.Module):
    """Ablation B3: + Factorized Nesting (P_c = P_d · Q).

    Architecture:
        PolarSamplingGrid → PolarEncoderDecoder
            ├── SimpleMonotoneHead (disc) → P_d
            └── CupGateHead              → L_c → Q = σ(L_c)
        P_c = P_d · Q   (nested by construction)
        PolarToCartesianWarper → Ŷ_d, Ŷ_c
        GeometryExtractor → r_d^m, r_c^m

    Output keys:
        'P_d_polar':   (B, 1, N_ρ, N_θ)
        'P_c_polar':   (B, 1, N_ρ, N_θ)
        'Y_d_cart':    (B, 1, H, W)
        'Y_c_cart':    (B, 1, H, W)
        'r_d_m':       (B, N_θ)
        'r_c_m':       (B, N_θ)
    """

    def __init__(
        self, image_size=IMAGE_SIZE, n_theta=N_THETA, n_rho=N_RHO, features=None
    ):
        super().__init__()
        if features is None:
            features = ENCODER_FEATURES

        self.image_size = image_size
        self.n_theta = n_theta
        self.n_rho = n_rho

        # Polar sampling grid
        self.polar_grid = PolarSamplingGrid(image_size, n_theta, n_rho)

        # Shared backbone
        self.polar_unet = PolarEncoderDecoder(in_channels=3, features=features)

        # Disc head: simple monotone (same as B2)
        self.disc_head = SimpleMonotoneHead(in_channels=features[0])

        # Cup head: CupGateHead (factorized nesting)
        self.cup_gate = CupGateHead(in_channels=features[0])

        # Warper
        self.warper = PolarToCartesianWarper(image_size, n_theta, n_rho)

        # Geometry extractor
        self.geom = GeometryExtractor(n_rho=n_rho)

        self._print_param_table()

    def _print_param_table(self):
        def count(m):
            return sum(p.numel() for p in m.parameters())

        mods = {
            "Polar Sampling Grid": self.polar_grid,
            "Polar Encoder-Decoder": self.polar_unet,
            "Disc Monotone Head": self.disc_head,
            "Cup Gate Head": self.cup_gate,
            "Polar→Cartesian Warper": self.warper,
            "Geometry Extractor": self.geom,
        }
        tot = sum(count(m) for m in mods.values())

        print("\n" + "-" * 62)
        print(f"{'Ablation B3 Component':<38} {'Params':>10}  {'%':>6}")
        print("-" * 62)
        for name, mod in mods.items():
            c = count(mod)
            print(f"{name:<38} {c:>10,}  {100 * c / max(tot, 1):>5.1f}%")
        print("-" * 62)
        print(f"{'TOTAL LEARNABLE':<38} {tot:>10,}")
        print(f"N_θ={self.n_theta}, N_ρ={self.n_rho}")
        print("-" * 62 + "\n")

    def forward(self, x):
        # Warp to polar
        polar_image = self.polar_grid(x)

        # Shared backbone
        features = self.polar_unet(polar_image)

        # Disc: simple monotone
        P_d = self.disc_head(features)  # (B, 1, N_ρ, N_θ)

        # Cup: factorized nesting P_c = P_d · Q
        L_c = self.cup_gate(features)  # (B, 1, N_ρ, N_θ)
        Q = torch.sigmoid(L_c)  # (B, 1, N_ρ, N_θ)
        P_c = P_d * Q  # nested: P_c ≤ P_d

        # Polar → Cartesian
        Y_d_cart = self.warper(P_d)
        Y_c_cart = self.warper(P_c)

        # Extract geometry
        r_d_m = self.geom(P_d)
        r_c_m = self.geom(P_c)

        return {
            "P_d_polar": P_d,
            "P_c_polar": P_c,
            "Y_d_cart": Y_d_cart,
            "Y_c_cart": Y_c_cart,
            "r_d_m": r_d_m,
            "r_c_m": r_c_m,
        }
