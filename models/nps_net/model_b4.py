# model_b4.py
#!/usr/bin/env python3
"""
Ablation B4 — + Shape Prior & Confidence Gating.

Architecture:
    PolarSamplingGrid → PolarEncoderDecoder (shared UNet w/ circular padding)
        ├── SimpleMonotoneHead (disc) → P_d  (NOT band-anchored)
        ├── CupGateHead              → L_c^app
        ├── ShapePriorBranch         → S_c, γ_c, distributions
        └── ConfidenceGatedFusion    → fused L_c → Q → P_c = P_d · Q
    PolarToCartesianWarper → Cartesian masks
    GeometryExtractor → radii from dense masks

Key properties:
    - Disc head: SimpleMonotoneHead (NOT band-anchored — that's B5)
    - Cup head: CupGateHead with shape prior fusion (from V3.1 §6)
    - ShapePriorBranch generates boundary distributions and confidence maps
    - ConfidenceGatedFusion injects shape prior into cup gate logits
    - Nesting: P_c = P_d · Q (by construction)

This is essentially NPS-Net V3.0 (before V3.1's band-anchored disc head).

Changes from B3:
    - Add ShapePriorBranch (§5)
    - Add ConfidenceGatedFusion (§6)
    - Full staged training (Stage A → B → C) with all 7 loss terms

Changes from B5 (full V3.1):
    - Disc head is SimpleMonotoneHead, NOT BandAnchoredDiscHead
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    IMAGE_SIZE, N_THETA, N_RHO, TAU, TAU_SHAPE,
    ENCODER_FEATURES, SHAPE_FEATURES,
    SOFTARGMAX_TEMPERATURE, PRIOR_SCALE_INIT,
)

# Import shared components from B2/B3
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

from model_b3 import CupGateHead


# ==============================================================================
# CIRCULAR 1D CONV (needed by ShapePriorBranch)
# ==============================================================================

class Circular1DConv(nn.Module):
    """Conv1d with circular padding along the angular dimension."""

    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True):
        super().__init__()
        self.pad = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=0, bias=bias)

    def forward(self, x):
        x = F.pad(x, (self.pad, self.pad), mode='circular')
        return self.conv(x)


# ==============================================================================
# §5 — SHAPE PRIOR BRANCH (from ThreeSixty/model.py, unchanged)
# ==============================================================================

class ShapePriorBranch(nn.Module):
    """Predict structured boundary distributions and nested shape radii
    from angular-compressed features.

    Pipeline:
        1. Pool features along ρ → angular descriptors g(θ)
        2. Circular 1D conv stack → rich angular representation H(θ)
        3. Disc boundary distribution π_d(ρ|θ) → softargmax → r_d^s(θ)
        4. Alpha distribution π_α(ρ|θ) → softargmax → α^s(θ)
        5. r_c^s(θ) = α^s(θ) · r_d^s(θ)
        6. Render soft prior masks S_d, S_c
        7. Compute entropy-based confidence γ_d, γ_c
    """

    def __init__(self, in_channels, n_rho=N_RHO,
                 shape_features=SHAPE_FEATURES,
                 temperature=SOFTARGMAX_TEMPERATURE,
                 tau_shape=TAU_SHAPE):
        super().__init__()
        self.n_rho = n_rho
        self.temperature = temperature
        self.tau_shape = tau_shape

        rho = torch.linspace(0, 1, n_rho)
        self.register_buffer('rho', rho)

        # Angular feature compression + processing
        self.angular_net = nn.Sequential(
            Circular1DConv(in_channels, shape_features, 5),
            nn.BatchNorm1d(shape_features),
            nn.ReLU(inplace=True),
            Circular1DConv(shape_features, shape_features, 5),
            nn.BatchNorm1d(shape_features),
            nn.ReLU(inplace=True),
            Circular1DConv(shape_features, shape_features, 3),
            nn.BatchNorm1d(shape_features),
            nn.ReLU(inplace=True),
        )

        # Distribution heads
        self.disc_dist_head  = nn.Conv1d(shape_features, n_rho, 1)
        self.alpha_dist_head = nn.Conv1d(shape_features, n_rho, 1)

    def forward(self, features):
        """
        Args:
            features: (B, C, N_ρ, N_θ)

        Returns:
            dict with:
                r_d_s, r_c_s, alpha_s, p_d_s, p_alpha_s,
                S_d, S_c, gamma_d, gamma_c
        """
        # §5.1 — Angular compression
        g = features.mean(dim=2)                # (B, C, N_θ)
        h = self.angular_net(g)                 # (B, C_s, N_θ)

        # §5.2 — Disc boundary distribution
        z_d = self.disc_dist_head(h)            # (B, N_ρ, N_θ)
        p_d = F.softmax(z_d / self.temperature, dim=1)

        rho = self.rho.view(1, -1, 1)
        r_d_s = (rho * p_d).sum(dim=1)         # (B, N_θ)

        # §5.3 — Alpha distribution
        z_alpha = self.alpha_dist_head(h)
        p_alpha = F.softmax(z_alpha / self.temperature, dim=1)
        alpha_s = (rho * p_alpha).sum(dim=1)
        r_c_s = alpha_s * r_d_s

        # §5.4 — Soft shape prior masks
        r_d_exp = r_d_s.unsqueeze(1)
        r_c_exp = r_c_s.unsqueeze(1)
        rho_2d = self.rho.view(1, -1, 1)

        S_d = torch.sigmoid((r_d_exp - rho_2d) / self.tau_shape)
        S_c = torch.sigmoid((r_c_exp - rho_2d) / self.tau_shape)

        # §6.1 — Entropy-based confidence
        eps = 1e-8
        log_n_rho = math.log(self.n_rho)
        H_d = -(p_d * torch.log(p_d + eps)).sum(dim=1) / log_n_rho
        H_alpha = -(p_alpha * torch.log(p_alpha + eps)).sum(dim=1) / log_n_rho

        gamma_d = (1.0 - H_d).unsqueeze(1)
        gamma_c = (1.0 - H_alpha).unsqueeze(1)

        return {
            'r_d_s':     r_d_s,
            'r_c_s':     r_c_s,
            'alpha_s':   alpha_s,
            'p_d_s':     p_d,
            'p_alpha_s': p_alpha,
            'S_d':       S_d.unsqueeze(1),
            'S_c':       S_c.unsqueeze(1),
            'gamma_d':   gamma_d,
            'gamma_c':   gamma_c,
        }


# ==============================================================================
# §6 — CONFIDENCE-GATED FUSION (from ThreeSixty/model.py, cup-only)
# ==============================================================================

class ConfidenceGatedFusion(nn.Module):
    """Fuse cup-gate logits with shape-prior logits via entropy-based
    confidence gates.  Only fuses the cup gate:

        L_c = L_c^app + λ_c · Γ_c · logit(S_c)

    Final masks:
        Q = σ(L_c)
        P_c = P_d · Q   (nested by construction)
    """

    def __init__(self, prior_scale_init=PRIOR_SCALE_INIT):
        super().__init__()
        self.lambda_c = nn.Parameter(torch.tensor(prior_scale_init))

    def forward(self, P_d, L_c_app, S_c, gamma_c):
        """
        Args:
            P_d:     (B, 1, N_ρ, N_θ) disc occupancy
            L_c_app: (B, 1, N_ρ, N_θ) dense cup-gate logits (monotone)
            S_c:     (B, 1, N_ρ, N_θ) soft shape-prior cup mask
            gamma_c: (B, 1, N_θ) cup confidence

        Returns:
            P_d: (B, 1, N_ρ, N_θ) disc occupancy (pass-through)
            P_c: (B, 1, N_ρ, N_θ) final cup occupancy (nested)
            L_c: (B, 1, N_ρ, N_θ) fused cup-gate logits
        """
        eps = 1e-6
        S_c_clamped = S_c.clamp(eps, 1.0 - eps)
        B_c = torch.log(S_c_clamped / (1.0 - S_c_clamped))

        Gamma_c = gamma_c.unsqueeze(2)

        L_c = L_c_app + self.lambda_c * Gamma_c * B_c

        Q = torch.sigmoid(L_c)
        P_c = P_d * Q

        return P_d, P_c, L_c


# ==============================================================================
# B4 MODEL — + SHAPE PRIOR & CONFIDENCE GATING
# ==============================================================================

class AblationB4(nn.Module):
    """Ablation B4: + Shape Prior & Confidence Gating.

    Architecture:
        PolarSamplingGrid → PolarEncoderDecoder
            ├── SimpleMonotoneHead (disc) → P_d  (NOT band-anchored)
            ├── CupGateHead              → L_c^app
            ├── ShapePriorBranch         → S_c, γ_c
            └── ConfidenceGatedFusion    → P_c = P_d · σ(L_c_fused)
        PolarToCartesianWarper → Ŷ_d, Ŷ_c
        GeometryExtractor → r_d^m, r_c^m

    Output keys (full set for staged loss):
        'P_d_polar':   (B, 1, N_ρ, N_θ)
        'P_c_polar':   (B, 1, N_ρ, N_θ)
        'Y_d_cart':    (B, 1, H, W)
        'Y_c_cart':    (B, 1, H, W)
        'r_d_m':       (B, N_θ)
        'r_c_m':       (B, N_θ)
        'r_d_s':       (B, N_θ)
        'r_c_s':       (B, N_θ)
        'alpha_s':     (B, N_θ)
        'p_d_s':       (B, N_ρ, N_θ)
        'p_alpha_s':   (B, N_ρ, N_θ)
        'gamma_d':     (B, 1, N_θ)
        'gamma_c':     (B, 1, N_θ)
        'lambda_c':    scalar
    """

    def __init__(self, image_size=IMAGE_SIZE, n_theta=N_THETA, n_rho=N_RHO,
                 features=None, shape_features=SHAPE_FEATURES,
                 temperature=SOFTARGMAX_TEMPERATURE,
                 prior_scale_init=PRIOR_SCALE_INIT):
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

        # Disc head: simple monotone (NOT band-anchored — that's B5)
        self.disc_head = SimpleMonotoneHead(in_channels=features[0])

        # Cup gate head
        self.cup_head = CupGateHead(in_channels=features[0])

        # §5 — Shape prior branch
        self.shape_prior = ShapePriorBranch(
            in_channels=features[0], n_rho=n_rho,
            shape_features=shape_features, temperature=temperature,
        )

        # §6 — Confidence-gated fusion (cup only)
        self.fusion = ConfidenceGatedFusion(prior_scale_init=prior_scale_init)

        # §7 — Warper
        self.warper = PolarToCartesianWarper(image_size, n_theta, n_rho)

        # §8 — Geometry extractor
        self.geom = GeometryExtractor(n_rho=n_rho)

        self._print_param_table()

    def _print_param_table(self):
        def count(m):
            return sum(p.numel() for p in m.parameters())

        mods = {
            'Polar Sampling Grid':       self.polar_grid,
            'Polar Encoder-Decoder':     self.polar_unet,
            'Disc Monotone Head':        self.disc_head,
            'Cup Gate Head':             self.cup_head,
            'Shape Prior Branch':        self.shape_prior,
            'Confidence-Gated Fusion':   self.fusion,
            'Polar→Cartesian Warper':    self.warper,
            'Geometry Extractor':        self.geom,
        }
        tot = sum(count(m) for m in mods.values())

        print("\n" + "-" * 62)
        print(f"{'Ablation B4 Component':<38} {'Params':>10}  {'%':>6}")
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

        # Disc: simple monotone
        P_d = self.disc_head(features)           # (B, 1, N_ρ, N_θ)

        # Cup gate logits
        L_c_app = self.cup_head(features)        # (B, 1, N_ρ, N_θ)

        # §5 — Shape prior
        shape_dict = self.shape_prior(features)

        # §6 — Confidence-gated fusion (cup only)
        P_d, P_c, L_c_fused = self.fusion(
            P_d, L_c_app,
            shape_dict['S_c'],
            shape_dict['gamma_c'],
        )

        # §7 — Polar → Cartesian
        Y_d_cart = self.warper(P_d)
        Y_c_cart = self.warper(P_c)

        # §8 — Extract geometry
        r_d_m = self.geom(P_d)
        r_c_m = self.geom(P_c)

        return {
            # Final dense polar masks
            'P_d_polar':  P_d,
            'P_c_polar':  P_c,
            # Final Cartesian masks
            'Y_d_cart':   Y_d_cart,
            'Y_c_cart':   Y_c_cart,
            # Dense-mask radii
            'r_d_m':      r_d_m,
            'r_c_m':      r_c_m,
            # Shape prior outputs (for staged loss)
            'r_d_s':      shape_dict['r_d_s'],
            'r_c_s':      shape_dict['r_c_s'],
            'alpha_s':    shape_dict['alpha_s'],
            'p_d_s':      shape_dict['p_d_s'],
            'p_alpha_s':  shape_dict['p_alpha_s'],
            'gamma_d':    shape_dict['gamma_d'],
            'gamma_c':    shape_dict['gamma_c'],
            # Learnable fusion scale (for logging)
            'lambda_c':   self.fusion.lambda_c,
        }
