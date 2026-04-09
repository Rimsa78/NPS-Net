# losses_ablation.py
#!/usr/bin/env python3
"""
Ablation Study — Loss Functions.

Two loss modes:
    1. AblationLossStageA  — for B2, B3: only L_cart + L_polar + L_rim
       Runs flat for all 80 epochs, no staged scheduling.

     2. AblationLossStaged  — for B4: full 7-term staged loss
        Stage A (ep 0–19):  L_cart + L_polar + L_rim
        Stage B (ep 20–29): + L_dist + L_shape + L_smooth
        Stage C (ep 30–79): + L_cons

 Loss terms:
     (1) L_cart    — Cartesian mask: BCE + Dice
     (2) L_polar   — Polar mask: BCE + Dice
     (3) L_rim     — Rim-profile on dense-mask radii
     (4) L_dist    — Shape distribution: soft cross-entropy    [B4 Stage B+]
     (5) L_shape   — Shape radial regression: SmoothL1         [B4 Stage B+]
     (6) L_cons    — Confidence-weighted consistency            [B4 Stage C+]
     (7) L_smooth  — Circular smoothness on shape prior         [B4 Stage B+]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    LAMBDA_CART,
    LAMBDA_POLAR,
    LAMBDA_RIM,
    LAMBDA_DIST,
    LAMBDA_SHAPE,
    LAMBDA_CONS,
    LAMBDA_SMOOTH,
    SMOOTHL1_DELTA,
    STAGE_A_END,
    STAGE_B_END,
)


# ==============================================================================
# BUILDING BLOCKS
# ==============================================================================


def soft_dice(pred, target, smooth=1.0):
    """Soft Dice coefficient.  pred and target in [0,1]."""
    p_flat = pred.reshape(-1)
    t_flat = target.reshape(-1)
    inter = (p_flat * t_flat).sum()
    return (2.0 * inter + smooth) / (p_flat.sum() + t_flat.sum() + smooth)


def seg_loss(pred, target, smooth=1.0):
    """L_seg(Ŷ, Y) = BCE(Ŷ, Y) + (1 − Dice(Ŷ, Y))."""
    pred = pred.float().clamp(1e-7, 1.0 - 1e-7)
    target = target.float()
    l_bce = F.binary_cross_entropy(pred, target, reduction="mean")
    l_dice = 1.0 - soft_dice(pred, target, smooth)
    return l_bce + l_dice


# ==============================================================================
# (1) CARTESIAN MASK LOSS
# ==============================================================================


class CartesianMaskLoss(nn.Module):
    def forward(self, Y_c_pred, Y_d_pred, Y_c_gt, Y_d_gt):
        with torch.amp.autocast("cuda", enabled=False):
            l_cup = seg_loss(Y_c_pred.float(), Y_c_gt.float())
            l_disc = seg_loss(Y_d_pred.float(), Y_d_gt.float())
        return l_cup + l_disc


# ==============================================================================
# (2) POLAR MASK LOSS
# ==============================================================================


class PolarMaskLoss(nn.Module):
    def forward(self, P_c_polar, P_d_polar, Y_c_polar_gt, Y_d_polar_gt):
        with torch.amp.autocast("cuda", enabled=False):
            l_cup = seg_loss(P_c_polar.squeeze(1).float(), Y_c_polar_gt.float())
            l_disc = seg_loss(P_d_polar.squeeze(1).float(), Y_d_polar_gt.float())
        return l_cup + l_disc


# ==============================================================================
# (3) RIM-PROFILE LOSS
# ==============================================================================


class RimProfileLoss(nn.Module):
    def __init__(self, delta=SMOOTHL1_DELTA):
        super().__init__()
        self.delta = delta

    def forward(self, r_c_m, r_d_m, r_c_gt, r_d_gt):
        rim_pred = r_d_m - r_c_m
        rim_gt = r_d_gt - r_c_gt
        return F.smooth_l1_loss(rim_pred, rim_gt, beta=self.delta)


# ==============================================================================
# (4) SHAPE DISTRIBUTION LOSS
# ==============================================================================


class ShapeDistributionLoss(nn.Module):
    def forward(self, p_d_s, p_alpha_s, q_d, q_alpha):
        eps = 1e-8
        l_d = -(q_d * torch.log(p_d_s + eps)).sum(dim=1).mean()
        l_alpha = -(q_alpha * torch.log(p_alpha_s + eps)).sum(dim=1).mean()
        return l_d + l_alpha


# ==============================================================================
# (5) SHAPE RADIAL REGRESSION LOSS
# ==============================================================================


class ShapeRadialLoss(nn.Module):
    def __init__(self, delta=SMOOTHL1_DELTA):
        super().__init__()
        self.delta = delta

    def forward(self, r_d_s, r_c_s, r_d_gt, r_c_gt):
        l_d = F.smooth_l1_loss(r_d_s, r_d_gt, beta=self.delta)
        l_c = F.smooth_l1_loss(r_c_s, r_c_gt, beta=self.delta)
        return l_d + l_c


# ==============================================================================
# (6) CONSISTENCY LOSS
# ==============================================================================


class ConsistencyLoss(nn.Module):
    def forward(self, r_d_m, r_c_m, r_d_s, r_c_s, gamma_d, gamma_c):
        gd = gamma_d.squeeze(1)
        gc = gamma_c.squeeze(1)
        l_d = (gd * (r_d_m - r_d_s).abs()).mean()
        l_c = (gc * (r_c_m - r_c_s).abs()).mean()
        return l_d + l_c


# ==============================================================================
# (7) SHAPE SMOOTHNESS LOSS
# ==============================================================================


class ShapeSmoothnessLoss(nn.Module):
    def __init__(self, delta=SMOOTHL1_DELTA):
        super().__init__()
        self.delta = delta

    def _second_diff(self, r):
        r_next = torch.roll(r, -1, dims=1)
        r_prev = torch.roll(r, 1, dims=1)
        return r_next - 2.0 * r + r_prev

    def forward(self, r_d_s, r_c_s):
        d2_d = self._second_diff(r_d_s)
        d2_c = self._second_diff(r_c_s)
        zero = torch.zeros_like(d2_d)
        l_d = F.smooth_l1_loss(d2_d, zero, beta=self.delta)
        l_c = F.smooth_l1_loss(d2_c, zero, beta=self.delta)
        return l_d + l_c


# ==============================================================================
# LOSS FOR B2 AND B3: STAGE A ONLY (flat, all 80 epochs)
# ==============================================================================


class AblationLossStageA(nn.Module):
    """Loss for B2 and B3 ablation variants.

    Only the three always-active losses:
        L = λ_cart · L_cart + λ_polar · L_polar + λ_rim · L_rim

    No shape prior, no distribution, no consistency, no smoothness.
    Runs for all 80 epochs with no stage transitions.
    """

    def __init__(self):
        super().__init__()
        self.cart_loss = CartesianMaskLoss()
        self.polar_loss = PolarMaskLoss()
        self.rim_loss = RimProfileLoss()

    def forward(self, outputs, batch, epoch=0):
        """
        Args:
            outputs: dict from AblationB2 or AblationB3 forward()
            batch:   dict from dataset
            epoch:   current epoch (unused — no staging)

        Returns:
            total_loss: scalar
            loss_dict:  dict of individual values for logging
        """
        l_cart = self.cart_loss(
            outputs["Y_c_cart"],
            outputs["Y_d_cart"],
            batch["cup_mask"],
            batch["disc_mask"],
        )

        l_polar = self.polar_loss(
            outputs["P_c_polar"],
            outputs["P_d_polar"],
            batch["Y_c_polar_gt"],
            batch["Y_d_polar_gt"],
        )

        l_rim = self.rim_loss(
            outputs["r_c_m"],
            outputs["r_d_m"],
            batch["r_c_gt"],
            batch["r_d_gt"],
        )

        total = LAMBDA_CART * l_cart + LAMBDA_POLAR * l_polar + LAMBDA_RIM * l_rim

        loss_dict = {
            "cart": l_cart.item(),
            "polar": l_polar.item(),
            "rim": l_rim.item(),
            "dist": 0.0,
            "shape": 0.0,
            "cons": 0.0,
            "smooth": 0.0,
            "total": total.item(),
        }

        return total, loss_dict


# ==============================================================================
# LOSS FOR B4: FULL STAGED TRAINING (identical to NPSCombinedLoss)
# ==============================================================================


class AblationLossStaged(nn.Module):
    """Full staged loss for B4 ablation variant.

    Stage A (ep 0–19):  L_cart + L_polar + L_rim
    Stage B (ep 20–29): + L_dist + L_shape + L_smooth
    Stage C (ep 30–79): + L_cons
    """

    def __init__(self):
        super().__init__()
        self.cart_loss = CartesianMaskLoss()
        self.polar_loss = PolarMaskLoss()
        self.rim_loss = RimProfileLoss()
        self.dist_loss = ShapeDistributionLoss()
        self.shape_loss = ShapeRadialLoss()
        self.cons_loss = ConsistencyLoss()
        self.smooth_loss = ShapeSmoothnessLoss()

    def forward(self, outputs, batch, epoch=0):
        # (1) Cartesian mask loss — always active
        l_cart = self.cart_loss(
            outputs["Y_c_cart"],
            outputs["Y_d_cart"],
            batch["cup_mask"],
            batch["disc_mask"],
        )

        # (2) Polar mask loss — always active
        l_polar = self.polar_loss(
            outputs["P_c_polar"],
            outputs["P_d_polar"],
            batch["Y_c_polar_gt"],
            batch["Y_d_polar_gt"],
        )

        # (3) Rim-profile loss — always active
        l_rim = self.rim_loss(
            outputs["r_c_m"],
            outputs["r_d_m"],
            batch["r_c_gt"],
            batch["r_d_gt"],
        )

        # Stage A total
        total = LAMBDA_CART * l_cart + LAMBDA_POLAR * l_polar + LAMBDA_RIM * l_rim

        loss_dict = {
            "cart": l_cart.item(),
            "polar": l_polar.item(),
            "rim": l_rim.item(),
        }

        # (4,5,7) Shape supervision — Stage B onward
        if epoch >= STAGE_A_END:
            l_dist = self.dist_loss(
                outputs["p_d_s"],
                outputs["p_alpha_s"],
                batch["q_d"],
                batch["q_alpha"],
            )
            l_shape = self.shape_loss(
                outputs["r_d_s"],
                outputs["r_c_s"],
                batch["r_d_gt"],
                batch["r_c_gt"],
            )
            l_smooth = self.smooth_loss(
                outputs["r_d_s"],
                outputs["r_c_s"],
            )

            total = total + (
                LAMBDA_DIST * l_dist + LAMBDA_SHAPE * l_shape + LAMBDA_SMOOTH * l_smooth
            )

            loss_dict["dist"] = l_dist.item()
            loss_dict["shape"] = l_shape.item()
            loss_dict["smooth"] = l_smooth.item()
        else:
            loss_dict["dist"] = 0.0
            loss_dict["shape"] = 0.0
            loss_dict["smooth"] = 0.0

        # (6) Consistency — Stage C onward
        if epoch >= STAGE_B_END:
            l_cons = self.cons_loss(
                outputs["r_d_m"],
                outputs["r_c_m"],
                outputs["r_d_s"],
                outputs["r_c_s"],
                outputs["gamma_d"],
                outputs["gamma_c"],
            )
            total = total + LAMBDA_CONS * l_cons
            loss_dict["cons"] = l_cons.item()
        else:
            loss_dict["cons"] = 0.0

        loss_dict["total"] = total.item()

        return total, loss_dict


# ==============================================================================
# FACTORY — select loss by variant
# ==============================================================================


def get_loss_for_variant(variant):
    """Return the appropriate loss module for a given ablation variant.

    Args:
        variant: 'b2', 'b3', or 'b4'

    Returns:
        nn.Module loss function
    """
    if variant in ("b2", "b3"):
        return AblationLossStageA()
    elif variant == "b4":
        return AblationLossStaged()
    else:
        raise ValueError(f"Unknown variant: {variant}. Expected 'b2', 'b3', or 'b4'.")
