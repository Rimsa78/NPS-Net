# beal_loss.py
"""
BEAL Loss — Segmentation + Boundary + Adversarial.

L_BEAL = λ_seg × L_seg + λ_bnd × L_bnd + λ_adv × L_adv

Where:
    L_seg = λ_cup × (BCE + Dice)(cup) + λ_disc × (BCE + Dice)(disc)
    L_bnd = BCE(pred_boundary, gt_boundary)  for cup + disc edges
    L_adv = -log(D(G(x)))  (generator adversarial loss)

The discriminator has its own separate loss:
    L_D = 0.5 × [BCE(D(GT), 1) + BCE(D(pred.detach()), 0)]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import LAMBDA_CUP, LAMBDA_DISC, LAMBDA_BND, LAMBDA_ADV


# ==============================================================================
# Helpers
# ==============================================================================

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        p_flat = p.reshape(-1)
        t_flat = targets.reshape(-1)
        inter = (p_flat * t_flat).sum()
        return 1.0 - (2.0 * inter + self.smooth) / (
            p_flat.sum() + t_flat.sum() + self.smooth)


def extract_boundary(mask, kernel_size=3):
    """Extract boundary from a binary mask using morphological erosion.

    Args:
        mask: (B, 1, H, W) binary float tensor
        kernel_size: erosion kernel size

    Returns:
        (B, 1, H, W) boundary map (binary float)
    """
    pad = kernel_size // 2
    # Max-pool to dilate, min-pool (neg-max-neg) to erode
    eroded = -F.max_pool2d(-mask, kernel_size, stride=1, padding=pad)
    boundary = mask - eroded
    return boundary.clamp(0, 1)


# ==============================================================================
# Generator Loss
# ==============================================================================

class BEALLoss(nn.Module):
    """BEAL generator loss: segmentation + boundary + adversarial.

    Expects the model to return a dict from forward_with_boundary():
        'seg_logits': (B, 2, H, W)
        'boundary_logits': (B, 2, H, W)
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, model_output, cup_mask, disc_mask, disc_score=None):
        """
        Args:
            model_output: dict with 'seg_logits' and 'boundary_logits'
            cup_mask:  (B, 1, H, W) binary
            disc_mask: (B, 1, H, W) binary
            disc_score: (B, 1) discriminator output on predicted masks (optional)

        Returns:
            total_loss, loss_dict
        """
        seg_logits = model_output['seg_logits']
        bnd_logits = model_output['boundary_logits']

        cup_logits  = seg_logits[:, 0:1]
        disc_logits = seg_logits[:, 1:2]

        # ── Segmentation loss (same as BaselineLoss) ──────────────────────
        l_cup_bce  = self.bce(cup_logits, cup_mask)
        l_cup_dice = self.dice(cup_logits, cup_mask)
        l_cup = l_cup_bce + l_cup_dice

        l_disc_bce  = self.bce(disc_logits, disc_mask)
        l_disc_dice = self.dice(disc_logits, disc_mask)
        l_disc = l_disc_bce + l_disc_dice

        l_seg = LAMBDA_CUP * l_cup + LAMBDA_DISC * l_disc

        # ── Boundary loss ─────────────────────────────────────────────────
        gt_cup_bnd  = extract_boundary(cup_mask)
        gt_disc_bnd = extract_boundary(disc_mask)

        bnd_cup_logits  = bnd_logits[:, 0:1]
        bnd_disc_logits = bnd_logits[:, 1:2]

        l_bnd_cup  = self.bce(bnd_cup_logits, gt_cup_bnd)
        l_bnd_disc = self.bce(bnd_disc_logits, gt_disc_bnd)
        l_bnd = l_bnd_cup + l_bnd_disc

        # ── Adversarial loss (generator wants discriminator to say "real") ─
        l_adv = torch.tensor(0.0, device=seg_logits.device)
        if disc_score is not None:
            # Generator wants disc_score → 1 (real)
            target_real = torch.ones_like(disc_score)
            l_adv = F.binary_cross_entropy_with_logits(disc_score, target_real)

        # ── Total ─────────────────────────────────────────────────────────
        total = l_seg + LAMBDA_BND * l_bnd + LAMBDA_ADV * l_adv

        loss_dict = {
            'total':     total.item(),
            'seg':       l_seg.item(),
            'cup_bce':   l_cup_bce.item(),
            'cup_dice':  l_cup_dice.item(),
            'disc_bce':  l_disc_bce.item(),
            'disc_dice': l_disc_dice.item(),
            'bnd':       l_bnd.item(),
            'adv':       l_adv.item(),
        }

        return total, loss_dict


# ==============================================================================
# Discriminator Loss
# ==============================================================================

class DiscriminatorLoss(nn.Module):
    """Discriminator loss: distinguish GT masks from predicted masks.

    L_D = 0.5 × [BCE(D(GT), 1) + BCE(D(pred), 0)]
    """

    def forward(self, d_real, d_fake):
        """
        Args:
            d_real: (B, 1) discriminator output on GT masks
            d_fake: (B, 1) discriminator output on predicted masks (detached)
        Returns:
            loss scalar
        """
        real_label = torch.ones_like(d_real)
        fake_label = torch.zeros_like(d_fake)
        loss_real = F.binary_cross_entropy_with_logits(d_real, real_label)
        loss_fake = F.binary_cross_entropy_with_logits(d_fake, fake_label)
        return 0.5 * (loss_real + loss_fake)
