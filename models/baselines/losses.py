# losses.py
"""
Standard BCE + Dice loss for baseline comparisons.

L = λ_cup * (BCE + Dice)(cup_logits, cup_mask)
  + λ_disc * (BCE + Dice)(disc_logits, disc_mask)

No special shape losses, no nesting penalties.
All baselines use this same loss for fair comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import LAMBDA_CUP, LAMBDA_DISC


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


class BaselineLoss(nn.Module):
    """Standard BCE + Dice for cup and disc.

    Input: model outputs (B, 2, H, W) logits where channel 0=cup, 1=disc.
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits, cup_mask, disc_mask):
        """
        Args:
            logits:    (B, 2, H, W) — channel 0 = cup, channel 1 = disc
            cup_mask:  (B, 1, H, W) binary
            disc_mask: (B, 1, H, W) binary
        """
        cup_logits  = logits[:, 0:1]   # (B, 1, H, W)
        disc_logits = logits[:, 1:2]   # (B, 1, H, W)

        # Cup loss
        l_cup_bce  = self.bce(cup_logits, cup_mask)
        l_cup_dice = self.dice(cup_logits, cup_mask)
        l_cup = l_cup_bce + l_cup_dice

        # Disc loss
        l_disc_bce  = self.bce(disc_logits, disc_mask)
        l_disc_dice = self.dice(disc_logits, disc_mask)
        l_disc = l_disc_bce + l_disc_dice

        total = LAMBDA_CUP * l_cup + LAMBDA_DISC * l_disc

        loss_dict = {
            'total':     total.item(),
            'cup_bce':   l_cup_bce.item(),
            'cup_dice':  l_cup_dice.item(),
            'disc_bce':  l_disc_bce.item(),
            'disc_dice': l_disc_dice.item(),
        }

        return total, loss_dict
