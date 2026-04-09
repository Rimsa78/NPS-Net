# dofe_loss.py
"""
DoFE Loss — Segmentation + Domain Classification.

L_DoFE = L_seg + λ_dom × L_dom

Where:
    L_seg = λ_cup × (BCE + Dice)(cup) + λ_disc × (BCE + Dice)(disc)
    L_dom = CrossEntropy(domain_pred, pseudo_domain_label)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import LAMBDA_CUP, LAMBDA_DISC, LAMBDA_DOM


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


class DoFELoss(nn.Module):
    """DoFE loss: segmentation + domain classification.

    Expects the model to return a dict from forward_with_domain():
        'seg_logits': (B, 2, H, W)
        'domain_logits': (B, K)
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.domain_ce = nn.CrossEntropyLoss()

    def forward(self, model_output, cup_mask, disc_mask, domain_labels=None):
        """
        Args:
            model_output: dict with 'seg_logits' and 'domain_logits'
            cup_mask:  (B, 1, H, W) binary
            disc_mask: (B, 1, H, W) binary
            domain_labels: (B,) long tensor — pseudo-domain labels (optional)

        Returns:
            total_loss, loss_dict
        """
        seg_logits = model_output['seg_logits']

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

        # ── Domain classification loss ────────────────────────────────────
        l_dom = torch.tensor(0.0, device=seg_logits.device)
        if domain_labels is not None and 'domain_logits' in model_output:
            domain_logits = model_output['domain_logits']
            l_dom = self.domain_ce(domain_logits, domain_labels)

        # ── Total ─────────────────────────────────────────────────────────
        total = l_seg + LAMBDA_DOM * l_dom

        loss_dict = {
            'total':     total.item(),
            'seg':       l_seg.item(),
            'cup_bce':   l_cup_bce.item(),
            'cup_dice':  l_cup_dice.item(),
            'disc_bce':  l_disc_bce.item(),
            'disc_dice': l_disc_dice.item(),
            'dom':       l_dom.item(),
        }

        return total, loss_dict
