# inference_polar_tta_ablation.py
#!/usr/bin/env python3
"""
Ablation B5 — Polar-Space Test-Time Augmentation (Polar-TTA) on B4 model.

Applies centre/scale marginalisation at inference time to the B4 model
(SimpleMonotoneHead disc + CupGateHead + ShapePriorBranch + ConfidenceGatedFusion).

No additional training required — uses the B4 checkpoint directly.

This script is adapted from ThreeSixty/inference_polar_tta.py to work with
the B4 architecture (AblationB4) rather than the full V3.1 (NPSNet).

Polar-TTA strategy:
    For each image in the batch:
        For each (Δx, Δy, scale) hypothesis in a small search grid:
            1. Re-warp the image to polar using shifted/scaled centre
            2. Run B4 forward pass (shared weights)
            3. Compute self-score (disc mass + boundary sharpness + compactness)
        Aggregate top-K hypotheses in Cartesian space (weighted by scores).

Self-Score (no GT required):
    1. disc_mass:       mean(P_d_polar) — fires confidently on disc
    2. boundary_sharp:  mean(γ_d) — shape branch is sharp when well-localised
    3. disc_compactness: fraction of disc pixels inside inscribed circle

Usage:
    python inference_polar_tta_ablation.py --test
    python inference_polar_tta_ablation.py --test --all-external
    python inference_polar_tta_ablation.py --test --top-k 1
    python inference_polar_tta_ablation.py --test --offsets 0,8,16 --scales 0.85,1.0,1.15
"""

import argparse
import math
import os
import itertools

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from scipy.ndimage import distance_transform_edt
from scipy.stats import pearsonr

from config import (
    DEVICE, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, N_THETA, N_RHO,
    get_checkpoint_dir,
)
from model_b4 import AblationB4
from dataset import get_test_dataloader, get_external_dataloader


# ==============================================================================
# DEFAULT SEARCH GRID
# ==============================================================================

DEFAULT_OFFSETS_PX = [0, 8, 16]          # → 9 (Δx,Δy) pairs after product
DEFAULT_SCALES = [0.85, 1.0, 1.15]       # shrink / nominal / grow

# Self-score weighting
W_MASS      = 0.4
W_SHARPNESS = 0.4
W_COMPACT   = 0.2

DEFAULT_TOP_K = 3


# ==============================================================================
# PER-HYPOTHESIS POLAR GRID
# ==============================================================================

def build_polar_grid(image_size, n_theta, n_rho,
                     cx_offset=0.0, cy_offset=0.0, scale=1.0):
    """Build a (1, N_ρ, N_θ, 2) sampling grid for one hypothesis."""
    H = W = image_size
    cx = (W - 1) / 2.0 + cx_offset
    cy = (H - 1) / 2.0 + cy_offset
    R  = min(H, W) / 2.0 * scale

    theta = torch.linspace(0, 2 * math.pi, n_theta + 1)[:n_theta]
    rho   = torch.linspace(0, 1, n_rho)

    rho_2d   = rho.unsqueeze(1)
    theta_2d = theta.unsqueeze(0)

    px = cx + R * rho_2d * torch.cos(theta_2d)
    py = cy + R * rho_2d * torch.sin(theta_2d)

    grid_x = 2.0 * px / (W - 1) - 1.0
    grid_y = 2.0 * py / (H - 1) - 1.0

    grid = torch.stack([grid_x, grid_y], dim=-1)
    return grid.unsqueeze(0)


def build_cartesian_grid(image_size, n_theta, n_rho,
                         cx_offset=0.0, cy_offset=0.0, scale=1.0):
    """Build inverse (Cartesian → polar) grid for this hypothesis."""
    H = W = image_size
    cx = (W - 1) / 2.0 + cx_offset
    cy = (H - 1) / 2.0 + cy_offset
    R  = min(H, W) / 2.0 * scale

    ys = torch.arange(H, dtype=torch.float32)
    xs = torch.arange(W, dtype=torch.float32)
    grid_y_img, grid_x_img = torch.meshgrid(ys, xs, indexing='ij')

    dx = grid_x_img - cx
    dy = grid_y_img - cy

    rho_cart   = torch.sqrt(dx ** 2 + dy ** 2) / R
    theta_cart = torch.atan2(dy, dx) % (2 * math.pi)

    inside_circle = (rho_cart <= 1.0).float()

    inv_grid_y = 2.0 * rho_cart - 1.0
    inv_grid_x = theta_cart / math.pi - 1.0

    inv_grid = torch.stack([inv_grid_x, inv_grid_y], dim=-1)
    return inv_grid.unsqueeze(0), inside_circle


# ==============================================================================
# HYPOTHESIS GRID CONSTRUCTION
# ==============================================================================

def make_hypotheses(offsets_px, scales):
    """Cartesian product of (Δx, Δy, scale) hypotheses."""
    pairs = list(itertools.product(offsets_px, offsets_px))
    hyps  = [(dx, dy, s) for (dx, dy) in pairs for s in scales]

    canonical = (0.0, 0.0, 1.0)
    hyps = [h for h in hyps if h != canonical]
    hyps = [canonical] + hyps
    return hyps


# ==============================================================================
# SELF-SCORE COMPUTATION
# ==============================================================================

def compute_self_score(out, inside_circle_dev):
    """Compute a no-GT proxy score for hypothesis quality.

    Uses B4's shape-prior confidence (gamma_d) for boundary sharpness.
    """
    # 1. Disc mass in polar space
    P_d_polar = out['P_d_polar']
    disc_mass = P_d_polar.mean(dim=[1, 2, 3])

    # 2. Shape-prior sharpness (mean confidence)
    gamma_d = out['gamma_d']
    sharpness = gamma_d.mean(dim=[1, 2])

    # 3. Cartesian compactness
    Y_d_cart = out['Y_d_cart']
    disc_bin = (Y_d_cart > 0.5).float()
    inside = inside_circle_dev.unsqueeze(0).unsqueeze(0)
    n_disc = disc_bin.sum(dim=[1, 2, 3]).clamp(min=1.0)
    n_inside = (disc_bin * inside).sum(dim=[1, 2, 3])
    compactness = n_inside / n_disc

    score = (W_MASS * disc_mass
             + W_SHARPNESS * sharpness
             + W_COMPACT * compactness)
    return score


# ==============================================================================
# POLAR-TTA FORWARD PASS (single batch)
# ==============================================================================

@torch.no_grad()
def polar_tta_forward(model, images, hypotheses, device, top_k=DEFAULT_TOP_K):
    """Run polar-TTA for one batch using the B4 model.

    Temporarily swaps polar_grid and warper buffers for each hypothesis.
    """
    B = images.size(0)
    H = W = IMAGE_SIZE
    n_h = len(hypotheses)

    # Pre-build all grids
    polar_grids  = []
    cart_grids   = []
    inside_masks = []
    for (dx, dy, sc) in hypotheses:
        pg = build_polar_grid(IMAGE_SIZE, N_THETA, N_RHO,
                              cx_offset=dx, cy_offset=dy, scale=sc)
        ig, ins = build_cartesian_grid(IMAGE_SIZE, N_THETA, N_RHO,
                                       cx_offset=dx, cy_offset=dy, scale=sc)
        polar_grids.append(pg.to(device))
        cart_grids.append(ig.to(device))
        inside_masks.append(ins.to(device))

    # Save canonical buffers
    orig_grid = model.polar_grid.grid.clone()

    all_cup_soft  = []
    all_disc_soft = []
    all_r_c       = []
    all_r_d       = []
    all_scores    = []
    canonical_out = None

    for h_idx, (dx, dy, sc) in enumerate(hypotheses):
        # Swap polar sampling grid
        model.polar_grid.grid = polar_grids[h_idx].expand(B, -1, -1, -1)

        # Swap inverse warper
        orig_inv_grid = model.warper.inv_grid.clone()
        orig_inside = model.warper.inside_circle.clone()
        model.warper.inv_grid = cart_grids[h_idx].expand(B, -1, -1, -1)
        model.warper.inside_circle = inside_masks[h_idx]

        with autocast('cuda'):
            out = model(images)

        # Restore
        model.polar_grid.grid = orig_grid
        model.warper.inv_grid = orig_inv_grid
        model.warper.inside_circle = orig_inside

        score = compute_self_score(out, inside_masks[h_idx])

        all_cup_soft.append(out['Y_c_cart'])
        all_disc_soft.append(out['Y_d_cart'])
        all_r_c.append(out['r_c_m'])
        all_r_d.append(out['r_d_m'])
        all_scores.append(score)

        if h_idx == 0:
            canonical_out = out

    # Restore (safety)
    model.polar_grid.grid = orig_grid

    # Stack scores: (B, n_h)
    scores_stk = torch.stack(all_scores, dim=1)

    best_idx = scores_stk.argmax(dim=1)
    best_hyps = [hypotheses[best_idx[i].item()] for i in range(B)]

    # Top-K blend
    k = min(top_k, n_h)
    topk_vals, topk_idx = scores_stk.topk(k, dim=1)
    weights = torch.softmax(topk_vals, dim=1)

    cup_stk  = torch.stack(all_cup_soft, dim=1)
    disc_stk = torch.stack(all_disc_soft, dim=1)
    r_c_stk  = torch.stack(all_r_c, dim=1)
    r_d_stk  = torch.stack(all_r_d, dim=1)

    idx_exp_cart = topk_idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    idx_exp_cart = idx_exp_cart.expand(-1, -1, 1, H, W)
    idx_exp_r = topk_idx.unsqueeze(-1).expand(-1, -1, N_THETA)

    cup_topk  = cup_stk.gather(1, idx_exp_cart)
    disc_topk = disc_stk.gather(1, idx_exp_cart)
    r_c_topk  = r_c_stk.gather(1, idx_exp_r)
    r_d_topk  = r_d_stk.gather(1, idx_exp_r)

    w = weights.view(B, k, 1, 1, 1)
    cup_prob  = (cup_topk * w).sum(dim=1)
    disc_prob = (disc_topk * w).sum(dim=1)

    w_r = weights.view(B, k, 1)
    r_c_avg = (r_c_topk * w_r).sum(dim=1)
    r_d_avg = (r_d_topk * w_r).sum(dim=1)

    return {
        'cup_prob':   cup_prob,
        'disc_prob':  disc_prob,
        'r_c_avg':    r_c_avg,
        'r_d_avg':    r_d_avg,
        'gamma_c':    canonical_out['gamma_c'],
        'gamma_d':    canonical_out['gamma_d'],
        'best_hyps':  best_hyps,
        'all_scores': scores_stk,
    }


# ==============================================================================
# METRICS (identical to inference_ablation.py)
# ==============================================================================

def dice_score(pred, target, eps=1e-7):
    p = pred.reshape(-1).float()
    t = target.reshape(-1).float()
    inter = (p * t).sum()
    return (2.0 * inter + eps) / (p.sum() + t.sum() + eps)


def iou_score(pred, target, eps=1e-7):
    p = pred.reshape(-1).float()
    t = target.reshape(-1).float()
    inter = (p * t).sum()
    union = p.sum() + t.sum() - inter
    return (inter + eps) / (union + eps)


def hausdorff_95(pred_np, target_np):
    if pred_np.sum() == 0 or target_np.sum() == 0:
        return float('nan')
    dt_p = distance_transform_edt(~pred_np.astype(bool))
    dt_t = distance_transform_edt(~target_np.astype(bool))
    d_p2t = dt_t[pred_np.astype(bool)]
    d_t2p = dt_p[target_np.astype(bool)]
    if len(d_p2t) == 0 or len(d_t2p) == 0:
        return float('nan')
    return float(np.percentile(np.concatenate([d_p2t, d_t2p]), 95))


def average_surface_distance(pred_np, target_np):
    if pred_np.sum() == 0 or target_np.sum() == 0:
        return float('nan')
    dt_p = distance_transform_edt(~pred_np.astype(bool))
    dt_t = distance_transform_edt(~target_np.astype(bool))
    d_p2t = dt_t[pred_np.astype(bool)]
    d_t2p = dt_p[target_np.astype(bool)]
    if len(d_p2t) == 0 or len(d_t2p) == 0:
        return float('nan')
    return float((d_p2t.mean() + d_t2p.mean()) / 2.0)


def compute_vcdr(cup_np, disc_np):
    cup_rows = np.where(cup_np)[0]
    disc_rows = np.where(disc_np)[0]
    if len(disc_rows) == 0:
        return 0.0
    cup_h = (cup_rows.max() - cup_rows.min() + 1) if len(cup_rows) > 0 else 0
    disc_h = disc_rows.max() - disc_rows.min() + 1
    return cup_h / disc_h


def compute_acdr(cup_np, disc_np):
    disc_area = disc_np.sum()
    return (cup_np.sum() / disc_area) if disc_area > 0 else 0.0


def compute_4sector_rim(r_c, r_d, n_theta):
    rim = (r_d - r_c)
    if isinstance(rim, torch.Tensor):
        rim = rim.cpu().numpy()
    apq = n_theta // 4
    sectors = {
        'temporal': np.concatenate([np.arange(0, apq // 2),
                                    np.arange(n_theta - apq // 2, n_theta)]),
        'superior': np.arange(apq // 2, apq // 2 + apq),
        'nasal':    np.arange(apq // 2 + apq, apq // 2 + 2 * apq),
        'inferior': np.arange(apq // 2 + 2 * apq, apq // 2 + 3 * apq),
    }
    result = {}
    for name, idx in sectors.items():
        idx = idx[idx < n_theta]
        result[name] = float(rim[idx].mean()) if len(idx) > 0 else 0.0
    return result


def nesting_violation_check(cup_np, disc_np):
    viol = (cup_np > 0.5) & (disc_np < 0.5)
    return viol.sum() > 0, float(viol.sum()) / cup_np.size


# ==============================================================================
# EVALUATION ENGINE
# ==============================================================================

@torch.no_grad()
def evaluate_polar_tta(model, loader, device, dataset_name="Test",
                       hypotheses=None, top_k=DEFAULT_TOP_K):
    model.eval()
    if hypotheses is None:
        hypotheses = make_hypotheses(DEFAULT_OFFSETS_PX, DEFAULT_SCALES)

    n_h = len(hypotheses)
    print(f"  [Polar-TTA B5] {n_h} hypotheses, top-K={top_k}")

    all_cup_dice, all_disc_dice   = [], []
    all_cup_iou,  all_disc_iou    = [], []
    all_cup_hd95, all_disc_hd95   = [], []
    all_cup_assd, all_disc_assd   = [], []
    all_vcdr_pred, all_vcdr_gt    = [], []
    all_acdr_pred, all_acdr_gt    = [], []
    all_r_c_mae,   all_r_d_mae    = [], []
    all_rim_mae,   all_4sector_mae = [], []
    all_nesting_viol, all_viol_frac = [], []
    all_rim_corr  = []
    best_hyp_counter = {}

    for batch in loader:
        images    = batch['image'].to(device)
        cup_mask  = batch['cup_mask'].to(device)
        disc_mask = batch['disc_mask'].to(device)
        gt_cdr    = batch['cdr']
        r_c_gt    = batch['r_c_gt'].to(device)
        r_d_gt    = batch['r_d_gt'].to(device)

        tta_out = polar_tta_forward(model, images, hypotheses, device, top_k)

        cup_pred  = (tta_out['cup_prob']  > 0.5).float()
        disc_pred = (tta_out['disc_prob'] > 0.5).float()

        for hyp in tta_out['best_hyps']:
            key = f"Δx={hyp[0]:+.0f} Δy={hyp[1]:+.0f} s={hyp[2]:.2f}"
            best_hyp_counter[key] = best_hyp_counter.get(key, 0) + 1

        B = images.size(0)
        for i in range(B):
            cp = cup_pred[i, 0]
            dp = disc_pred[i, 0]
            cg = cup_mask[i, 0]
            dg = disc_mask[i, 0]

            all_cup_dice.append(dice_score(cp, cg).item())
            all_disc_dice.append(dice_score(dp, dg).item())
            all_cup_iou.append(iou_score(cp, cg).item())
            all_disc_iou.append(iou_score(dp, dg).item())

            cp_np = cp.cpu().numpy().astype(np.uint8)
            dp_np = dp.cpu().numpy().astype(np.uint8)
            cg_np = cg.cpu().numpy().astype(np.uint8)
            dg_np = dg.cpu().numpy().astype(np.uint8)

            all_cup_hd95.append(hausdorff_95(cp_np, cg_np))
            all_disc_hd95.append(hausdorff_95(dp_np, dg_np))
            all_cup_assd.append(average_surface_distance(cp_np, cg_np))
            all_disc_assd.append(average_surface_distance(dp_np, dg_np))

            all_vcdr_pred.append(compute_vcdr(cp_np, dp_np))
            all_vcdr_gt.append(gt_cdr[i].item())
            all_acdr_pred.append(compute_acdr(cp_np, dp_np))
            all_acdr_gt.append(compute_acdr(cg_np, dg_np))

            rc_pred = tta_out['r_c_avg'][i]
            rd_pred = tta_out['r_d_avg'][i]
            rc_gt_i = r_c_gt[i]
            rd_gt_i = r_d_gt[i]

            all_r_c_mae.append((rc_pred - rc_gt_i).abs().mean().item())
            all_r_d_mae.append((rd_pred - rd_gt_i).abs().mean().item())

            rim_pred = rd_pred - rc_pred
            rim_gt   = rd_gt_i - rc_gt_i
            all_rim_mae.append((rim_pred - rim_gt).abs().mean().item())

            rp_np = rim_pred.cpu().numpy()
            rg_np = rim_gt.cpu().numpy()
            if rg_np.std() > 1e-6 and rp_np.std() > 1e-6:
                corr, _ = pearsonr(rp_np, rg_np)
                all_rim_corr.append(corr)

            sectors_pred = compute_4sector_rim(rc_pred, rd_pred, N_THETA)
            sectors_gt   = compute_4sector_rim(rc_gt_i, rd_gt_i, N_THETA)
            all_4sector_mae.append(np.mean([abs(sectors_pred[s] - sectors_gt[s])
                                            for s in sectors_pred]))

            has_viol, viol_frac = nesting_violation_check(cp_np, dp_np)
            all_nesting_viol.append(has_viol)
            all_viol_frac.append(viol_frac)

    def safe_mean(lst):
        valid = [x for x in lst if not (isinstance(x, float) and math.isnan(x))]
        return np.mean(valid) if valid else float('nan')

    results = {
        'dataset':          dataset_name,
        'n':                len(all_cup_dice),
        'n_hypotheses':     n_h,
        'top_k':            top_k,
        'cup_dice':         np.mean(all_cup_dice),
        'disc_dice':        np.mean(all_disc_dice),
        'cup_iou':          np.mean(all_cup_iou),
        'disc_iou':         np.mean(all_disc_iou),
        'cup_hd95':         safe_mean(all_cup_hd95),
        'disc_hd95':        safe_mean(all_disc_hd95),
        'cup_assd':         safe_mean(all_cup_assd),
        'disc_assd':        safe_mean(all_disc_assd),
        'vcdr_mae':         np.mean(np.abs(np.array(all_vcdr_pred) - np.array(all_vcdr_gt))),
        'acdr_mae':         np.mean(np.abs(np.array(all_acdr_pred) - np.array(all_acdr_gt))),
        'r_c_mae':          np.mean(all_r_c_mae),
        'r_d_mae':          np.mean(all_r_d_mae),
        'rim_mae':          np.mean(all_rim_mae),
        'sector_rim_mae':   np.mean(all_4sector_mae),
        'rim_corr':         safe_mean(all_rim_corr),
        'nesting_viol_pct': 100.0 * np.mean(all_nesting_viol),
        'viol_pixel_frac':  np.mean(all_viol_frac),
        'cup_dice_std':     np.std(all_cup_dice),
        'disc_dice_std':    np.std(all_disc_dice),
        'best_hyp_counter': best_hyp_counter,
    }
    return results


# ==============================================================================
# PRINTING
# ==============================================================================

def print_results(r):
    print(f"\n{'='*74}")
    print(f"  {r['dataset']} [B5: Polar-TTA]  "
          f"({r['n']} samples | {r['n_hypotheses']} hyps | top-K={r['top_k']})")
    print(f"{'='*74}")
    print(f"  --- Standard Segmentation ---")
    print(f"  Cup  Dice  : {r['cup_dice']:.4f} ± {r['cup_dice_std']:.4f}")
    print(f"  Disc Dice  : {r['disc_dice']:.4f} ± {r['disc_dice_std']:.4f}")
    print(f"  Cup  IoU   : {r['cup_iou']:.4f}")
    print(f"  Disc IoU   : {r['disc_iou']:.4f}")
    print(f"  Cup  HD95  : {r['cup_hd95']:.2f}")
    print(f"  Disc HD95  : {r['disc_hd95']:.2f}")
    print(f"  Cup  ASSD  : {r['cup_assd']:.2f}")
    print(f"  Disc ASSD  : {r['disc_assd']:.2f}")
    print(f"  --- Geometry Fidelity ---")
    print(f"  vCDR MAE   : {r['vcdr_mae']:.4f}")
    print(f"  aCDR MAE   : {r['acdr_mae']:.4f}")
    print(f"  Cup r MAE  : {r['r_c_mae']:.4f}")
    print(f"  Disc r MAE : {r['r_d_mae']:.4f}")
    print(f"  Rim MAE    : {r['rim_mae']:.4f}")
    print(f"  4-sect MAE : {r['sector_rim_mae']:.4f}")
    print(f"  Rim corr   : {r['rim_corr']:.4f}")
    print(f"  --- Anatomical Validity ---")
    print(f"  Nesting viol%: {r['nesting_viol_pct']:.1f}%")
    print(f"  Viol px frac : {r['viol_pixel_frac']:.6f}")
    print(f"  --- Polar-TTA Diagnostics ---")
    if r['best_hyp_counter']:
        sorted_hyps = sorted(r['best_hyp_counter'].items(), key=lambda x: -x[1])
        total = sum(v for v in r['best_hyp_counter'].values())
        for key, cnt in sorted_hyps[:6]:
            print(f"  {key:<30}  won {cnt:4d} / {total}  ({100*cnt/total:.1f}%)")
    print(f"{'='*74}\n")


# ==============================================================================
# MODEL LOADING
# ==============================================================================

def load_model(checkpoint_path=None):
    """Load B4 model for Polar-TTA inference."""
    model = AblationB4().to(DEVICE)

    if checkpoint_path is None:
        checkpoint_path = os.path.join(get_checkpoint_dir('b4'), 'best_model.pth')

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"B4 checkpoint not found: {checkpoint_path}\n"
                                "Train B4 first with: python train_ablation.py --variant b4")

    state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    if 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
        epoch = state.get('epoch', '?')
        cup_dice = state.get('best_cup_dice', '?')
        print(f"[B5/Polar-TTA] Loaded B4 checkpoint  epoch={epoch}  cup_dice={cup_dice}")
    else:
        model.load_state_dict(state)
        print("[B5/Polar-TTA] Loaded raw B4 state dict")

    model.eval()
    return model


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ablation B5 — Polar-TTA on B4 model")
    parser.add_argument('--test', action='store_true',
                        help="Evaluate on held-out internal test split")
    parser.add_argument('--all-external', action='store_true',
                        help="Evaluate on RIM and Dristi external splits")
    parser.add_argument('--csv', type=str, default=None)
    parser.add_argument('--name', type=str, default='External')
    parser.add_argument('--save-vis', action='store_true')
    parser.add_argument('--max-vis', type=int, default=16)
    parser.add_argument('--output-dir', type=str, default='predictions_polar_tta_b5')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to B4 checkpoint (default: checkpoints/b4/best_model.pth)')
    parser.add_argument('--top-k', type=int, default=DEFAULT_TOP_K)
    parser.add_argument('--offsets', type=str,
                        default=','.join(str(o) for o in DEFAULT_OFFSETS_PX))
    parser.add_argument('--scales', type=str,
                        default=','.join(str(s) for s in DEFAULT_SCALES))
    args = parser.parse_args()

    if not (args.test or args.csv or args.all_external):
        parser.error("Provide at least one of --test, --csv, or --all-external")

    offsets_px = [float(x) for x in args.offsets.split(',')]
    scales     = [float(x) for x in args.scales.split(',')]
    hypotheses = make_hypotheses(offsets_px, scales)

    print(f"\n[B5: Polar-TTA on B4] {len(hypotheses)} hypotheses  "
          f"(offsets={offsets_px}px  scales={scales}  top-K={args.top_k})")

    model = load_model(args.checkpoint)
    all_results = []

    if args.test:
        print("\n>>> B5 Polar-TTA on held-out TEST split ...")
        loader = get_test_dataloader(BATCH_SIZE, NUM_WORKERS)
        r = evaluate_polar_tta(model, loader, DEVICE, "Test Split",
                               hypotheses, args.top_k)
        print_results(r)
        all_results.append(r)

    if args.csv:
        print(f"\n>>> B5 Polar-TTA on {args.csv} ...")
        loader = get_external_dataloader(args.csv, BATCH_SIZE, NUM_WORKERS)
        r = evaluate_polar_tta(model, loader, DEVICE, args.name,
                               hypotheses, args.top_k)
        print_results(r)
        all_results.append(r)

    if args.all_external:
        ext_datasets = [
            ("RIM",    "../../Map/Corrected_testrim.csv"),
            ("Dristi", "../../Map/Corrected_DristiTest.csv"),
        ]
        for name, csv_path in ext_datasets:
            abs_path = os.path.join(os.path.dirname(__file__), csv_path)
            if not os.path.isfile(abs_path):
                print(f"[SKIP] {name}: file not found  {abs_path}")
                continue
            print(f"\n>>> B5 Polar-TTA on {name} ...")
            loader = get_external_dataloader(abs_path, BATCH_SIZE, NUM_WORKERS)
            r = evaluate_polar_tta(model, loader, DEVICE, name,
                                   hypotheses, args.top_k)
            print_results(r)
            all_results.append(r)

    if len(all_results) > 1:
        print("\n" + "=" * 100)
        print(f"  SUMMARY — B5: Polar-TTA on B4  (hyps={len(hypotheses)}  top-K={args.top_k})")
        print("=" * 100)
        header = (f"{'Dataset':<15} {'Cup Dice':>10} {'Disc Dice':>10} "
                  f"{'Cup HD95':>10} {'vCDR MAE':>10} {'Rim MAE':>10} "
                  f"{'Rim Corr':>10} {'Nest%':>8}")
        print(header)
        print("-" * 100)
        for r in all_results:
            print(f"{r['dataset']:<15} {r['cup_dice']:>10.4f} {r['disc_dice']:>10.4f} "
                  f"{r['cup_hd95']:>10.2f} {r['vcdr_mae']:>10.4f} {r['rim_mae']:>10.4f} "
                  f"{r['rim_corr']:>10.4f} {r['nesting_viol_pct']:>7.1f}%")
        print("=" * 100)


if __name__ == '__main__':
    main()
