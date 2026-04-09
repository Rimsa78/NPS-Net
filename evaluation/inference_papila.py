#!/usr/bin/env python3
"""
Papila External-Set Inference — All baselines + NPS-Net.

Runs each model on the cropped Papila dataset and reports standard segmentation
and geometry-fidelity metrics for cross-domain generalisation evaluation.

Baseline models (from Comparision/):
    vanilla, attunet, resunet, polar_unet, transunet, beal, dofe

NPS-Net (from best/):
    NPSNet — confidence-gated monotone factorised polar network

Usage:
    python inference_papila.py --model vanilla
    python inference_papila.py --model npsnet
    python inference_papila.py --model all          # run all models
    python inference_papila.py --model all --save-vis
"""

import argparse
import itertools
import math
import os
import sys

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.amp import autocast
from scipy.ndimage import distance_transform_edt
from scipy.stats import pearsonr

# ── Add parent directory to path so we can import from modules ────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "training"))  # for training modules
sys.path.insert(0, os.path.join(PROJECT_ROOT, "models"))  # for model modules
sys.path.insert(0, os.path.join(PROJECT_ROOT, "datasets"))  # for dataset modules

from training.config import (
    DEVICE,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    N_THETA,
    N_RHO,
    FEATURES,
    CHECKPOINT_DIR,
)
from datasets.dataset import get_external_dataloader

PAPILA_CSV = os.path.join(PROJECT_ROOT, "Map", "Corrected_papila.csv")

BASELINE_MODELS = [
    "vanilla",
    "attunet",
    "resunet",
    "polar_unet",
    "transunet",
    "beal",
    "dofe",
]
ALL_MODELS = BASELINE_MODELS + ["npsnet"]


# ==============================================================================
# METRICS — Standard Segmentation
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
        return float("nan")
    dt_pred = distance_transform_edt(~pred_np.astype(bool))
    dt_target = distance_transform_edt(~target_np.astype(bool))
    d1 = dt_target[pred_np.astype(bool)]
    d2 = dt_pred[target_np.astype(bool)]
    if len(d1) == 0 or len(d2) == 0:
        return float("nan")
    return float(np.percentile(np.concatenate([d1, d2]), 95))


def average_surface_distance(pred_np, target_np):
    if pred_np.sum() == 0 or target_np.sum() == 0:
        return float("nan")
    dt_pred = distance_transform_edt(~pred_np.astype(bool))
    dt_target = distance_transform_edt(~target_np.astype(bool))
    d1 = dt_target[pred_np.astype(bool)]
    d2 = dt_pred[target_np.astype(bool)]
    if len(d1) == 0 or len(d2) == 0:
        return float("nan")
    return float((d1.mean() + d2.mean()) / 2.0)


# ==============================================================================
# METRICS — Geometry Fidelity
# ==============================================================================


def compute_vcdr(cup_2d, disc_2d):
    if isinstance(cup_2d, torch.Tensor):
        cup_rows = cup_2d.nonzero(as_tuple=True)[0]
        disc_rows = disc_2d.nonzero(as_tuple=True)[0]
        if disc_rows.numel() == 0:
            return 0.0
        cup_h = (
            (cup_rows.max() - cup_rows.min() + 1).float()
            if cup_rows.numel() > 0
            else torch.tensor(0.0)
        )
        disc_h = (disc_rows.max() - disc_rows.min() + 1).float()
        return (cup_h / disc_h).item()
    else:
        cr = np.where(cup_2d)[0]
        dr = np.where(disc_2d)[0]
        if len(dr) == 0:
            return 0.0
        ch = (cr.max() - cr.min() + 1) if len(cr) > 0 else 0
        dh = dr.max() - dr.min() + 1
        return ch / dh


def compute_acdr(cup_2d, disc_2d):
    ca = cup_2d.sum()
    da = disc_2d.sum()
    if isinstance(ca, torch.Tensor):
        ca, da = ca.item(), da.item()
    return ca / da if da > 0 else 0.0


def extract_radial_profile(mask_np, n_theta, image_size):
    H = W = image_size
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0
    R = min(H, W) / 2.0
    n_rho = 256
    thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    rhos = np.linspace(0, 1, n_rho)
    r_profile = np.zeros(n_theta, dtype=np.float32)
    for i, theta in enumerate(thetas):
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        px = cx + R * rhos * cos_t
        py = cy + R * rhos * sin_t
        px_int = np.clip(np.round(px).astype(int), 0, W - 1)
        py_int = np.clip(np.round(py).astype(int), 0, H - 1)
        samples = mask_np[py_int, px_int]
        fg = np.where(samples > 0.5)[0]
        r_profile[i] = rhos[fg[-1]] if len(fg) > 0 else 0.0
    return r_profile


def compute_4sector_rim(r_c, r_d, n_theta):
    rim = r_d - r_c
    aps = n_theta // 4
    sectors = {
        "temporal": np.concatenate(
            [np.arange(0, aps // 2), np.arange(n_theta - aps // 2, n_theta)]
        ),
        "superior": np.arange(aps // 2, aps // 2 + aps),
        "nasal": np.arange(aps // 2 + aps, aps // 2 + 2 * aps),
        "inferior": np.arange(aps // 2 + 2 * aps, aps // 2 + 3 * aps),
    }
    result = {}
    for name, idx in sectors.items():
        idx = idx[idx < n_theta]
        result[name] = float(rim[idx].mean()) if len(idx) > 0 else 0.0
    return result


def nesting_violation_check(cup_np, disc_np):
    violation = (cup_np > 0.5) & (disc_np < 0.5)
    n_viol = violation.sum()
    return n_viol > 0, float(n_viol) / cup_np.size


# ==============================================================================
# POLAR-TTA — Centre/Scale marginalisation for NPSNet
# ==============================================================================

DEFAULT_OFFSETS_PX = [0, 8, 16]
DEFAULT_SCALES = [0.85, 1.0, 1.15]
W_MASS, W_SHARPNESS, W_COMPACT = 0.4, 0.4, 0.2
DEFAULT_TOP_K = 3


def build_polar_grid(
    image_size, n_theta, n_rho, cx_offset=0.0, cy_offset=0.0, scale=1.0
):
    """Build a (1, N_ρ, N_θ, 2) sampling grid for one hypothesis."""
    H = W = image_size
    cx = (W - 1) / 2.0 + cx_offset
    cy = (H - 1) / 2.0 + cy_offset
    R = min(H, W) / 2.0 * scale

    theta = torch.linspace(0, 2 * math.pi, n_theta + 1)[:n_theta]
    rho = torch.linspace(0, 1, n_rho)

    rho_2d = rho.unsqueeze(1)
    theta_2d = theta.unsqueeze(0)

    px = cx + R * rho_2d * torch.cos(theta_2d)
    py = cy + R * rho_2d * torch.sin(theta_2d)

    grid_x = 2.0 * px / (W - 1) - 1.0
    grid_y = 2.0 * py / (H - 1) - 1.0

    grid = torch.stack([grid_x, grid_y], dim=-1)
    return grid.unsqueeze(0)


def build_cartesian_grid(
    image_size, n_theta, n_rho, cx_offset=0.0, cy_offset=0.0, scale=1.0
):
    """Build inverse (Cartesian → polar) grid for this hypothesis."""
    H = W = image_size
    cx = (W - 1) / 2.0 + cx_offset
    cy = (H - 1) / 2.0 + cy_offset
    R = min(H, W) / 2.0 * scale

    ys = torch.arange(H, dtype=torch.float32)
    xs = torch.arange(W, dtype=torch.float32)
    grid_y_img, grid_x_img = torch.meshgrid(ys, xs, indexing="ij")

    dx = grid_x_img - cx
    dy = grid_y_img - cy

    rho_cart = torch.sqrt(dx**2 + dy**2) / R
    theta_cart = torch.atan2(dy, dx) % (2 * math.pi)

    inside_circle = (rho_cart <= 1.0).float()

    inv_grid_y = 2.0 * rho_cart - 1.0
    inv_grid_x = theta_cart / math.pi - 1.0

    inv_grid = torch.stack([inv_grid_x, inv_grid_y], dim=-1)
    return inv_grid.unsqueeze(0), inside_circle


def make_hypotheses(offsets_px, scales):
    """Cartesian product of (Δx, Δy, scale) hypotheses."""
    pairs = list(itertools.product(offsets_px, offsets_px))
    hyps = [(dx, dy, s) for (dx, dy) in pairs for s in scales]
    canonical = (0.0, 0.0, 1.0)
    hyps = [h for h in hyps if h != canonical]
    hyps = [canonical] + hyps
    return hyps


def compute_self_score(out, inside_circle_dev):
    """Compute a no-GT proxy score for hypothesis quality."""
    P_d_polar = out["P_d_polar"]
    disc_mass = P_d_polar.mean(dim=[1, 2, 3])

    gamma_d = out["gamma_d"]
    sharpness = gamma_d.mean(dim=[1, 2])

    Y_d_cart = out["Y_d_cart"]
    disc_bin = (Y_d_cart > 0.5).float()
    inside = inside_circle_dev.unsqueeze(0).unsqueeze(0)
    n_disc = disc_bin.sum(dim=[1, 2, 3]).clamp(min=1.0)
    n_inside = (disc_bin * inside).sum(dim=[1, 2, 3])
    compactness = n_inside / n_disc

    return W_MASS * disc_mass + W_SHARPNESS * sharpness + W_COMPACT * compactness


@torch.no_grad()
def polar_tta_forward(model, images, hypotheses, device, top_k=DEFAULT_TOP_K):
    """Run Polar-TTA for one batch using the NPSNet (B4) model."""
    B = images.size(0)
    H = W = IMAGE_SIZE
    n_h = len(hypotheses)

    polar_grids, cart_grids, inside_masks = [], [], []
    for dx, dy, sc in hypotheses:
        pg = build_polar_grid(
            IMAGE_SIZE, N_THETA, N_RHO, cx_offset=dx, cy_offset=dy, scale=sc
        )
        ig, ins = build_cartesian_grid(
            IMAGE_SIZE, N_THETA, N_RHO, cx_offset=dx, cy_offset=dy, scale=sc
        )
        polar_grids.append(pg.to(device))
        cart_grids.append(ig.to(device))
        inside_masks.append(ins.to(device))

    orig_grid = model.polar_grid.grid.clone()

    all_cup_soft, all_disc_soft = [], []
    all_scores = []

    for h_idx, (dx, dy, sc) in enumerate(hypotheses):
        model.polar_grid.grid = polar_grids[h_idx].expand(B, -1, -1, -1)

        orig_inv_grid = model.warper.inv_grid.clone()
        orig_inside = model.warper.inside_circle.clone()
        model.warper.inv_grid = cart_grids[h_idx].expand(B, -1, -1, -1)
        model.warper.inside_circle = inside_masks[h_idx]

        with autocast("cuda"):
            out = model(images)

        model.polar_grid.grid = orig_grid
        model.warper.inv_grid = orig_inv_grid
        model.warper.inside_circle = orig_inside

        score = compute_self_score(out, inside_masks[h_idx])

        all_cup_soft.append(out["Y_c_cart"])
        all_disc_soft.append(out["Y_d_cart"])
        all_scores.append(score)

    model.polar_grid.grid = orig_grid

    scores_stk = torch.stack(all_scores, dim=1)
    k = min(top_k, n_h)
    topk_vals, topk_idx = scores_stk.topk(k, dim=1)
    weights = torch.softmax(topk_vals, dim=1)

    cup_stk = torch.stack(all_cup_soft, dim=1)
    disc_stk = torch.stack(all_disc_soft, dim=1)

    idx_exp_cart = topk_idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    idx_exp_cart = idx_exp_cart.expand(-1, -1, 1, H, W)

    cup_topk = cup_stk.gather(1, idx_exp_cart)
    disc_topk = disc_stk.gather(1, idx_exp_cart)

    w = weights.view(B, k, 1, 1, 1)
    cup_prob = (cup_topk * w).sum(dim=1)
    disc_prob = (disc_topk * w).sum(dim=1)

    best_idx = scores_stk.argmax(dim=1)
    best_hyps = [hypotheses[best_idx[i].item()] for i in range(B)]

    return {
        "cup_prob": cup_prob,
        "disc_prob": disc_prob,
        "best_hyps": best_hyps,
        "all_scores": scores_stk,
    }


# ==============================================================================
# EVALUATION — Unified for both baselines and NPS-Net
# ==============================================================================


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    dataset_name="Papila",
    n_theta=360,
    model_type="baseline",
    use_polar_tta=False,
    hypotheses=None,
    top_k=DEFAULT_TOP_K,
):
    """
    Evaluate a model on the given dataloader.

    Args:
        model_type: 'baseline' for 2-channel logit models,
                    'npsnet' for dict-output NPS-Net
        use_polar_tta: if True, use Polar-TTA for NPSNet inference
        hypotheses: list of (dx, dy, scale) TTA hypotheses
        top_k: number of top hypotheses to blend
    """
    model.eval()

    if use_polar_tta and hypotheses is None:
        hypotheses = make_hypotheses(DEFAULT_OFFSETS_PX, DEFAULT_SCALES)

    if use_polar_tta:
        n_h = len(hypotheses)
        print(f"  [Polar-TTA] {n_h} hypotheses, top-K={top_k}")

    all_cup_dice, all_disc_dice = [], []
    all_cup_iou, all_disc_iou = [], []
    all_cup_hd95, all_disc_hd95 = [], []
    all_cup_assd, all_disc_assd = [], []
    all_vcdr_pred, all_vcdr_gt = [], []
    all_acdr_pred, all_acdr_gt = [], []
    all_r_c_mae, all_r_d_mae = [], []
    all_rim_mae, all_4sec_mae = [], []
    all_nesting, all_viol_frac = [], []
    all_rim_corr = []
    best_hyp_counter = {}

    for batch in loader:
        images = batch["image"].to(device)
        cup_gt = batch["cup_mask"].to(device)
        disc_gt = batch["disc_mask"].to(device)
        gt_cdr = batch["cdr"]

        # ── Extract predictions based on model type ──────────────────────
        if model_type == "npsnet" and use_polar_tta:
            tta_out = polar_tta_forward(model, images, hypotheses, device, top_k)
            cup_pred = (tta_out["cup_prob"] > 0.5).float()
            disc_pred = (tta_out["disc_prob"] > 0.5).float()
            for hyp in tta_out["best_hyps"]:
                key = f"dx={hyp[0]:+.0f} dy={hyp[1]:+.0f} s={hyp[2]:.2f}"
                best_hyp_counter[key] = best_hyp_counter.get(key, 0) + 1
        elif model_type == "npsnet":
            with autocast("cuda"):
                output = model(images)
            cup_pred = (output["Y_c_cart"] > 0.5).float()
            disc_pred = (output["Y_d_cart"] > 0.5).float()
        else:
            with autocast("cuda"):
                output = model(images)
            # Baseline: (B, 2, H, W) logits → channel 0 = cup, channel 1 = disc
            cup_pred = (torch.sigmoid(output[:, 0:1]) > 0.5).float()
            disc_pred = (torch.sigmoid(output[:, 1:2]) > 0.5).float()

        B = images.size(0)
        for i in range(B):
            cp = cup_pred[i, 0]
            dp = disc_pred[i, 0]
            cg = cup_gt[i, 0]
            dg = disc_gt[i, 0]

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

            vcdr_p = compute_vcdr(cp, dp)
            all_vcdr_pred.append(vcdr_p)
            all_vcdr_gt.append(gt_cdr[i].item())

            acdr_p = compute_acdr(cp, dp)
            acdr_g = compute_acdr(cg, dg)
            all_acdr_pred.append(acdr_p)
            all_acdr_gt.append(acdr_g)

            # Radial profiles from Cartesian masks
            rc_pred = extract_radial_profile(cp_np, n_theta, IMAGE_SIZE)
            rd_pred = extract_radial_profile(dp_np, n_theta, IMAGE_SIZE)
            rc_gt = extract_radial_profile(cg_np, n_theta, IMAGE_SIZE)
            rd_gt = extract_radial_profile(dg_np, n_theta, IMAGE_SIZE)

            all_r_c_mae.append(np.abs(rc_pred - rc_gt).mean())
            all_r_d_mae.append(np.abs(rd_pred - rd_gt).mean())

            rim_pred = rd_pred - rc_pred
            rim_gt = rd_gt - rc_gt
            all_rim_mae.append(np.abs(rim_pred - rim_gt).mean())

            if rim_gt.std() > 1e-6 and rim_pred.std() > 1e-6:
                corr, _ = pearsonr(rim_pred, rim_gt)
                all_rim_corr.append(corr)

            sec_p = compute_4sector_rim(rc_pred, rd_pred, n_theta)
            sec_g = compute_4sector_rim(rc_gt, rd_gt, n_theta)
            all_4sec_mae.append(np.mean([abs(sec_p[s] - sec_g[s]) for s in sec_p]))

            has_v, vf = nesting_violation_check(cp_np, dp_np)
            all_nesting.append(has_v)
            all_viol_frac.append(vf)

    def safe_mean(lst):
        v = [x for x in lst if not (isinstance(x, float) and math.isnan(x))]
        return np.mean(v) if v else float("nan")

    result = {
        "dataset": dataset_name,
        "n": len(all_cup_dice),
        "cup_dice": np.mean(all_cup_dice),
        "disc_dice": np.mean(all_disc_dice),
        "cup_iou": np.mean(all_cup_iou),
        "disc_iou": np.mean(all_disc_iou),
        "cup_hd95": safe_mean(all_cup_hd95),
        "disc_hd95": safe_mean(all_disc_hd95),
        "cup_assd": safe_mean(all_cup_assd),
        "disc_assd": safe_mean(all_disc_assd),
        "vcdr_mae": np.mean(np.abs(np.array(all_vcdr_pred) - np.array(all_vcdr_gt))),
        "acdr_mae": np.mean(np.abs(np.array(all_acdr_pred) - np.array(all_acdr_gt))),
        "r_c_mae": np.mean(all_r_c_mae),
        "r_d_mae": np.mean(all_r_d_mae),
        "rim_mae": np.mean(all_rim_mae),
        "sector_rim_mae": np.mean(all_4sec_mae),
        "rim_corr": safe_mean(all_rim_corr),
        "nesting_viol_pct": 100.0 * np.mean(all_nesting),
        "viol_pixel_frac": np.mean(all_viol_frac),
        "cup_dice_std": np.std(all_cup_dice),
        "disc_dice_std": np.std(all_disc_dice),
    }

    if use_polar_tta:
        result["best_hyp_counter"] = best_hyp_counter

    return result


def print_results(r):
    print(f"\n{'=' * 70}")
    print(f"  {r['dataset']}  ({r['n']} samples)")
    print(f"{'=' * 70}")
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
    if "best_hyp_counter" in r and r["best_hyp_counter"]:
        print(f"  --- Polar-TTA Diagnostics ---")
        sorted_hyps = sorted(r["best_hyp_counter"].items(), key=lambda x: -x[1])
        total = sum(v for v in r["best_hyp_counter"].values())
        for key, cnt in sorted_hyps[:6]:
            print(f"  {key:<30}  won {cnt:4d} / {total}  ({100 * cnt / total:.1f}%)")
    print(f"{'=' * 70}\n")


# ==============================================================================
# VISUALISATION
# ==============================================================================


@torch.no_grad()
def save_visualisations(
    model, loader, device, output_dir, dataset_name, model_type="baseline", max_vis=16
):
    model.eval()
    vis_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(vis_dir, exist_ok=True)
    count = 0

    for batch in loader:
        if count >= max_vis:
            break
        images = batch["image"].to(device)
        with autocast("cuda"):
            output = model(images)

        if model_type == "npsnet":
            cup_pred = (output["Y_c_cart"] > 0.5).float()
            disc_pred = (output["Y_d_cart"] > 0.5).float()
        else:
            cup_pred = (torch.sigmoid(output[:, 0:1]) > 0.5).float()
            disc_pred = (torch.sigmoid(output[:, 1:2]) > 0.5).float()

        B = images.size(0)
        for i in range(B):
            if count >= max_vis:
                break
            prefix = os.path.join(vis_dir, f"{count:03d}")
            img = (images[i].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            cup_np = cup_pred[i, 0].cpu().numpy().astype(np.uint8)
            disc_np = disc_pred[i, 0].cpu().numpy().astype(np.uint8)

            overlay = img_bgr.copy()
            overlay[disc_np == 1] = (
                overlay[disc_np == 1] * 0.6 + np.array([0, 180, 0]) * 0.4
            ).astype(np.uint8)
            overlay[cup_np == 1] = (
                overlay[cup_np == 1] * 0.6 + np.array([0, 0, 200]) * 0.4
            ).astype(np.uint8)
            cv2.imwrite(f"{prefix}_overlay.png", overlay)

            if "cup_mask" in batch:
                gt_cup = batch["cup_mask"][i, 0].numpy().astype(np.uint8)
                gt_disc = batch["disc_mask"][i, 0].numpy().astype(np.uint8)
                gt_overlay = img_bgr.copy()
                gt_overlay[gt_disc == 1] = (
                    gt_overlay[gt_disc == 1] * 0.6 + np.array([0, 180, 0]) * 0.4
                ).astype(np.uint8)
                gt_overlay[gt_cup == 1] = (
                    gt_overlay[gt_cup == 1] * 0.6 + np.array([0, 0, 200]) * 0.4
                ).astype(np.uint8)
                cv2.imwrite(f"{prefix}_gt.png", gt_overlay)

            cv2.imwrite(f"{prefix}_mask_cup.png", cup_np * 255)
            cv2.imwrite(f"{prefix}_mask_disc.png", disc_np * 255)
            count += 1

    print(f"[vis] Saved {count} visualisations → {vis_dir}/")


# ==============================================================================
# MODEL LOADING
# ==============================================================================


def load_baseline_model(model_name, checkpoint_path=None):
    """Load a baseline model from Comparision/checkpoints/."""
    from training.train import build_model

    model = build_model(model_name).to(DEVICE)

    if checkpoint_path is None:
        # Try best_model first, then final_model
        ckpt = os.path.join(CHECKPOINT_DIR, model_name, "best_model.pth")
        if not os.path.isfile(ckpt):
            ckpt = os.path.join(CHECKPOINT_DIR, model_name, "final_model.pth")
        checkpoint_path = ckpt

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
        print(
            f"[baseline] Loaded {model_name} "
            f"(cup_dice={state.get('best_cup_dice', '?')})"
        )
    else:
        model.load_state_dict(state)
        print(f"[baseline] Loaded {model_name}")

    model.eval()
    return model


def _import_from_file(module_name, file_path):
    """Import a module from an explicit file path (avoids name collisions)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_npsnet(checkpoint_path=None):
    """Load NPSNet (AblationB4) from models/nps_net/ using explicit
    file imports to avoid config name collisions.

    The model is defined across three files:
        models/nps_net/config.py   — config constants
        models/nps_net/model_b2.py — shared polar components
        models/nps_net/model_b3.py — CupGateHead
        models/nps_net/model_b4.py — AblationB4 = full NPSNet
    """
    nps_dir = os.path.join(PROJECT_ROOT, "models", "nps_net")

    # Save modules that might collide
    saved_modules = {}
    for mod_name in ["config", "model_b2", "model_b3", "model_b4"]:
        saved_modules[mod_name] = sys.modules.pop(mod_name, None)

    # Import config first (model files do `from config import ...`)
    nps_config = _import_from_file("config", os.path.join(nps_dir, "config.py"))

    # Import model chain: model_b2 → model_b3 → model_b4
    _import_from_file("model_b2", os.path.join(nps_dir, "model_b2.py"))
    _import_from_file("model_b3", os.path.join(nps_dir, "model_b3.py"))
    b4_mod = _import_from_file("model_b4", os.path.join(nps_dir, "model_b4.py"))

    AblationB4 = b4_mod.AblationB4

    model = AblationB4(
        image_size=nps_config.IMAGE_SIZE,
        n_theta=nps_config.N_THETA,
        n_rho=nps_config.N_RHO,
        features=nps_config.ENCODER_FEATURES,
        shape_features=nps_config.SHAPE_FEATURES,
        temperature=nps_config.SOFTARGMAX_TEMPERATURE,
        prior_scale_init=nps_config.PRIOR_SCALE_INIT,
    ).to(DEVICE)

    # Restore original modules
    for mod_name, mod in saved_modules.items():
        if mod is not None:
            sys.modules[mod_name] = mod
        else:
            sys.modules.pop(mod_name, None)

    if checkpoint_path is None:
        candidates = [
            os.path.join(PROJECT_ROOT, "checkpoints", "b4", "best_model.pth"),
            os.path.join(PROJECT_ROOT, "checkpoints", "b4", "final_model.pth"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                checkpoint_path = c
                break

    if checkpoint_path is None or not os.path.isfile(str(checkpoint_path)):
        raise FileNotFoundError(
            f"NPSNet checkpoint not found. Searched:\n"
            + "\n".join(f"  - {c}" for c in candidates)
        )

    state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
        print(
            f"[npsnet] Loaded NPSNet (B4) from {checkpoint_path} "
            f"(cup_dice={state.get('best_cup_dice', '?')})"
        )
    else:
        model.load_state_dict(state)
        print(f"[npsnet] Loaded NPSNet (B4) from {checkpoint_path}")

    model.eval()
    return model


# ==============================================================================
# MAIN
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(description="Papila Inference — All Models")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=ALL_MODELS + ["all"],
        help='Model to evaluate, or "all" for every model',
    )
    parser.add_argument(
        "--csv", type=str, default=PAPILA_CSV, help="Path to Papila CSV"
    )
    parser.add_argument(
        "--save-vis", action="store_true", help="Save overlay visualisations"
    )
    parser.add_argument("--max-vis", type=int, default=16)
    parser.add_argument("--output-dir", type=str, default="predictions")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Override checkpoint path"
    )
    parser.add_argument(
        "--no-tta",
        action="store_true",
        help="Disable Polar-TTA for NPSNet (use vanilla forward)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Top-K hypotheses to blend in Polar-TTA",
    )
    parser.add_argument(
        "--offsets",
        type=str,
        default=",".join(str(o) for o in DEFAULT_OFFSETS_PX),
        help="Polar-TTA centre offsets in pixels",
    )
    parser.add_argument(
        "--scales",
        type=str,
        default=",".join(str(s) for s in DEFAULT_SCALES),
        help="Polar-TTA scale factors",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f"[ERROR] Papila CSV not found: {args.csv}")
        print("Run prepare_refuge_csv.py first to generate the CSV.")
        sys.exit(1)

    # Determine which models to run
    if args.model == "all":
        models_to_run = ALL_MODELS
    else:
        models_to_run = [args.model]

    # Build Polar-TTA hypotheses
    offsets_px = [float(x) for x in args.offsets.split(",")]
    scales = [float(x) for x in args.scales.split(",")]
    hypotheses = make_hypotheses(offsets_px, scales)

    all_results = []

    for model_name in models_to_run:
        print(f"\n{'═' * 70}")
        print(f"  Papila inference: {model_name}")
        print(f"{'═' * 70}")

        try:
            if model_name == "npsnet":
                model = load_npsnet(args.checkpoint)
                model_type = "npsnet"
            else:
                model = load_baseline_model(model_name, args.checkpoint)
                model_type = "baseline"
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue

        # Use Polar-TTA for NPSNet by default (disable with --no-tta)
        use_tta = model_type == "npsnet" and not args.no_tta

        loader = get_external_dataloader(args.csv, BATCH_SIZE, NUM_WORKERS)
        r = evaluate(
            model,
            loader,
            DEVICE,
            f"{model_name} Papila",
            n_theta=N_THETA,
            model_type=model_type,
            use_polar_tta=use_tta,
            hypotheses=hypotheses,
            top_k=args.top_k,
        )
        print_results(r)
        all_results.append(r)

        if args.save_vis:
            # Reload dataloader for vis (in case it was exhausted)
            loader = get_external_dataloader(args.csv, BATCH_SIZE, NUM_WORKERS)
            save_visualisations(
                model,
                loader,
                DEVICE,
                args.output_dir,
                f"{model_name}_Papila",
                model_type=model_type,
                max_vis=args.max_vis,
            )

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # ── Summary table ─────────────────────────────────────────────────────
    if len(all_results) > 1:
        print(f"\n{'=' * 110}")
        print(f"  Papila SUMMARY — Cross-Domain Generalisation")
        print(f"{'=' * 110}")
        h = (
            f"{'Model':<18} {'Cup Dice':>10} {'Disc Dice':>10} "
            f"{'Cup HD95':>10} {'Disc HD95':>10} {'vCDR MAE':>10} "
            f"{'Rim MAE':>10} {'Nest%':>8}"
        )
        print(h)
        print("-" * 110)
        for r in all_results:
            name = r["dataset"].replace(" Papila", "")
            print(
                f"{name:<18} {r['cup_dice']:>10.4f} {r['disc_dice']:>10.4f} "
                f"{r['cup_hd95']:>10.2f} {r['disc_hd95']:>10.2f} "
                f"{r['vcdr_mae']:>10.4f} {r['rim_mae']:>10.4f} "
                f"{r['nesting_viol_pct']:>7.1f}%"
            )
        print("=" * 110)


if __name__ == "__main__":
    main()
