# inference.py
"""
Comprehensive inference & evaluation for all baseline models.

Reports the SAME metrics as NPS-Net for fair comparison:
  - Standard segmentation: Cup/Disc Dice, IoU, HD95, ASSD
  - Geometry fidelity: vCDR MAE, aCDR MAE, radial MAE, rim MAE, 4-sector rim, rim corr
  - Anatomical validity: nesting violation %, violating-pixel fraction

Usage:
    python inference.py --model vanilla --test
    python inference.py --model attunet --test --all-external --save-vis
    python inference.py --model polar_unet --test --all-external
"""

import argparse
import math
import os

import numpy as np
import cv2
import torch
from torch.amp import autocast
from scipy.ndimage import distance_transform_edt
from scipy.stats import pearsonr

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
from datasets.dataset import get_test_dataloader, get_external_dataloader


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
    """Extract GT-style radial boundary from a binary Cartesian mask."""
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
# EVALUATION
# ==============================================================================


@torch.no_grad()
def evaluate(model, loader, device, dataset_name="Test", n_theta=360):
    model.eval()

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

    for batch in loader:
        images = batch["image"].to(device)
        cup_gt = batch["cup_mask"].to(device)
        disc_gt = batch["disc_mask"].to(device)
        gt_cdr = batch["cdr"]

        with autocast("cuda"):
            logits = model(images)  # (B, 2, H, W)

        cup_pred = (torch.sigmoid(logits[:, 0:1]) > 0.5).float()
        disc_pred = (torch.sigmoid(logits[:, 1:2]) > 0.5).float()

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

            # Radial profiles from predicted masks
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

    return {
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
    print(f"{'=' * 70}\n")


# ==============================================================================
# VISUALISATION
# ==============================================================================


@torch.no_grad()
def save_visualisations(model, loader, device, output_dir, dataset_name, max_vis=16):
    model.eval()
    vis_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(vis_dir, exist_ok=True)
    count = 0

    for batch in loader:
        if count >= max_vis:
            break
        images = batch["image"].to(device)
        with autocast("cuda"):
            logits = model(images)

        cup_pred = (torch.sigmoid(logits[:, 0:1]) > 0.5).float()
        disc_pred = (torch.sigmoid(logits[:, 1:2]) > 0.5).float()

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


def load_model(model_name, checkpoint_path=None):
    from training.train import build_model

    model = build_model(model_name).to(DEVICE)

    if checkpoint_path is None:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, model_name, "best_model.pth")

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
        print(
            f"[model] Loaded {model_name} (cup_dice={state.get('best_cup_dice', '?')})"
        )
    else:
        model.load_state_dict(state)

    model.eval()
    return model


# ==============================================================================
# MAIN
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(description="Baseline Inference")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "vanilla",
            "attunet",
            "resunet",
            "polar_unet",
            "transunet",
            "beal",
            "dofe",
        ],
    )
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--name", type=str, default="External")
    parser.add_argument("--all-external", action="store_true")
    parser.add_argument("--save-vis", action="store_true")
    parser.add_argument("--max-vis", type=int, default=16)
    parser.add_argument("--output-dir", type=str, default="predictions")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    if not (args.test or args.csv or args.all_external):
        parser.error("Provide --test, --csv, or --all-external")

    model = load_model(args.model, args.checkpoint)
    all_results = []

    if args.test:
        print(f"\n>>> Test evaluation: {args.model}")
        loader = get_test_dataloader(BATCH_SIZE, NUM_WORKERS)
        r = evaluate(model, loader, DEVICE, f"{args.model} Test")
        print_results(r)
        all_results.append(r)
        if args.save_vis:
            save_visualisations(
                model,
                loader,
                DEVICE,
                args.output_dir,
                f"{args.model}_test",
                args.max_vis,
            )

    if args.csv:
        loader = get_external_dataloader(args.csv, BATCH_SIZE, NUM_WORKERS)
        r = evaluate(model, loader, DEVICE, f"{args.model} {args.name}")
        print_results(r)
        all_results.append(r)

    if args.all_external:
        ext = [
            ("RIM", "../Map/Corrected_testrim.csv"),
            ("Dristi", "../Map/Corrected_DristiTest.csv"),
        ]
        for name, csv_path in ext:
            abs_path = os.path.join(os.path.dirname(__file__), csv_path)
            if not os.path.isfile(abs_path):
                print(f"[SKIP] {name}: not found {abs_path}")
                continue
            print(f"\n>>> External: {name}")
            loader = get_external_dataloader(abs_path, BATCH_SIZE, NUM_WORKERS)
            r = evaluate(model, loader, DEVICE, f"{args.model} {name}")
            print_results(r)
            all_results.append(r)
            if args.save_vis:
                save_visualisations(
                    model,
                    loader,
                    DEVICE,
                    args.output_dir,
                    f"{args.model}_{name}",
                    args.max_vis,
                )

    if len(all_results) > 1:
        print(f"\n{'=' * 100}")
        print(f"  SUMMARY — {args.model}")
        print(f"{'=' * 100}")
        h = f"{'Dataset':<25} {'Cup Dice':>10} {'Disc Dice':>10} {'Cup HD95':>10} {'vCDR MAE':>10} {'Rim MAE':>10} {'Nest%':>8}"
        print(h)
        print("-" * 100)
        for r in all_results:
            print(
                f"{r['dataset']:<25} {r['cup_dice']:>10.4f} {r['disc_dice']:>10.4f} "
                f"{r['cup_hd95']:>10.2f} {r['vcdr_mae']:>10.4f} {r['rim_mae']:>10.4f} "
                f"{r['nesting_viol_pct']:>7.1f}%"
            )
        print("=" * 100)


if __name__ == "__main__":
    main()
