# inference_ablation.py
#!/usr/bin/env python3
"""
Ablation Study — Standard Single-Pass Inference & Evaluation.

Evaluates any ablation variant (B1–B5) with standard single-pass inference.
All rows B1–B5 use identical evaluation; only B6 uses Polar-TTA (separate script).

Metrics computed:
    Standard segmentation: Cup/Disc Dice, IoU, HD95, ASSD
    Geometry fidelity: vCDR MAE, aCDR MAE, radial MAE, rim, 4-sector, rim corr
    Anatomical validity: nesting violation rate, violating-pixel fraction

Usage:
    python inference_ablation.py --variant b2 --test
    python inference_ablation.py --variant b3 --test --all-external
    python inference_ablation.py --variant b5 --test --save-vis
    python inference_ablation.py --variant b1 --test --checkpoint /path/to/b1.pth
"""

import argparse
import math
import os
import sys

import numpy as np
import cv2
import torch
from torch.amp import autocast
from scipy.ndimage import distance_transform_edt
from scipy.stats import pearsonr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "training"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "datasets"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "models"))

# Import from root config.py (ablation-specific)
import config as ablation_config
from training.config import (
    DEVICE,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    N_THETA,
    N_RHO,
    get_checkpoint_dir,
    B1_CHECKPOINT_DIR,
    B5_CHECKPOINT_DIR,
    ENCODER_FEATURES,
)
from datasets.dataset import get_test_dataloader, get_external_dataloader


# ==============================================================================
# MODEL FACTORY
# ==============================================================================


def get_model_for_variant(variant):
    """Instantiate the model for a given ablation variant."""
    if variant == "b1":
        # B1: Polar UNet baseline from models/baselines/
        from models.baselines.polar_unet import PolarUNet

        return PolarUNet(
            in_channels=3,
            out_channels=2,
            image_size=IMAGE_SIZE,
            n_theta=N_THETA,
            n_rho=N_RHO,
            features=ENCODER_FEATURES,
        )
    elif variant == "b2":
        from models.nps_net.model_b2 import AblationB2

        return AblationB2()
    elif variant == "b3":
        from models.nps_net.model_b3 import AblationB3

        return AblationB3()
    elif variant == "b4":
        from models.nps_net.model_b4 import AblationB4

        return AblationB4()
    elif variant == "b5":
        # B5: Polar-TTA on B4 — inference-only, uses B4 model
        # Redirect to inference_polar_tta_ablation.py for full Polar-TTA eval
        print("[B5] Note: B5 is Polar-TTA on B4. For full Polar-TTA evaluation,")
        print("     use: python inference_polar_tta_ablation.py --test --all-external")
        print("     Running standard (non-TTA) B4 eval as fallback...")
        from models.nps_net.model_b4 import AblationB4

        return AblationB4()
    else:
        raise ValueError(f"Unknown variant: {variant}")


def get_default_checkpoint(variant):
    """Get default checkpoint path for a variant."""
    if variant == "b1":
        return os.path.join(B1_CHECKPOINT_DIR, "polar_unet_best.pth")
    elif variant in ("b2", "b3", "b4"):
        return os.path.join(get_checkpoint_dir(variant), "best_model.pth")
    elif variant == "b5":
        # B5 is Polar-TTA on B4 — uses B4's checkpoint
        return os.path.join(get_checkpoint_dir("b4"), "best_model.pth")
    else:
        raise ValueError(f"Unknown variant: {variant}")


# ==============================================================================
# B1 ADAPTER — wrap PolarUNet output to match ablation evaluation API
# ==============================================================================


class B1Adapter(torch.nn.Module):
    """Wrap PolarUNet to produce output dict compatible with evaluation.

    PolarUNet outputs (B, 2, H, W) logits where channel 0 = cup, 1 = disc.
    We convert to the same dict format as B2-B5 models.
    """

    def __init__(self, model, n_theta=N_THETA, n_rho=N_RHO, image_size=IMAGE_SIZE):
        super().__init__()
        self.model = model
        self.n_theta = n_theta
        self.n_rho = n_rho
        self.image_size = image_size

    def _compute_radii_from_cartesian(self, mask_2d):
        """Extract approximate radii from a Cartesian binary mask.

        Args:
            mask_2d: (H, W) binary tensor

        Returns:
            r: (N_θ,) radii in [0, 1]
        """
        H = W = self.image_size
        cx = (W - 1) / 2.0
        cy = (H - 1) / 2.0
        R = min(H, W) / 2.0

        mask_np = mask_2d.cpu().numpy()
        thetas = np.linspace(0, 2 * np.pi, self.n_theta, endpoint=False)
        rhos = np.linspace(0, 1, self.n_rho)

        r_out = np.zeros(self.n_theta, dtype=np.float32)
        for i, theta in enumerate(thetas):
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            px = cx + R * rhos * cos_t
            py = cy + R * rhos * sin_t
            px_int = np.clip(np.round(px).astype(int), 0, W - 1)
            py_int = np.clip(np.round(py).astype(int), 0, H - 1)
            samples = mask_np[py_int, px_int]
            fg_idx = np.where(samples > 0.5)[0]
            r_out[i] = rhos[fg_idx[-1]] if len(fg_idx) > 0 else 0.0

        return torch.from_numpy(r_out)

    def forward(self, x):
        logits = self.model(x)  # (B, 2, H, W)
        cup_prob = torch.sigmoid(logits[:, 0:1])  # (B, 1, H, W)
        disc_prob = torch.sigmoid(logits[:, 1:2])

        B = x.size(0)
        r_c_list = []
        r_d_list = []
        for i in range(B):
            cup_bin = (cup_prob[i, 0] > 0.5).float()
            disc_bin = (disc_prob[i, 0] > 0.5).float()
            r_c_list.append(self._compute_radii_from_cartesian(cup_bin))
            r_d_list.append(self._compute_radii_from_cartesian(disc_bin))

        device = x.device
        return {
            "Y_c_cart": cup_prob,
            "Y_d_cart": disc_prob,
            "r_c_m": torch.stack(r_c_list).to(device),
            "r_d_m": torch.stack(r_d_list).to(device),
        }


# ==============================================================================
# METRICS — Standard Segmentation
# ==============================================================================


def dice_score(pred, target, eps=1e-7):
    pred_flat = pred.reshape(-1).float()
    tgt_flat = target.reshape(-1).float()
    inter = (pred_flat * tgt_flat).sum()
    return (2.0 * inter + eps) / (pred_flat.sum() + tgt_flat.sum() + eps)


def iou_score(pred, target, eps=1e-7):
    pred_flat = pred.reshape(-1).float()
    tgt_flat = target.reshape(-1).float()
    inter = (pred_flat * tgt_flat).sum()
    union = pred_flat.sum() + tgt_flat.sum() - inter
    return (inter + eps) / (union + eps)


def hausdorff_95(pred_np, target_np):
    if pred_np.sum() == 0 or target_np.sum() == 0:
        return float("nan")
    dt_pred = distance_transform_edt(~pred_np.astype(bool))
    dt_target = distance_transform_edt(~target_np.astype(bool))
    d_pred_to_gt = dt_target[pred_np.astype(bool)]
    d_gt_to_pred = dt_pred[target_np.astype(bool)]
    if len(d_pred_to_gt) == 0 or len(d_gt_to_pred) == 0:
        return float("nan")
    return float(np.percentile(np.concatenate([d_pred_to_gt, d_gt_to_pred]), 95))


def average_surface_distance(pred_np, target_np):
    if pred_np.sum() == 0 or target_np.sum() == 0:
        return float("nan")
    dt_pred = distance_transform_edt(~pred_np.astype(bool))
    dt_target = distance_transform_edt(~target_np.astype(bool))
    d_pred_to_gt = dt_target[pred_np.astype(bool)]
    d_gt_to_pred = dt_pred[target_np.astype(bool)]
    if len(d_pred_to_gt) == 0 or len(d_gt_to_pred) == 0:
        return float("nan")
    return float((d_pred_to_gt.mean() + d_gt_to_pred.mean()) / 2.0)


# ==============================================================================
# METRICS — Geometry Fidelity
# ==============================================================================


def compute_vcdr(cup_mask_2d, disc_mask_2d):
    if isinstance(cup_mask_2d, torch.Tensor):
        cup_rows = cup_mask_2d.nonzero(as_tuple=True)[0]
        disc_rows = disc_mask_2d.nonzero(as_tuple=True)[0]
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
        cup_rows = np.where(cup_mask_2d)[0]
        disc_rows = np.where(disc_mask_2d)[0]
        if len(disc_rows) == 0:
            return 0.0
        cup_h = (cup_rows.max() - cup_rows.min() + 1) if len(cup_rows) > 0 else 0
        disc_h = disc_rows.max() - disc_rows.min() + 1
        return cup_h / disc_h


def compute_acdr(cup_mask_2d, disc_mask_2d):
    cup_area = cup_mask_2d.sum()
    disc_area = disc_mask_2d.sum()
    if isinstance(cup_area, torch.Tensor):
        cup_area = cup_area.item()
        disc_area = disc_area.item()
    if disc_area == 0:
        return 0.0
    return cup_area / disc_area


def compute_4sector_rim(r_c, r_d, n_theta):
    rim = r_d - r_c
    if isinstance(rim, torch.Tensor):
        rim = rim.cpu().numpy()
    apq = n_theta // 4
    sectors = {
        "temporal": np.concatenate(
            [np.arange(0, apq // 2), np.arange(n_theta - apq // 2, n_theta)]
        ),
        "superior": np.arange(apq // 2, apq // 2 + apq),
        "nasal": np.arange(apq // 2 + apq, apq // 2 + 2 * apq),
        "inferior": np.arange(apq // 2 + 2 * apq, apq // 2 + 3 * apq),
    }
    result = {}
    for name, idx in sectors.items():
        idx = idx[idx < n_theta]
        result[name] = float(rim[idx].mean()) if len(idx) > 0 else 0.0
    return result


def nesting_violation_check(cup_mask, disc_mask):
    if isinstance(cup_mask, torch.Tensor):
        cup_mask = cup_mask.cpu().numpy()
        disc_mask = disc_mask.cpu().numpy()
    violation = (cup_mask > 0.5) & (disc_mask < 0.5)
    n_violation = violation.sum()
    return n_violation > 0, float(n_violation) / cup_mask.size


# ==============================================================================
# EVALUATION ENGINE
# ==============================================================================


@torch.no_grad()
def evaluate(model, loader, device, dataset_name="Test"):
    model.eval()

    all_cup_dice, all_disc_dice = [], []
    all_cup_iou, all_disc_iou = [], []
    all_cup_hd95, all_disc_hd95 = [], []
    all_cup_assd, all_disc_assd = [], []
    all_vcdr_pred, all_vcdr_gt = [], []
    all_acdr_pred, all_acdr_gt = [], []
    all_r_c_mae, all_r_d_mae = [], []
    all_rim_mae, all_4sector_mae = [], []
    all_nesting_viol, all_viol_frac = [], []
    all_rim_corr = []

    for batch in loader:
        images = batch["image"].to(device)
        cup_mask = batch["cup_mask"].to(device)
        disc_mask = batch["disc_mask"].to(device)
        gt_cdr = batch["cdr"]
        r_c_gt = batch["r_c_gt"].to(device)
        r_d_gt = batch["r_d_gt"].to(device)

        with autocast("cuda"):
            out = model(images)

        cup_pred = (out["Y_c_cart"] > 0.5).float()
        disc_pred = (out["Y_d_cart"] > 0.5).float()

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

            vcdr_p = compute_vcdr(cp, dp)
            vcdr_g = gt_cdr[i].item()
            all_vcdr_pred.append(vcdr_p)
            all_vcdr_gt.append(vcdr_g)

            acdr_p = compute_acdr(cp, dp)
            acdr_g = compute_acdr(cg, dg)
            all_acdr_pred.append(acdr_p)
            all_acdr_gt.append(acdr_g)

            rc_pred = out["r_c_m"][i]
            rd_pred = out["r_d_m"][i]
            rc_gt_i = r_c_gt[i]
            rd_gt_i = r_d_gt[i]

            all_r_c_mae.append((rc_pred - rc_gt_i).abs().mean().item())
            all_r_d_mae.append((rd_pred - rd_gt_i).abs().mean().item())

            rim_pred = rd_pred - rc_pred
            rim_gt = rd_gt_i - rc_gt_i
            all_rim_mae.append((rim_pred - rim_gt).abs().mean().item())

            rp_np = rim_pred.cpu().numpy()
            rg_np = rim_gt.cpu().numpy()
            if rg_np.std() > 1e-6 and rp_np.std() > 1e-6:
                corr, _ = pearsonr(rp_np, rg_np)
                all_rim_corr.append(corr)

            sectors_pred = compute_4sector_rim(rc_pred, rd_pred, N_THETA)
            sectors_gt = compute_4sector_rim(rc_gt_i, rd_gt_i, N_THETA)
            sector_mae = np.mean(
                [abs(sectors_pred[s] - sectors_gt[s]) for s in sectors_pred]
            )
            all_4sector_mae.append(sector_mae)

            has_viol, viol_frac = nesting_violation_check(cp_np, dp_np)
            all_nesting_viol.append(has_viol)
            all_viol_frac.append(viol_frac)

    def safe_mean(lst):
        valid = [x for x in lst if not (isinstance(x, float) and math.isnan(x))]
        return np.mean(valid) if valid else float("nan")

    results = {
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
        "sector_rim_mae": np.mean(all_4sector_mae),
        "rim_corr": safe_mean(all_rim_corr),
        "nesting_viol_pct": 100.0 * np.mean(all_nesting_viol),
        "viol_pixel_frac": np.mean(all_viol_frac),
        "cup_dice_std": np.std(all_cup_dice),
        "disc_dice_std": np.std(all_disc_dice),
    }
    return results


# ==============================================================================
# PRINTING
# ==============================================================================


def print_results(r, variant=None):
    variant_str = f" [{variant.upper()}]" if variant else ""
    print(f"\n{'=' * 70}")
    print(f"  {r['dataset']}{variant_str}  ({r['n']} samples)")
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
# MODEL LOADING
# ==============================================================================


def load_model(variant, checkpoint_path=None):
    """Load and return the model for a given variant."""
    model = get_model_for_variant(variant)

    # Wrap B1 in adapter
    if variant == "b1":
        model = B1Adapter(model)

    model = model.to(DEVICE)

    if checkpoint_path is None:
        checkpoint_path = get_default_checkpoint(variant)

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    if "model_state_dict" in state:
        # For B1 wrapped in adapter, load into inner model
        if variant == "b1":
            model.model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state["model_state_dict"])
        epoch = state.get("epoch", "?")
        cup_dice = state.get("best_cup_dice", state.get("best_dice", "?"))
        print(
            f"[{variant.upper()}] Loaded checkpoint  epoch={epoch}  cup_dice={cup_dice}"
        )
    else:
        if variant == "b1":
            model.model.load_state_dict(state)
        else:
            model.load_state_dict(state)
        print(f"[{variant.upper()}] Loaded raw state dict")

    model.eval()
    return model


# ==============================================================================
# MAIN
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="NPS-Net Ablation — Standard Inference"
    )
    parser.add_argument(
        "--variant",
        type=str,
        required=True,
        choices=["b1", "b2", "b3", "b4", "b5"],
        help="Ablation variant to evaluate",
    )
    parser.add_argument(
        "--test", action="store_true", help="Evaluate on held-out internal test split"
    )
    parser.add_argument(
        "--csv", type=str, default=None, help="Path to external CSV for evaluation"
    )
    parser.add_argument(
        "--name", type=str, default="External", help="Display name for --csv dataset"
    )
    parser.add_argument(
        "--all-external",
        action="store_true",
        help="Evaluate on RIM and Dristi external splits",
    )
    parser.add_argument(
        "--save-vis", action="store_true", help="Save per-sample visualisations"
    )
    parser.add_argument("--max-vis", type=int, default=16)
    parser.add_argument("--output-dir", type=str, default="predictions")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to model checkpoint"
    )
    args = parser.parse_args()

    if not (args.test or args.csv or args.all_external):
        parser.error("Provide at least one of --test, --csv, or --all-external")

    model = load_model(args.variant, args.checkpoint)
    all_results = []

    if args.test:
        print(f"\n>>> Evaluating {args.variant.upper()} on held-out TEST split ...")
        loader = get_test_dataloader(BATCH_SIZE, NUM_WORKERS)
        r = evaluate(model, loader, DEVICE, "Test Split")
        print_results(r, args.variant)
        all_results.append(r)

    if args.csv:
        print(f"\n>>> Evaluating {args.variant.upper()} on {args.csv} ...")
        loader = get_external_dataloader(args.csv, BATCH_SIZE, NUM_WORKERS)
        r = evaluate(model, loader, DEVICE, args.name)
        print_results(r, args.variant)
        all_results.append(r)

    if args.all_external:
        ext_datasets = [
            ("RIM", "../../Map/Corrected_testrim.csv"),
            ("Dristi", "../../Map/Corrected_DristiTest.csv"),
        ]
        for name, csv_path in ext_datasets:
            abs_path = os.path.join(os.path.dirname(__file__), csv_path)
            if not os.path.isfile(abs_path):
                print(f"[SKIP] {name}: file not found  {abs_path}")
                continue
            print(f"\n>>> Evaluating {args.variant.upper()} on {name} ...")
            loader = get_external_dataloader(abs_path, BATCH_SIZE, NUM_WORKERS)
            r = evaluate(model, loader, DEVICE, name)
            print_results(r, args.variant)
            all_results.append(r)

    if len(all_results) > 1:
        print("\n" + "=" * 100)
        print(f"  SUMMARY — Ablation {args.variant.upper()}")
        print("=" * 100)
        header = (
            f"{'Dataset':<15} {'Cup Dice':>10} {'Disc Dice':>10} "
            f"{'Cup HD95':>10} {'vCDR MAE':>10} {'Rim MAE':>10} "
            f"{'Rim Corr':>10} {'Nest%':>8}"
        )
        print(header)
        print("-" * 100)
        for r in all_results:
            row = (
                f"{r['dataset']:<15} {r['cup_dice']:>10.4f} {r['disc_dice']:>10.4f} "
                f"{r['cup_hd95']:>10.2f} {r['vcdr_mae']:>10.4f} {r['rim_mae']:>10.4f} "
                f"{r['rim_corr']:>10.4f} {r['nesting_viol_pct']:>7.1f}%"
            )
            print(row)
        print("=" * 100)


if __name__ == "__main__":
    main()
