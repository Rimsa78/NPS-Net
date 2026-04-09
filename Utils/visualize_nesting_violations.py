#!/usr/bin/env python3
"""
Visualize nesting constraint violations.

Saves individual images for each model prediction that violates the nesting constraint
(cup extends outside disc boundary).

Usage:
    cd Comparision/
    python visualize_nesting_violations.py              # RIM (default)
    python visualize_nesting_violations.py --dataset papila
    python visualize_nesting_violations.py --n-plots 5   # 5 plots (default)
    python visualize_nesting_violations.py --out figures/
"""

import argparse
import os
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.amp import autocast

from config import DEVICE, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS
from dataset import get_external_dataloader, load_and_resize
from inference import dice_score, load_model as load_baseline_model

BASELINE_MODELS = [
    "vanilla",
    "attunet",
    "resunet",
    "polar_unet",
    "transunet",
    "beal",
    "dofe",
]

MODEL_DISPLAY = {
    "vanilla": "VanillaUNet",
    "attunet": "AttentionUNet",
    "resunet": "ResUNet",
    "polar_unet": "PolarUNet",
    "transunet": "TransUNet",
    "beal": "BEAL",
    "dofe": "DoFE",
    "npsnet": "NPSNet",
}

ALL_MODEL_KEYS = BASELINE_MODELS + ["npsnet"]

CUP_COLOR_RGB = (30, 120, 255)
DISC_COLOR_RGB = (0, 255, 50)
CUP_COLOR_BGR = CUP_COLOR_RGB[::-1]
DISC_COLOR_BGR = DISC_COLOR_RGB[::-1]

CONTOUR_THICKNESS = 3

PHOENIX_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RIM_CSV = os.path.join(PHOENIX_ROOT, "Map", "Corrected_testrim.csv")
PAPILA_CSV = os.path.join(PHOENIX_ROOT, "Map", "Corrected_papila.csv")
NPSNET_CHECKPOINT = os.path.join(
    PHOENIX_ROOT, "ThreeSixty", "ablation", "checkpoints", "b4", "best_model.pth"
)


def load_npsnet_model(checkpoint_path=None):
    if checkpoint_path is None:
        checkpoint_path = NPSNET_CHECKPOINT
    ablation_dir = os.path.join(PHOENIX_ROOT, "ThreeSixty", "ablation")
    saved = {}
    conflict_names = ["config", "dataset", "model_b2", "model_b3", "model_b4"]
    for name in conflict_names:
        if name in sys.modules:
            saved[name] = sys.modules.pop(name)
    sys.path.insert(0, ablation_dir)
    try:
        from model_b4 import AblationB4

        model = AblationB4().to(DEVICE)
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"NPSNet checkpoint not found: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
        model.eval()
        return model
    finally:
        sys.path.remove(ablation_dir)
        for name in conflict_names:
            sys.modules.pop(name, None)
        for name, mod in saved.items():
            sys.modules[name] = mod


def draw_boundary_overlay(bgr_uint8, cup_mask, disc_mask):
    overlay = bgr_uint8.copy()
    dc, _ = cv2.findContours(
        disc_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, dc, -1, DISC_COLOR_BGR, CONTOUR_THICKNESS)
    cc, _ = cv2.findContours(
        cup_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, cc, -1, CUP_COLOR_BGR, CONTOUR_THICKNESS)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


@torch.no_grad()
def collect_predictions_baseline(model, loader, device):
    model.eval()
    cup_dices, disc_dices, cup_preds, disc_preds = [], [], [], []
    for batch in loader:
        images = batch["image"].to(device)
        cup_gt = batch["cup_mask"].to(device)
        disc_gt = batch["disc_mask"].to(device)
        with autocast("cuda"):
            logits = model(images)
        cp = (torch.sigmoid(logits[:, 0:1]) > 0.5).float()
        dp = (torch.sigmoid(logits[:, 1:2]) > 0.5).float()
        for i in range(images.size(0)):
            cup_dices.append(dice_score(cp[i, 0], cup_gt[i, 0]).item())
            disc_dices.append(dice_score(dp[i, 0], disc_gt[i, 0]).item())
            cup_preds.append(cp[i, 0].cpu().numpy().astype(np.uint8))
            disc_preds.append(dp[i, 0].cpu().numpy().astype(np.uint8))
    return np.array(cup_dices), np.array(disc_dices), cup_preds, disc_preds


@torch.no_grad()
def collect_predictions_npsnet(model, loader, device):
    model.eval()
    cup_dices, disc_dices, cup_preds, disc_preds = [], [], [], []
    for batch in loader:
        images = batch["image"].to(device)
        cup_gt = batch["cup_mask"].to(device)
        disc_gt = batch["disc_mask"].to(device)
        with autocast("cuda"):
            out = model(images)
        cp = (out["Y_c_cart"] > 0.5).float()
        dp = (out["Y_d_cart"] > 0.5).float()
        for i in range(images.size(0)):
            cup_dices.append(dice_score(cp[i, 0], cup_gt[i, 0]).item())
            disc_dices.append(dice_score(dp[i, 0], disc_gt[i, 0]).item())
            cup_preds.append(cp[i, 0].cpu().numpy().astype(np.uint8))
            disc_preds.append(dp[i, 0].cpu().numpy().astype(np.uint8))
    return np.array(cup_dices), np.array(disc_dices), cup_preds, disc_preds


def load_raw_images_and_gt(csv_path, image_size):
    import pandas as pd

    df = pd.read_csv(csv_path)
    valid = (
        df["Raw"].apply(lambda x: os.path.isfile(str(x)))
        & df["Disk"].apply(lambda x: os.path.isfile(str(x)))
        & df["Cup"].apply(lambda x: os.path.isfile(str(x)))
    )
    df = df[valid].reset_index(drop=True)
    images, cup_gts, disc_gts = [], [], []
    for idx in range(len(df)):
        row = df.iloc[idx]
        img = load_and_resize(row["Raw"], image_size, is_mask=False)
        cup = load_and_resize(row["Cup"], image_size, is_mask=True)
        disc = load_and_resize(row["Disk"], image_size, is_mask=True)
        images.append(img)
        cup_gts.append((cup > 127).astype(np.uint8))
        disc_gts.append((disc > 127).astype(np.uint8))
    return images, cup_gts, disc_gts


def compute_nesting_scores(results, n_samples, all_models):
    nesting_scores = {}
    violations_per_sample = {i: [] for i in range(n_samples)}
    for mname in all_models:
        nesting_scores[mname] = np.zeros(n_samples, dtype=np.float64)
        for i in range(n_samples):
            cp = results[mname]["cup_pred"][i]
            dp = results[mname]["disc_pred"][i]
            violation = (cp > 0.5) & (dp < 0.5)
            nesting_scores[mname][i] = float(violation.sum()) / cp.size
            if nesting_scores[mname][i] > 0:
                violations_per_sample[i].append(mname)
    return nesting_scores, violations_per_sample


def render_single_violation_figure(
    sample_idx,
    raw_image,
    model_name,
    cup_pred,
    disc_pred,
    out_path,
    dpi=200,
):
    img_bgr = raw_image
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    ax.imshow(draw_boundary_overlay(img_bgr, cup_pred, disc_pred))
    ax.set_title(
        f"{MODEL_DISPLAY[model_name]} - Sample {sample_idx}\nNesting Violation",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_axis_off()

    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"[saved] {out_path}")
    plt.close(fig)


def render_combined_violation_figure(
    sample_idx,
    raw_image,
    cup_gt,
    disc_gt,
    results,
    model_names,
    out_path,
    dpi=200,
):
    img_bgr = raw_image
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    n_cols = len(model_names) + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]

    axes[0].imshow(img_rgb)
    axes[0].set_title("Input Image", fontsize=12, fontweight="bold")
    axes[0].set_axis_off()

    for col, mname in enumerate(model_names):
        cp = results[mname]["cup_pred"][sample_idx]
        dp = results[mname]["disc_pred"][sample_idx]
        axes[col + 1].imshow(draw_boundary_overlay(img_bgr, cp, dp))
        axes[col + 1].set_title(MODEL_DISPLAY[mname], fontsize=12, fontweight="bold")
        axes[col + 1].set_axis_off()

    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"[saved] {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize nesting constraint violations"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="rim",
        choices=["rim", "papila"],
        help="Dataset to evaluate on (default: rim)",
    )
    parser.add_argument(
        "--n-plots",
        type=int,
        default=5,
        help="Number of violation plots to generate (default: 5)",
    )
    parser.add_argument("--out", type=str, default="figures/", help="Output directory")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--npsnet-ckpt", type=str, default=NPSNET_CHECKPOINT)
    args = parser.parse_args()

    if args.dataset == "rim":
        csv_path = os.path.abspath(RIM_CSV)
        dataset_label = "RIM-ONE"
    else:
        csv_path = os.path.abspath(PAPILA_CSV)
        dataset_label = "Papila"

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isfile(csv_path):
        sys.exit(f"[ERROR] CSV not found: {csv_path}")

    loader = get_external_dataloader(
        csv_path, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )
    N = len(loader.dataset)
    print(f"[info] {dataset_label} dataset: {N} samples")

    results = {}
    for mname in BASELINE_MODELS:
        print(f"[info] Inference: {MODEL_DISPLAY[mname]} ...")
        model = load_baseline_model(mname)
        cd, dd, cp, dp = collect_predictions_baseline(model, loader, DEVICE)
        results[mname] = {
            "cup_dice": cd,
            "disc_dice": dd,
            "cup_pred": cp,
            "disc_pred": dp,
        }
        del model
        torch.cuda.empty_cache()

    print("[info] Inference: NPSNet ...")
    nps_model = load_npsnet_model(args.npsnet_ckpt)
    cd, dd, cp, dp = collect_predictions_npsnet(nps_model, loader, DEVICE)
    results["npsnet"] = {
        "cup_dice": cd,
        "disc_dice": dd,
        "cup_pred": cp,
        "disc_pred": dp,
    }
    del nps_model
    torch.cuda.empty_cache()

    nesting_scores, violations_per_sample = compute_nesting_scores(
        results, N, ALL_MODEL_KEYS
    )

    samples_with_multiple_violations = [
        i for i in range(N) if len(violations_per_sample[i]) >= 2
    ]

    if not samples_with_multiple_violations:
        print(
            "[warning] No samples with >= 2 models violations, using any violation samples"
        )
        samples_with_multiple_violations = [
            i for i in range(N) if len(violations_per_sample[i]) >= 1
        ]

    if not samples_with_multiple_violations:
        print("[warning] No nesting violations found at all!")
        samples_with_multiple_violations = []

    sorted_indices = sorted(
        samples_with_multiple_violations,
        key=lambda i: sum(nesting_scores[m][i] for m in violations_per_sample[i]),
        reverse=True,
    )
    top_violations = sorted_indices[: args.n_plots]

    print(f"\n[info] Top {args.n_plots} nesting violations:")
    for rank, idx in enumerate(top_violations):
        models_with_violation = violations_per_sample[idx]
        total_score = sum(nesting_scores[m][idx] for m in models_with_violation)
        print(
            f"  {rank + 1}. sample {idx}: models={models_with_violation}, score={total_score:.6f}"
        )

    print(f"\n[info] Loading raw {dataset_label} images ...")
    raw_images, cup_gts, disc_gts = load_raw_images_and_gt(csv_path, IMAGE_SIZE)

    for plot_num, sample_idx in enumerate(top_violations):
        models_with_violation = violations_per_sample[sample_idx]
        for model_idx, model_name in enumerate(models_with_violation):
            cp = results[model_name]["cup_pred"][sample_idx]
            dp = results[model_name]["disc_pred"][sample_idx]
            out_path = os.path.join(
                out_dir, f"nesting_violation_{plot_num + 1}_{model_name}.png"
            )
            render_single_violation_figure(
                sample_idx=sample_idx,
                raw_image=raw_images[sample_idx],
                model_name=model_name,
                cup_pred=cp,
                disc_pred=dp,
                out_path=out_path,
                dpi=args.dpi,
            )

    print(f"\n[done] Generated {args.n_plots} nesting violation plots in {out_dir}")


if __name__ == "__main__":
    main()
