#!/usr/bin/env python3
"""
Qualitative comparison: Best, Worst, and Extreme cases (single figure).

Supports both RIM-ONE and Papila datasets via --dataset flag.

One publication-quality figure with 3 sections (3 rows each = 9 rows total):
    Section 1 — Best Cases       (3 samples with highest cross-model mean Dice)
    Section 2 — Worst Cases      (prioritises nesting-violation samples)
    Section 3 — Extreme Cases    (3 most challenging, near-zero Dice for many models)

Columns:
    [Image] [GT] [VanillaUNet] [AttentionUNet] [ResUNet]
    [PolarUNet] [TransUNet] [NPSNet]

Overlays: clean RGB fundus + cup boundary (blue) + disc boundary (green).

Usage:
    cd Comparision/
    python visualize_qualitative.py                        # RIM (default)
    python visualize_qualitative.py --dataset papila       # Papila
    python visualize_qualitative.py --dataset rim --out figures/rim_qualitative.pdf
    python visualize_qualitative.py --cases-per-section 2
"""

import argparse
import os
import sys

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from torch.amp import autocast

# ── project imports (Comparision/) ────────────────────────────────────────────
from config import DEVICE, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS
from dataset import get_external_dataloader, load_and_resize
from inference import dice_score, load_model as load_baseline_model

# ── Constants ─────────────────────────────────────────────────────────────────
BASELINE_MODELS = ['vanilla', 'attunet', 'resunet', 'polar_unet', 'transunet',
                   'beal', 'dofe']

MODEL_DISPLAY = {
    'vanilla':    'VanillaUNet',
    'attunet':    'AttentionUNet',
    'resunet':    'ResUNet',
    'polar_unet': 'PolarUNet',
    'transunet':  'TransUNet',
    'beal':       'BEAL',
    'dofe':       'DoFE',
    'npsnet':     'Ours (NPSNet)',
}

ALL_MODEL_KEYS = BASELINE_MODELS + ['npsnet']

# Boundary colours (BGR for OpenCV, RGB for matplotlib legend)
CUP_COLOR_RGB  = (30,  120, 255)   # bright blue
DISC_COLOR_RGB = (0,   255,  50)   # vivid green
CUP_COLOR_BGR  = CUP_COLOR_RGB[::-1]
DISC_COLOR_BGR = DISC_COLOR_RGB[::-1]

CONTOUR_THICKNESS = 3

PHOENIX_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RIM_CSV = os.path.join(PHOENIX_ROOT, 'Map', 'Corrected_testrim.csv')
PAPILA_CSV = os.path.join(PHOENIX_ROOT, 'Map', 'Corrected_papila.csv')
NPSNET_CHECKPOINT = os.path.join(
    PHOENIX_ROOT, 'ThreeSixty', 'ablation', 'checkpoints', 'b4', 'best_model.pth')

# Section labels and colours
SECTIONS = [
    ('Best Cases',     '#27AE60'),
    ('Worst Cases',    '#E67E22'),
    ('Extreme Cases',  '#C0392B'),
]


# ==============================================================================
# NPSNet loader
# ==============================================================================

def load_npsnet_model(checkpoint_path=None):
    if checkpoint_path is None:
        checkpoint_path = NPSNET_CHECKPOINT
    ablation_dir = os.path.join(PHOENIX_ROOT, 'ThreeSixty', 'ablation')
    saved = {}
    conflict_names = ['config', 'dataset', 'model_b2', 'model_b3', 'model_b4']
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
        if 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
            print(f"[NPSNet] Loaded B4 (cup_dice={state.get('best_cup_dice', '?')})")
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


# ==============================================================================
# Helpers
# ==============================================================================

def draw_boundary_overlay(bgr_uint8, cup_mask, disc_mask):
    overlay = bgr_uint8.copy()
    dc, _ = cv2.findContours(disc_mask.astype(np.uint8),
                              cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, dc, -1, DISC_COLOR_BGR, CONTOUR_THICKNESS)
    cc, _ = cv2.findContours(cup_mask.astype(np.uint8),
                              cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, cc, -1, CUP_COLOR_BGR, CONTOUR_THICKNESS)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


@torch.no_grad()
def collect_predictions_baseline(model, loader, device):
    model.eval()
    cup_dices, disc_dices, cup_preds, disc_preds = [], [], [], []
    for batch in loader:
        images = batch['image'].to(device)
        cup_gt = batch['cup_mask'].to(device)
        disc_gt = batch['disc_mask'].to(device)
        with autocast('cuda'):
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
        images = batch['image'].to(device)
        cup_gt = batch['cup_mask'].to(device)
        disc_gt = batch['disc_mask'].to(device)
        with autocast('cuda'):
            out = model(images)
        cp = (out['Y_c_cart'] > 0.5).float()
        dp = (out['Y_d_cart'] > 0.5).float()
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
        df['Raw'].apply(lambda x: os.path.isfile(str(x))) &
        df['Disk'].apply(lambda x: os.path.isfile(str(x))) &
        df['Cup'].apply(lambda x: os.path.isfile(str(x)))
    )
    df = df[valid].reset_index(drop=True)
    images, cup_gts, disc_gts = [], [], []
    for idx in range(len(df)):
        row = df.iloc[idx]
        img  = load_and_resize(row['Raw'],  image_size, is_mask=False)
        cup  = load_and_resize(row['Cup'],  image_size, is_mask=True)
        disc = load_and_resize(row['Disk'], image_size, is_mask=True)
        images.append(img)
        cup_gts.append((cup > 127).astype(np.uint8))
        disc_gts.append((disc > 127).astype(np.uint8))
    return images, cup_gts, disc_gts


# ==============================================================================
# Single unified figure
# ==============================================================================

def render_unified_figure(
    section_indices,   # list of 3 lists of sample indices
    raw_images,
    cup_gts,
    disc_gts,
    results,
    out_path,
    dpi=200,
):
    """Render one figure with 3 labelled sections.

    Parameters
    ----------
    section_indices : list[list[int]]  — 3 sections, each a list of sample indices
    """
    K = len(section_indices[0])
    n_sections = len(section_indices)
    n_rows = K * n_sections
    n_cols = 2 + len(ALL_MODEL_KEYS)   # Image + GT + 6 models
    col_headers = ['Image', 'Ground Truth'] + [MODEL_DISPLAY[m] for m in ALL_MODEL_KEYS]

    cell_w = 2.2
    cell_h = 2.2
    label_w = 0.9   # left margin for section labels
    fig_w = cell_w * n_cols + label_w
    fig_h = cell_h * n_rows + 2.0   # headroom for headers + legend

    fig = plt.figure(figsize=(fig_w, fig_h))

    # Gridspec: n_rows × (1 + n_cols)  — first column is for section labels
    gs = gridspec.GridSpec(
        n_rows, n_cols + 1, figure=fig,
        width_ratios=[label_w / cell_w] + [1] * n_cols,
        wspace=0.03, hspace=0.08)

    # ── Fill image cells ──────────────────────────────────────────────────────
    # Store axes for row 0 so we can set column headers on them
    row0_axes = {}

    for sec_i, indices in enumerate(section_indices):
        for local_i, sample_idx in enumerate(indices):
            row = sec_i * K + local_i
            img_bgr = raw_images[sample_idx]
            gt_cup  = cup_gts[sample_idx]
            gt_disc = disc_gts[sample_idx]
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Col 1: clean image
            ax = fig.add_subplot(gs[row, 1])
            ax.imshow(img_rgb)
            ax.set_axis_off()
            if row == 0:
                row0_axes[1] = ax

            # Col 2: GT overlay
            ax = fig.add_subplot(gs[row, 2])
            ax.imshow(draw_boundary_overlay(img_bgr, gt_cup, gt_disc))
            ax.set_axis_off()
            if row == 0:
                row0_axes[2] = ax

            # Cols 3+: model predictions
            for col_j, mname in enumerate(ALL_MODEL_KEYS):
                cp = results[mname]['cup_pred'][sample_idx]
                dp = results[mname]['disc_pred'][sample_idx]
                ax = fig.add_subplot(gs[row, 3 + col_j])
                ax.imshow(draw_boundary_overlay(img_bgr, cp, dp))
                ax.set_axis_off()
                if row == 0:
                    row0_axes[3 + col_j] = ax

    # ── Column headers (set on the first-row image axes) ──────────────────────
    for col_j, header in enumerate(col_headers):
        ax = row0_axes.get(col_j + 1)
        if ax is not None:
            ax.set_title(header, fontsize=16, fontweight='bold', pad=10)

    # ── Section labels (left margin) ──────────────────────────────────────────
    for sec_i, (label, color) in enumerate(SECTIONS):
        row_start = sec_i * K
        row_end   = row_start + K - 1

        ax_label = fig.add_subplot(gs[row_start:row_end + 1, 0])
        ax_label.set_axis_off()
        ax_label.text(0.5, 0.5, label, transform=ax_label.transAxes,
                      ha='center', va='center', fontsize=16, fontweight='bold',
                      color=color, rotation=90)

    fig.savefig(out_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"[saved] {out_path}")
    plt.close(fig)


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Qualitative comparison: best / worst / extreme')
    parser.add_argument('--dataset', type=str, default='rim',
                        choices=['rim', 'papila'],
                        help='Dataset to evaluate on (default: rim)')
    parser.add_argument('--cases-per-section', type=int, default=3,
                        help='Samples per section (default: 3)')
    parser.add_argument('--out', type=str, default=None,
                        help='Output figure path (auto-set if omitted)')
    parser.add_argument('--dpi', type=int, default=200)
    parser.add_argument('--npsnet-ckpt', type=str, default=NPSNET_CHECKPOINT)
    args = parser.parse_args()

    # ── Resolve dataset CSV and default output path ───────────────────────────
    if args.dataset == 'rim':
        csv_path = os.path.abspath(RIM_CSV)
        dataset_label = 'RIM-ONE'
        default_out = 'figures/rim_qualitative.png'
    else:
        csv_path = os.path.abspath(PAPILA_CSV)
        dataset_label = 'Papila'
        default_out = 'figures/papila_qualitative.png'

    out_path = args.out if args.out is not None else default_out

    K = args.cases_per_section
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    # ── 1) Dataloader ─────────────────────────────────────────────────────────
    if not os.path.isfile(csv_path):
        sys.exit(f"[ERROR] CSV not found: {csv_path}")
    loader = get_external_dataloader(csv_path, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    N = len(loader.dataset)
    print(f"[info] {dataset_label} dataset: {N} samples")

    # ── 2) Inference: baselines ───────────────────────────────────────────────
    results = {}
    for mname in BASELINE_MODELS:
        print(f"[info] Inference: {MODEL_DISPLAY[mname]} ...")
        model = load_baseline_model(mname)
        cd, dd, cp, dp = collect_predictions_baseline(model, loader, DEVICE)
        results[mname] = {'cup_dice': cd, 'disc_dice': dd,
                          'cup_pred': cp, 'disc_pred': dp}
        del model; torch.cuda.empty_cache()

    # ── 3) Inference: NPSNet ─────────────────────────────────────────────────
    print("[info] Inference: NPSNet ...")
    nps_model = load_npsnet_model(args.npsnet_ckpt)
    cd, dd, cp, dp = collect_predictions_npsnet(nps_model, loader, DEVICE)
    results['npsnet'] = {'cup_dice': cd, 'disc_dice': dd,
                         'cup_pred': cp, 'disc_pred': dp}
    del nps_model; torch.cuda.empty_cache()

    # ── 4) Rank samples by cross-model mean Dice ─────────────────────────────
    mean_dice = np.zeros(N, dtype=np.float64)
    for mname in ALL_MODEL_KEYS:
        mean_dice += (results[mname]['cup_dice'] + results[mname]['disc_dice']) / 2.0
    mean_dice /= len(ALL_MODEL_KEYS)

    sorted_idx = np.argsort(mean_dice)  # ascending (worst → best)

    # Section 1: Best K
    best = sorted_idx[-K:][::-1].tolist()

    # Section 3: Extreme K (absolute worst)
    extreme = sorted_idx[:K].tolist()

    # Section 2: Worst — prioritise samples with nesting violations
    nesting_scores = np.zeros(N, dtype=np.float64)
    for mname in BASELINE_MODELS:
        for i in range(N):
            cp = results[mname]['cup_pred'][i]
            dp = results[mname]['disc_pred'][i]
            violation = (cp > 0.5) & (dp < 0.5)
            nesting_scores[i] += float(violation.sum()) / cp.size
    nesting_scores /= len(BASELINE_MODELS)

    extreme_set = set(extreme)
    violation_candidates = [
        i for i in range(N)
        if i not in extreme_set and nesting_scores[i] > 0
    ]
    violation_candidates.sort(key=lambda i: nesting_scores[i], reverse=True)

    if len(violation_candidates) >= K:
        worst = violation_candidates[:K]
    else:
        worst = list(violation_candidates)
        for idx in sorted_idx:
            if len(worst) >= K:
                break
            if idx not in extreme_set and idx not in worst:
                worst.append(idx)
        worst = worst[:K]

    def _log(label, indices):
        print(f"\n  [{label}]")
        for idx in indices:
            pm = {MODEL_DISPLAY[m]:
                  f"{(results[m]['cup_dice'][idx]+results[m]['disc_dice'][idx])/2:.3f}"
                  for m in ALL_MODEL_KEYS}
            nest_info = f"  nest_viol={nesting_scores[idx]:.6f}" if nesting_scores[idx] > 0 else ""
            print(f"    sample {idx:3d}  mean={mean_dice[idx]:.3f}{nest_info}  {pm}")

    _log("BEST", best)
    _log("WORST", worst)
    _log("EXTREME", extreme)

    # ── 5) Load raw images + GT ───────────────────────────────────────────────
    print(f"\n[info] Loading raw {dataset_label} images ...")
    raw_images, cup_gts, disc_gts = load_raw_images_and_gt(csv_path, IMAGE_SIZE)

    # ── 6) Render single unified figure ───────────────────────────────────────
    render_unified_figure(
        section_indices=[best, worst, extreme],
        raw_images=raw_images,
        cup_gts=cup_gts,
        disc_gts=disc_gts,
        results=results,
        out_path=out_path,
        dpi=args.dpi,
    )

    print(f"\n[done] {dataset_label} figure saved → {out_path}")


if __name__ == '__main__':
    main()
