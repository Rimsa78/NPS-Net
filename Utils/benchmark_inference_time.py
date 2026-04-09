#!/usr/bin/env python3
"""
Inference Time Benchmark — All Models with and without Polar-TTA.

Reports wall-clock inference time per image for:
    1. All 5 baseline models (single forward pass)
    2. NPS-Net without Polar-TTA (single forward pass)
    3. NPS-Net with Polar-TTA (27 hypotheses, top-K=3 blend)

Performs GPU warm-up, then times N repetitions for stable measurements.
Reports mean ± std and FPS.

Usage:
    cd Comparision/
    python benchmark_inference_time.py
    python benchmark_inference_time.py --n-repeats 50
"""

import argparse
import itertools
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast

from config import DEVICE, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS
from dataset import get_external_dataloader
from inference import load_model as load_baseline_model

PHOENIX_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RIM_CSV = os.path.join(PHOENIX_ROOT, 'Map', 'Corrected_testrim.csv')
NPSNET_CHECKPOINT = os.path.join(
    PHOENIX_ROOT, 'ThreeSixty', 'ablation', 'checkpoints', 'b4', 'best_model.pth')

N_THETA = 360
N_RHO = 256

# Polar-TTA grid (same as plot_polar_tta.py / inference_polar_tta_ablation.py)
OFFSETS_PX = [0, 8, 16]
SCALES = [0.85, 1.0, 1.15]
W_MASS = 0.4
W_SHARPNESS = 0.4
W_COMPACT = 0.2
TOP_K = 3

BASELINE_MODELS = ['vanilla', 'attunet', 'resunet', 'polar_unet', 'transunet',
                   'beal', 'dofe']
MODEL_DISPLAY = {
    'vanilla': 'VanillaUNet', 'attunet': 'AttentionUNet',
    'resunet': 'ResUNet', 'polar_unet': 'PolarUNet',
    'transunet': 'TransUNet', 'beal': 'BEAL', 'dofe': 'DoFE',
}


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
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        if 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
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
# Polar-TTA helpers (standalone)
# ==============================================================================

def build_polar_grid(image_size, n_theta, n_rho, cx_offset=0.0, cy_offset=0.0, scale=1.0):
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
    return torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)


def build_cartesian_grid(image_size, n_theta, n_rho, cx_offset=0.0, cy_offset=0.0, scale=1.0):
    H = W = image_size
    cx = (W - 1) / 2.0 + cx_offset
    cy = (H - 1) / 2.0 + cy_offset
    R = min(H, W) / 2.0 * scale
    ys = torch.arange(H, dtype=torch.float32)
    xs = torch.arange(W, dtype=torch.float32)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    dx = gx - cx
    dy = gy - cy
    rho_cart = torch.sqrt(dx ** 2 + dy ** 2) / R
    theta_cart = torch.atan2(dy, dx) % (2 * math.pi)
    inside = (rho_cart <= 1.0).float()
    inv_y = 2.0 * rho_cart - 1.0
    inv_x = theta_cart / math.pi - 1.0
    inv_grid = torch.stack([inv_x, inv_y], dim=-1)
    return inv_grid.unsqueeze(0), inside


def make_hypotheses(offsets_px, scales):
    pairs = list(itertools.product(offsets_px, offsets_px))
    hyps = [(dx, dy, s) for (dx, dy) in pairs for s in scales]
    canonical = (0.0, 0.0, 1.0)
    hyps = [h for h in hyps if h != canonical]
    hyps = [canonical] + hyps
    return hyps


def compute_self_score(out, inside_circle_dev):
    P_d = out['P_d_polar']
    disc_mass = P_d.mean(dim=[1, 2, 3])
    gamma_d = out['gamma_d']
    sharpness = gamma_d.mean(dim=[1, 2])
    Y_d = out['Y_d_cart']
    disc_bin = (Y_d > 0.5).float()
    inside = inside_circle_dev.unsqueeze(0).unsqueeze(0)
    n_disc = disc_bin.sum(dim=[1, 2, 3]).clamp(min=1.0)
    n_inside = (disc_bin * inside).sum(dim=[1, 2, 3])
    compactness = n_inside / n_disc
    return W_MASS * disc_mass + W_SHARPNESS * sharpness + W_COMPACT * compactness


@torch.no_grad()
def polar_tta_forward(model, images, hypotheses, device, top_k=TOP_K):
    """Full Polar-TTA forward pass for one batch."""
    B = images.size(0)
    H = W = IMAGE_SIZE
    n_h = len(hypotheses)

    polar_grids, cart_grids, inside_masks = [], [], []
    for (dx, dy, sc) in hypotheses:
        pg = build_polar_grid(IMAGE_SIZE, N_THETA, N_RHO, dx, dy, sc).to(device)
        ig, ins = build_cartesian_grid(IMAGE_SIZE, N_THETA, N_RHO, dx, dy, sc)
        polar_grids.append(pg)
        cart_grids.append(ig.to(device))
        inside_masks.append(ins.to(device))

    orig_grid = model.polar_grid.grid.clone()
    all_scores = []
    all_cup_soft = []
    all_disc_soft = []

    for h_idx, (dx, dy, sc) in enumerate(hypotheses):
        model.polar_grid.grid = polar_grids[h_idx].expand(B, -1, -1, -1)
        orig_inv = model.warper.inv_grid.clone()
        orig_ins = model.warper.inside_circle.clone()
        model.warper.inv_grid = cart_grids[h_idx].expand(B, -1, -1, -1)
        model.warper.inside_circle = inside_masks[h_idx]

        with autocast('cuda'):
            out = model(images)

        model.polar_grid.grid = orig_grid
        model.warper.inv_grid = orig_inv
        model.warper.inside_circle = orig_ins

        all_scores.append(compute_self_score(out, inside_masks[h_idx]))
        all_cup_soft.append(out['Y_c_cart'])
        all_disc_soft.append(out['Y_d_cart'])

    model.polar_grid.grid = orig_grid

    scores_stk = torch.stack(all_scores, dim=1)
    k = min(top_k, n_h)
    topk_vals, topk_idx = scores_stk.topk(k, dim=1)
    weights = torch.softmax(topk_vals, dim=1)

    cup_stk = torch.stack(all_cup_soft, dim=1)
    disc_stk = torch.stack(all_disc_soft, dim=1)

    idx_exp = topk_idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, H, W)
    cup_topk = cup_stk.gather(1, idx_exp)
    disc_topk = disc_stk.gather(1, idx_exp)

    w = weights.view(B, k, 1, 1, 1)
    cup_prob = (cup_topk * w).sum(dim=1)
    disc_prob = (disc_topk * w).sum(dim=1)

    return cup_prob, disc_prob


# ==============================================================================
# Timing utilities
# ==============================================================================

def warmup_gpu(model, dummy_input, n=5):
    """GPU warmup passes (not timed)."""
    model.eval()
    with torch.no_grad():
        for _ in range(n):
            with autocast('cuda'):
                _ = model(dummy_input)
    torch.cuda.synchronize()


def time_baseline(model, dummy_input, n_repeats):
    """Time single-pass inference for a baseline model."""
    model.eval()
    times = []
    with torch.no_grad():
        for _ in range(n_repeats):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with autocast('cuda'):
                _ = model(dummy_input)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)
    return np.array(times)


def time_npsnet_single(model, dummy_input, n_repeats):
    """Time NPS-Net single forward pass (no Polar-TTA)."""
    model.eval()
    times = []
    with torch.no_grad():
        for _ in range(n_repeats):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with autocast('cuda'):
                _ = model(dummy_input)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)
    return np.array(times)


def time_npsnet_tta(model, dummy_input, hypotheses, n_repeats):
    """Time NPS-Net with full Polar-TTA."""
    model.eval()
    device = dummy_input.device
    times = []
    with torch.no_grad():
        for _ in range(n_repeats):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = polar_tta_forward(model, dummy_input, hypotheses, device)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)
    return np.array(times)


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Inference time benchmark: all models ± Polar-TTA')
    parser.add_argument('--n-repeats', type=int, default=30,
                        help='Timing repetitions (default: 30)')
    parser.add_argument('--n-warmup', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for timing (default: 1 = per-image)')
    parser.add_argument('--npsnet-ckpt', default=NPSNET_CHECKPOINT)
    args = parser.parse_args()

    device = torch.device(DEVICE)
    BS = args.batch_size
    dummy = torch.randn(BS, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)

    hypotheses = make_hypotheses(OFFSETS_PX, SCALES)
    n_hyps = len(hypotheses)
    print(f"[info] Device: {device}")
    if device.type == 'cuda':
        print(f"[info] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[info] Image: {IMAGE_SIZE}×{IMAGE_SIZE}, batch={BS}")
    print(f"[info] Polar-TTA: {n_hyps} hypotheses, top-K={TOP_K}")
    print(f"[info] Repeats: {args.n_repeats}, warmup: {args.n_warmup}")

    results = []

    # ── Baselines ─────────────────────────────────────────────────────────────
    for mname in BASELINE_MODELS:
        print(f"\n[bench] {MODEL_DISPLAY[mname]} ...")
        model = load_baseline_model(mname)
        warmup_gpu(model, dummy, args.n_warmup)
        times = time_baseline(model, dummy, args.n_repeats)
        per_img = times / BS
        results.append({
            'Model': MODEL_DISPLAY[mname],
            'Mode': 'Single pass',
            'mean_ms': per_img.mean() * 1000,
            'std_ms': per_img.std() * 1000,
            'fps': 1.0 / per_img.mean(),
        })
        del model; torch.cuda.empty_cache()

    # ── NPS-Net (single pass) ────────────────────────────────────────────────
    print(f"\n[bench] NPS-Net (single pass) ...")
    nps_model = load_npsnet_model(args.npsnet_ckpt)
    warmup_gpu(nps_model, dummy, args.n_warmup)
    times = time_npsnet_single(nps_model, dummy, args.n_repeats)
    per_img = times / BS
    results.append({
        'Model': 'NPS-Net',
        'Mode': 'Single pass',
        'mean_ms': per_img.mean() * 1000,
        'std_ms': per_img.std() * 1000,
        'fps': 1.0 / per_img.mean(),
    })

    # ── NPS-Net (Polar-TTA) ─────────────────────────────────────────────────
    print(f"\n[bench] NPS-Net + Polar-TTA ({n_hyps} hypotheses) ...")
    warmup_gpu(nps_model, dummy, args.n_warmup)
    times = time_npsnet_tta(nps_model, dummy, hypotheses, args.n_repeats)
    per_img = times / BS
    results.append({
        'Model': 'NPS-Net',
        'Mode': f'Polar-TTA ({n_hyps}h)',
        'mean_ms': per_img.mean() * 1000,
        'std_ms': per_img.std() * 1000,
        'fps': 1.0 / per_img.mean(),
    })
    del nps_model; torch.cuda.empty_cache()

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print(f"  INFERENCE TIME BENCHMARK (batch={BS}, {args.n_repeats} repeats)")
    print(f"{'=' * 80}")
    print(f"  {'Model':<20} {'Mode':<20} {'Time/img (ms)':>16} {'FPS':>10}")
    print(f"  {'-' * 72}")
    for r in results:
        print(f"  {r['Model']:<20} {r['Mode']:<20} "
              f"{r['mean_ms']:>7.1f} ± {r['std_ms']:.1f} ms "
              f"{r['fps']:>8.1f}")
    print(f"{'=' * 80}")

    # Speedup ratios
    nps_single = [r for r in results if r['Model'] == 'NPS-Net' and r['Mode'] == 'Single pass'][0]
    nps_tta = [r for r in results if 'Polar-TTA' in r['Mode']][0]
    ratio = nps_tta['mean_ms'] / nps_single['mean_ms']
    print(f"\n  Polar-TTA overhead: {ratio:.1f}× single-pass time")
    print(f"  Polar-TTA still achieves {nps_tta['fps']:.1f} FPS "
          f"({'practical' if nps_tta['fps'] > 1 else 'offline-only'} for screening)")


if __name__ == '__main__':
    main()
