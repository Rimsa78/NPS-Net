#!/usr/bin/env python3
"""
Statistical Significance Testing — NPS-Net vs All Baselines.

Runs all 6 models on both the internal test set and RIM-ONE external set,
collects per-sample metrics, and performs Wilcoxon signed-rank tests for
every (NPS-Net vs baseline) pair.

Outputs:
    1. Console summary with p-values and effect sizes
    2. LaTeX-ready table (figures/significance_table.tex)
    3. Per-sample CSV dump for reproducibility (figures/per_sample_metrics.csv)

Metrics tested:
    Cup Dice, Disc Dice, vCDR MAE

Usage:
    cd Comparision/
    python compute_significance.py
    python compute_significance.py --out-dir figures
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.amp import autocast
from scipy.stats import wilcoxon

# ── Project imports ───────────────────────────────────────────────────────────
from config import DEVICE, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS
from dataset import get_test_dataloader, get_external_dataloader
from inference import dice_score, compute_vcdr, load_model as load_baseline_model

PHOENIX_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RIM_CSV = os.path.join(PHOENIX_ROOT, 'Map', 'Corrected_testrim.csv')
NPSNET_CHECKPOINT = os.path.join(
    PHOENIX_ROOT, 'ThreeSixty', 'ablation', 'checkpoints', 'b4', 'best_model.pth')

BASELINE_MODELS = ['vanilla', 'attunet', 'resunet', 'polar_unet', 'transunet',
                   'beal', 'dofe']
MODEL_DISPLAY = {
    'vanilla': 'VanillaUNet', 'attunet': 'AttentionUNet',
    'resunet': 'ResUNet', 'polar_unet': 'PolarUNet',
    'transunet': 'TransUNet', 'beal': 'BEAL', 'dofe': 'DoFE',
    'npsnet': 'NPS-Net',
}
ALL_MODEL_KEYS = BASELINE_MODELS + ['npsnet']


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
# Per-sample metric collection
# ==============================================================================

@torch.no_grad()
def collect_metrics_baseline(model, loader, device):
    """Collect per-sample Cup Dice, Disc Dice, vCDR(pred), vCDR(GT)."""
    model.eval()
    cup_d, disc_d, vcdr_p, vcdr_g = [], [], [], []
    for batch in loader:
        images = batch['image'].to(device)
        cup_gt = batch['cup_mask'].to(device)
        disc_gt = batch['disc_mask'].to(device)
        gt_cdr = batch['cdr']
        with autocast('cuda'):
            logits = model(images)
        cp = (torch.sigmoid(logits[:, 0:1]) > 0.5).float()
        dp = (torch.sigmoid(logits[:, 1:2]) > 0.5).float()
        for i in range(images.size(0)):
            cup_d.append(dice_score(cp[i, 0], cup_gt[i, 0]).item())
            disc_d.append(dice_score(dp[i, 0], disc_gt[i, 0]).item())
            vcdr_p.append(compute_vcdr(cp[i, 0], dp[i, 0]))
            vcdr_g.append(gt_cdr[i].item())
    return {
        'cup_dice': np.array(cup_d),
        'disc_dice': np.array(disc_d),
        'vcdr_pred': np.array(vcdr_p),
        'vcdr_gt': np.array(vcdr_g),
    }


@torch.no_grad()
def collect_metrics_npsnet(model, loader, device):
    model.eval()
    cup_d, disc_d, vcdr_p, vcdr_g = [], [], [], []
    for batch in loader:
        images = batch['image'].to(device)
        cup_gt = batch['cup_mask'].to(device)
        disc_gt = batch['disc_mask'].to(device)
        gt_cdr = batch['cdr']
        with autocast('cuda'):
            out = model(images)
        cp = (out['Y_c_cart'] > 0.5).float()
        dp = (out['Y_d_cart'] > 0.5).float()
        for i in range(images.size(0)):
            cup_d.append(dice_score(cp[i, 0], cup_gt[i, 0]).item())
            disc_d.append(dice_score(dp[i, 0], disc_gt[i, 0]).item())
            vcdr_p.append(compute_vcdr(cp[i, 0], dp[i, 0]))
            vcdr_g.append(gt_cdr[i].item())
    return {
        'cup_dice': np.array(cup_d),
        'disc_dice': np.array(disc_d),
        'vcdr_pred': np.array(vcdr_p),
        'vcdr_gt': np.array(vcdr_g),
    }


# ==============================================================================
# Statistical tests
# ==============================================================================

def wilcoxon_test(a, b):
    """Paired Wilcoxon signed-rank test. Returns (statistic, p-value).
    If all differences are zero, returns (nan, 1.0)."""
    diff = a - b
    if np.all(diff == 0):
        return float('nan'), 1.0
    try:
        stat, p = wilcoxon(a, b, alternative='two-sided')
        return stat, p
    except ValueError:
        return float('nan'), 1.0


def format_p(p):
    """Format p-value for display."""
    if p < 0.001:
        return '< 0.001'
    elif p < 0.01:
        return f'{p:.3f}'
    elif p < 0.05:
        return f'{p:.3f}'
    else:
        return f'{p:.3f}'


def format_p_latex(p):
    """Format p-value for LaTeX."""
    if p < 0.001:
        return '$< 0.001$'
    else:
        return f'${p:.3f}$'


def significance_stars(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'n.s.'


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Statistical significance: NPS-Net vs baselines')
    parser.add_argument('--out-dir', default='figures')
    parser.add_argument('--rim-csv', default=RIM_CSV)
    parser.add_argument('--npsnet-ckpt', default=NPSNET_CHECKPOINT)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── 1) Load dataloaders ──────────────────────────────────────────────────
    print("[info] Loading internal test set ...")
    test_loader = get_test_dataloader(BATCH_SIZE, NUM_WORKERS)

    rim_csv = os.path.abspath(args.rim_csv)
    print("[info] Loading RIM-ONE ...")
    rim_loader = get_external_dataloader(rim_csv, BATCH_SIZE, NUM_WORKERS)

    # ── 2) Collect per-sample metrics for each model ─────────────────────────
    # results[model_key][dataset_name] = {cup_dice, disc_dice, vcdr_pred, vcdr_gt}
    results = {}

    for mname in BASELINE_MODELS:
        print(f"[info] Inference: {MODEL_DISPLAY[mname]} ...")
        model = load_baseline_model(mname)
        results[mname] = {
            'test': collect_metrics_baseline(model, test_loader, DEVICE),
            'rim':  collect_metrics_baseline(model, rim_loader, DEVICE),
        }
        del model; torch.cuda.empty_cache()

    print("[info] Inference: NPS-Net ...")
    nps_model = load_npsnet_model(args.npsnet_ckpt)
    results['npsnet'] = {
        'test': collect_metrics_npsnet(nps_model, test_loader, DEVICE),
        'rim':  collect_metrics_npsnet(nps_model, rim_loader, DEVICE),
    }
    del nps_model; torch.cuda.empty_cache()

    # ── 3) Run Wilcoxon tests ────────────────────────────────────────────────
    metrics = ['cup_dice', 'disc_dice']
    metric_labels = {'cup_dice': 'Cup Dice', 'disc_dice': 'Disc Dice'}
    datasets = ['test', 'rim']
    dataset_labels = {'test': 'Internal Test', 'rim': 'RIM-ONE'}

    # Also test vCDR MAE
    vcdr_metric = 'vcdr_mae'

    print("\n" + "=" * 90)
    print("  STATISTICAL SIGNIFICANCE — NPS-Net vs Each Baseline (Wilcoxon Signed-Rank)")
    print("=" * 90)

    all_rows = []   # for LaTeX table

    for ds in datasets:
        print(f"\n  {'─' * 40}")
        print(f"  Dataset: {dataset_labels[ds]} (N={len(results['npsnet'][ds]['cup_dice'])})")
        print(f"  {'─' * 40}")

        nps = results['npsnet'][ds]
        nps_vcdr_mae = np.abs(nps['vcdr_pred'] - nps['vcdr_gt'])

        for metric in metrics + [vcdr_metric]:
            if metric == vcdr_metric:
                label = 'vCDR MAE'
                nps_vals = nps_vcdr_mae
            else:
                label = metric_labels[metric]
                nps_vals = nps[metric]

            print(f"\n    {label}:")
            print(f"    {'Baseline':<18} {'Mean±Std':>18} {'NPS Mean±Std':>18} "
                  f"{'Δ':>8} {'p-value':>10} {'Sig':>6}")
            print(f"    {'-' * 82}")

            for bname in BASELINE_MODELS:
                baseline = results[bname][ds]
                if metric == vcdr_metric:
                    b_vals = np.abs(baseline['vcdr_pred'] - baseline['vcdr_gt'])
                else:
                    b_vals = baseline[metric]

                b_mean = b_vals.mean()
                b_std = b_vals.std()
                n_mean = nps_vals.mean()
                n_std = nps_vals.std()

                # For Dice, higher is better → NPS should be higher
                # For MAE, lower is better → NPS should be lower
                if metric == vcdr_metric:
                    delta = b_mean - n_mean   # positive = NPS is better
                else:
                    delta = n_mean - b_mean   # positive = NPS is better

                _, p = wilcoxon_test(nps_vals, b_vals)
                stars = significance_stars(p)

                print(f"    {MODEL_DISPLAY[bname]:<18} "
                      f"{b_mean:>7.4f} ± {b_std:.4f} "
                      f"{n_mean:>7.4f} ± {n_std:.4f} "
                      f"{delta:>+7.4f}  {format_p(p):>10}  {stars:>6}")

                all_rows.append({
                    'Dataset': dataset_labels[ds],
                    'Metric': label,
                    'Baseline': MODEL_DISPLAY[bname],
                    'Baseline_Mean': f'{b_mean:.4f}',
                    'Baseline_Std': f'{b_std:.4f}',
                    'NPSNet_Mean': f'{n_mean:.4f}',
                    'NPSNet_Std': f'{n_std:.4f}',
                    'Delta': f'{delta:+.4f}',
                    'p_value': p,
                    'Sig': stars,
                })

    print(f"\n{'=' * 90}\n")

    # ── 4) Save per-sample CSV ───────────────────────────────────────────────
    csv_rows = []
    for ds in datasets:
        N = len(results['npsnet'][ds]['cup_dice'])
        for i in range(N):
            row = {'dataset': dataset_labels[ds], 'sample_idx': i}
            for mname in ALL_MODEL_KEYS:
                m = results[mname][ds]
                row[f'{mname}_cup_dice'] = m['cup_dice'][i]
                row[f'{mname}_disc_dice'] = m['disc_dice'][i]
                row[f'{mname}_vcdr_pred'] = m['vcdr_pred'][i]
            row['vcdr_gt'] = results['npsnet'][ds]['vcdr_gt'][i]
            csv_rows.append(row)

    csv_path = os.path.join(args.out_dir, 'per_sample_metrics.csv')
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"[saved] {csv_path}")

    # ── 5) Generate LaTeX table ──────────────────────────────────────────────
    tex_path = os.path.join(args.out_dir, 'significance_table.tex')
    with open(tex_path, 'w') as f:
        f.write("% Statistical significance: NPS-Net vs Baselines (Wilcoxon signed-rank)\n")
        f.write("% Auto-generated by compute_significance.py\n\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Statistical significance of NPS-Net improvements "
                "(Wilcoxon signed-rank test, two-sided).}\n")
        f.write("\\label{tab:significance}\n")
        f.write("\\resizebox{\\textwidth}{!}{%\n")
        f.write("\\begin{tabular}{ll l cc c r l}\n")
        f.write("\\toprule\n")
        f.write("Dataset & Metric & Baseline & Baseline & NPS-Net "
                "& $\\Delta$ & $p$-value & Sig. \\\\\n")
        f.write("\\midrule\n")

        prev_ds = None
        prev_metric = None
        for row in all_rows:
            ds = row['Dataset']
            met = row['Metric']

            if ds != prev_ds:
                if prev_ds is not None:
                    f.write("\\midrule\n")
                prev_ds = ds
                prev_metric = None

            ds_str = ds if ds != prev_ds else ''

            met_str = met if met != prev_metric else ''
            prev_metric = met

            p = row['p_value']
            p_str = format_p_latex(p)

            # Bold NPS-Net if significant
            nps_str = f"{row['NPSNet_Mean']} $\\pm$ {row['NPSNet_Std']}"
            if p < 0.05:
                nps_str = f"\\textbf{{{row['NPSNet_Mean']}}} $\\pm$ {row['NPSNet_Std']}"

            f.write(f"  {ds} & {met} & {row['Baseline']} & "
                    f"{row['Baseline_Mean']} $\\pm$ {row['Baseline_Std']} & "
                    f"{nps_str} & "
                    f"{row['Delta']} & {p_str} & {row['Sig']} \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}}\n")
        f.write("\\end{table}\n")

    print(f"[saved] {tex_path}")
    print("\n[done] Statistical significance analysis complete.")


if __name__ == '__main__':
    main()
