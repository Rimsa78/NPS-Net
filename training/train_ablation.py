# train_ablation.py
#!/usr/bin/env python3
"""
Ablation Study — Unified Training Script.

Trains one of B2, B3, or B4 ablation variants with identical hyperparameters.
Uses the same data splits, learning rate schedule, and training loop as
the full NPS-Net.

Usage:
    python train_ablation.py --variant b2
    python train_ablation.py --variant b3
    python train_ablation.py --variant b4
    python train_ablation.py --variant b2 --resume
"""

import argparse
import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast

from config import (
    BATCH_SIZE,
    IMAGE_SIZE,
    DEVICE,
    NUM_WORKERS,
    SEED,
    NUM_EPOCHS,
    MAX_LR,
    N_THETA,
    N_RHO,
    WEIGHT_DECAY,
    GRAD_CLIP_MAX_NORM,
    STAGE_A_END,
    STAGE_B_END,
    get_checkpoint_dir,
)
from dataset import get_dataloaders, get_test_dataloader
from losses_ablation import get_loss_for_variant

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ==============================================================================
# MODEL FACTORY
# ==============================================================================


def get_model_for_variant(variant):
    """Instantiate the model for a given ablation variant."""
    if variant == "b2":
        from model_b2 import AblationB2

        return AblationB2()
    elif variant == "b3":
        from model_b3 import AblationB3

        return AblationB3()
    elif variant == "b4":
        from model_b4 import AblationB4

        return AblationB4()
    else:
        raise ValueError(f"Unknown variant: {variant}. Expected 'b2', 'b3', or 'b4'.")


# ==============================================================================
# METRICS
# ==============================================================================


def dice_score(pred_mask, target_mask, smooth=1.0):
    p = pred_mask.float().reshape(-1)
    t = target_mask.float().reshape(-1)
    inter = (p * t).sum()
    return (2.0 * inter + smooth) / (p.sum() + t.sum() + smooth)


def iou_score(pred_mask, target_mask, smooth=1.0):
    p = pred_mask.float().reshape(-1)
    t = target_mask.float().reshape(-1)
    inter = (p * t).sum()
    union = p.sum() + t.sum() - inter
    return (inter + smooth) / (union + smooth)


def compute_vcdr(cup_mask_2d, disc_mask_2d):
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


def batch_to_device(batch, device):
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
    }


# ==============================================================================
# TRAINING LOOP
# ==============================================================================

LOSS_KEYS = ["total", "cart", "polar", "rim", "dist", "shape", "cons", "smooth"]


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    epoch,
    total_epochs,
    device,
    batch_scheduler=None,
):
    model.train()

    running = {k: 0.0 for k in LOSS_KEYS}
    cup_dice_sum = 0.0
    disc_dice_sum = 0.0
    r_c_mae_sum = 0.0
    r_d_mae_sum = 0.0
    n_batches = 0
    nan_count = 0

    for batch in loader:
        batch = batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=(device.type == "cuda")):
            outputs = model(batch["image"])
            loss, loss_dict = criterion(outputs, batch, epoch=epoch)

        if not torch.isfinite(loss):
            nan_count += 1
            if nan_count <= 3:
                print(f"  ⚠ NaN/Inf loss at batch {n_batches + nan_count}, skipping")
            optimizer.zero_grad(set_to_none=True)
            if batch_scheduler is not None:
                batch_scheduler.step()
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
        scaler.step(optimizer)
        scaler.update()

        if batch_scheduler is not None:
            batch_scheduler.step()

        for k in running:
            running[k] += loss_dict.get(k, 0.0)

        with torch.no_grad():
            cup_pred = (outputs["Y_c_cart"] > 0.5).float()
            disc_pred = (outputs["Y_d_cart"] > 0.5).float()
            cup_dice_sum += dice_score(cup_pred, batch["cup_mask"]).item()
            disc_dice_sum += dice_score(disc_pred, batch["disc_mask"]).item()
            r_c_mae_sum += (outputs["r_c_m"] - batch["r_c_gt"]).abs().mean().item()
            r_d_mae_sum += (outputs["r_d_m"] - batch["r_d_gt"]).abs().mean().item()

        n_batches += 1

    if nan_count > 0:
        print(f"  ⚠ {nan_count} NaN batches skipped this epoch")

    N = max(n_batches, 1)
    avg = {k: v / N for k, v in running.items()}
    avg["cup_dice"] = cup_dice_sum / N
    avg["disc_dice"] = disc_dice_sum / N
    avg["r_c_mae"] = r_c_mae_sum / N
    avg["r_d_mae"] = r_d_mae_sum / N
    return avg


# ==============================================================================
# VALIDATION
# ==============================================================================


@torch.no_grad()
def validate(model, loader, criterion, device, epoch=0):
    model.eval()
    running = {k: 0.0 for k in LOSS_KEYS}
    cup_dice_sum = 0.0
    disc_dice_sum = 0.0
    cup_iou_sum = 0.0
    disc_iou_sum = 0.0
    r_c_mae_sum = 0.0
    r_d_mae_sum = 0.0
    rim_mae_sum = 0.0
    vcdr_mae_sum = 0.0
    n_samples = 0
    n_batches = 0

    for batch in loader:
        batch = batch_to_device(batch, device)
        outputs = model(batch["image"])
        loss, loss_dict = criterion(outputs, batch, epoch=epoch)

        if not torch.isfinite(loss):
            continue

        for k in running:
            running[k] += loss_dict.get(k, 0.0)

        cup_pred = (outputs["Y_c_cart"] > 0.5).float()
        disc_pred = (outputs["Y_d_cart"] > 0.5).float()

        cup_dice_sum += dice_score(cup_pred, batch["cup_mask"]).item()
        disc_dice_sum += dice_score(disc_pred, batch["disc_mask"]).item()
        cup_iou_sum += iou_score(cup_pred, batch["cup_mask"]).item()
        disc_iou_sum += iou_score(disc_pred, batch["disc_mask"]).item()

        r_c_mae_sum += (outputs["r_c_m"] - batch["r_c_gt"]).abs().mean().item()
        r_d_mae_sum += (outputs["r_d_m"] - batch["r_d_gt"]).abs().mean().item()

        rim_pred = outputs["r_d_m"] - outputs["r_c_m"]
        rim_gt = batch["r_d_gt"] - batch["r_c_gt"]
        rim_mae_sum += (rim_pred - rim_gt).abs().mean().item()

        B = batch["image"].size(0)
        for i in range(B):
            vcdr_pred = compute_vcdr(cup_pred[i, 0], disc_pred[i, 0])
            vcdr_gt = batch["cdr"][i].item()
            vcdr_mae_sum += abs(vcdr_pred - vcdr_gt)
        n_samples += B
        n_batches += 1

    N = max(n_batches, 1)
    avg = {k: v / N for k, v in running.items()}
    avg["cup_dice"] = cup_dice_sum / N
    avg["disc_dice"] = disc_dice_sum / N
    avg["cup_iou"] = cup_iou_sum / N
    avg["disc_iou"] = disc_iou_sum / N
    avg["r_c_mae"] = r_c_mae_sum / N
    avg["r_d_mae"] = r_d_mae_sum / N
    avg["rim_mae"] = rim_mae_sum / N
    avg["vcdr_mae"] = vcdr_mae_sum / max(n_samples, 1)
    return avg


# ==============================================================================
# CHECKPOINT UTILITIES
# ==============================================================================


def _save_checkpoint(model, optimizer, epoch, best_dice, path):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_cup_dice": best_dice,
        },
        path,
    )
    print(f"  -> Saved checkpoint: {path} (cup_dice={best_dice:.4f})")


def _save_resume_checkpoint(model, optimizer, epoch, best_dice, scaler, ckpt_dir):
    path = os.path.join(ckpt_dir, "resume_checkpoint.pth")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_cup_dice": best_dice,
        },
        path,
    )


def _load_resume_checkpoint(device, ckpt_dir):
    path = os.path.join(ckpt_dir, "resume_checkpoint.pth")
    if os.path.isfile(path):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        print(
            f"  Loaded resume checkpoint: epoch={ckpt['epoch']}, "
            f"best_cup_dice={ckpt['best_cup_dice']:.4f}"
        )
        return ckpt
    return None


# ==============================================================================
# STAGE NAME (for B4 with staged training; B2/B3 always show "A-only")
# ==============================================================================


def _stage_name(epoch, variant):
    if variant in ("b2", "b3"):
        return "A-only (flat)"
    elif epoch < STAGE_A_END:
        return "A (dense warmup)"
    elif epoch < STAGE_B_END:
        return "B (+shape prior)"
    else:
        return "C (+consistency)"


# ==============================================================================
# MAIN TRAINING PIPELINE
# ==============================================================================


def train(variant, resume=False):
    warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*optimizer.step.*")
    warnings.filterwarnings("ignore", message=".*ShiftScaleRotate.*Affine.*")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device(DEVICE)
    ckpt_dir = get_checkpoint_dir(variant)

    print("Loading data...")
    train_loader, val_loader = get_dataloaders(BATCH_SIZE, NUM_WORKERS)

    print(f"Building Ablation {variant.upper()} model...")
    model = get_model_for_variant(variant).to(device)
    criterion = get_loss_for_variant(variant).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    best_cup_dice = 0.0
    start_epoch = 0

    if resume:
        resume_ckpt = _load_resume_checkpoint(device, ckpt_dir)
        if resume_ckpt is not None:
            model.load_state_dict(resume_ckpt["model_state_dict"])
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
            scaler.load_state_dict(resume_ckpt["scaler_state_dict"])
            best_cup_dice = resume_ckpt["best_cup_dice"]
            start_epoch = resume_ckpt["epoch"] + 1
            print(
                f"  Resuming from epoch {start_epoch}, best_cup_dice={best_cup_dice:.4f}"
            )
        else:
            print("  No resume checkpoint found, starting from scratch.")

    if HAS_WANDB:
        wandb.init(
            project="NPS_Net_Ablation",
            config={
                "variant": variant,
                "image_size": IMAGE_SIZE,
                "batch_size": BATCH_SIZE,
                "num_epochs": NUM_EPOCHS,
                "max_lr": MAX_LR,
                "n_theta": N_THETA,
                "n_rho": N_RHO,
            },
            mode="offline",
            name=f"ablation_{variant}",
            resume="allow" if resume else None,
        )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=MAX_LR,
        steps_per_epoch=len(train_loader),
        epochs=NUM_EPOCHS,
        pct_start=0.1,
    )
    if start_epoch > 0:
        for _ in range(start_epoch * len(train_loader)):
            scheduler.step()
        print(f"  Fast-forwarded OneCycleLR by {start_epoch} epochs")

    variant_desc = {
        "b2": "B2: + Monotone Occupancy (Independent Heads)",
        "b3": "B3: + Factorized Nesting (P_c = P_d · Q)",
        "b4": "B4: + Shape Prior & Confidence Gating",
    }

    print("\n" + "=" * 70)
    print(f"Ablation Training — {variant_desc[variant]}")
    print(f"  Epochs: {NUM_EPOCHS}, Max LR: {MAX_LR}")
    print(f"  Polar grid: N_θ={N_THETA}, N_ρ={N_RHO}")
    if variant in ("b2", "b3"):
        print(f"  Loss: Stage A only (L_cart + L_polar + L_rim) for all epochs")
    else:
        print(f"  Stage A (ep 0-{STAGE_A_END - 1}): Dense warmup")
        print(f"  Stage B (ep {STAGE_A_END}-{STAGE_B_END - 1}): + Shape prior")
        print(f"  Stage C (ep {STAGE_B_END}-{NUM_EPOCHS - 1}): + Consistency")
    print(f"  Checkpoints: {ckpt_dir}")
    print("=" * 70)

    for epoch in range(start_epoch, NUM_EPOCHS):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            epoch=epoch,
            total_epochs=NUM_EPOCHS,
            device=device,
            batch_scheduler=scheduler,
        )

        val_metrics = validate(model, val_loader, criterion, device=device, epoch=epoch)
        dt = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        stage = _stage_name(epoch, variant)

        # Get fusion scale if B4
        lambda_c_str = ""
        if variant == "b4" and hasattr(model, "fusion"):
            lambda_c_str = f"λ_c={model.fusion.lambda_c.item():.3f} "

        print(
            f"[E{epoch + 1:02d}/{NUM_EPOCHS}|{stage}] "
            f"lr={lr:.2e} | "
            f"loss={val_metrics['total']:.4f} "
            f"(c={val_metrics['cart']:.3f} "
            f"p={val_metrics['polar']:.3f} "
            f"rm={val_metrics['rim']:.4f} "
            f"d={val_metrics['dist']:.3f} "
            f"sh={val_metrics['shape']:.3f} "
            f"cn={val_metrics['cons']:.3f}) | "
            f"cup={val_metrics['cup_dice']:.4f} "
            f"disc={val_metrics['disc_dice']:.4f} "
            f"vCDR={val_metrics['vcdr_mae']:.4f} "
            f"{lambda_c_str}| "
            f"{dt:.1f}s"
        )

        if HAS_WANDB:
            log = {f"train/{k}": v for k, v in train_metrics.items()}
            log.update({f"val/{k}": v for k, v in val_metrics.items()})
            log["lr"] = lr
            wandb.log(log, step=epoch)

        if val_metrics["cup_dice"] > best_cup_dice:
            best_cup_dice = val_metrics["cup_dice"]
            _save_checkpoint(
                model,
                optimizer,
                epoch,
                best_cup_dice,
                path=os.path.join(ckpt_dir, "best_model.pth"),
            )

        _save_resume_checkpoint(
            model,
            optimizer,
            epoch,
            best_dice=best_cup_dice,
            scaler=scaler,
            ckpt_dir=ckpt_dir,
        )

    _save_checkpoint(
        model,
        optimizer,
        NUM_EPOCHS - 1,
        best_cup_dice,
        path=os.path.join(ckpt_dir, "final_model.pth"),
    )

    print("\n" + "=" * 70)
    print(f"Training complete! Variant: {variant.upper()}")
    print(f"Best val cup Dice: {best_cup_dice:.4f}")
    print(f"Checkpoints saved in: {ckpt_dir}")
    print("=" * 70)

    _evaluate_test_set(model, criterion, device, ckpt_dir, variant)

    if HAS_WANDB:
        wandb.finish()


def _evaluate_test_set(model, criterion, device, ckpt_dir, variant):
    best_path = os.path.join(ckpt_dir, "best_model.pth")
    if not os.path.isfile(best_path):
        print("  No best_model.pth found, skipping test evaluation.")
        return

    print("\n" + "=" * 70)
    print(f"TEST SET EVALUATION — Ablation {variant.upper()}")
    print("=" * 70)

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    try:
        model.load_state_dict(ckpt["model_state_dict"])
    except RuntimeError as e:
        print(f"  [WARN] Could not load best_model.pth: {e}")
        return
    print(f"  Loaded best_model.pth (cup_dice={ckpt['best_cup_dice']:.4f})")

    test_loader = get_test_dataloader(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    test_metrics = validate(
        model, test_loader, criterion, device=device, epoch=NUM_EPOCHS
    )

    print(f"\n{'Metric':<30} {'Value':>10}")
    print("-" * 42)
    print(f"{'Cup Dice':<30} {test_metrics['cup_dice']:>10.4f}")
    print(f"{'Disc Dice':<30} {test_metrics['disc_dice']:>10.4f}")
    print(f"{'Cup IoU':<30} {test_metrics['cup_iou']:>10.4f}")
    print(f"{'Disc IoU':<30} {test_metrics['disc_iou']:>10.4f}")
    print(f"{'vCDR MAE':<30} {test_metrics['vcdr_mae']:>10.4f}")
    print(f"{'Cup Radial MAE':<30} {test_metrics['r_c_mae']:>10.4f}")
    print(f"{'Disc Radial MAE':<30} {test_metrics['r_d_mae']:>10.4f}")
    print(f"{'Rim Profile MAE':<30} {test_metrics['rim_mae']:>10.4f}")
    print(f"{'Total Loss':<30} {test_metrics['total']:>10.4f}")
    print("-" * 42)

    if variant in ("b3", "b4"):
        print(f"  Nesting: by construction (P_c = P_d · Q)")
    else:
        print(f"  Nesting: NOT guaranteed (independent heads)")

    if HAS_WANDB:
        wandb.log({f"test/{k}": v for k, v in test_metrics.items()})

    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NPS-Net Ablation — Training")
    parser.add_argument(
        "--variant",
        type=str,
        required=True,
        choices=["b2", "b3", "b4"],
        help="Ablation variant to train",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()
    train(variant=args.variant, resume=args.resume)
