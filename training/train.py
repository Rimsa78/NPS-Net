# train.py
"""
Unified training script for all baseline models.

Usage:
    python train.py --model vanilla
    python train.py --model attunet
    python train.py --model resunet
    python train.py --model polar_unet
    python train.py --model transunet
    python train.py --model beal
    python train.py --model dofe
    python train.py --model vanilla --resume
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

from models.baselines import (
    build_model,
    vanilla,
    attunet,
    resunet,
    polar_unet,
    transunet,
    beal,
    dofe,
)
from models.baselines.losses import BaselineLoss
from training.config import (
    BATCH_SIZE,
    IMAGE_SIZE,
    DEVICE,
    NUM_WORKERS,
    SEED,
    NUM_EPOCHS,
    MAX_LR,
    FEATURES,
    N_THETA,
    N_RHO,
    WEIGHT_DECAY,
    CHECKPOINT_DIR,
    GRAD_CLIP_MAX_NORM,
    DISC_LR,
    N_PSEUDO_DOMAINS,
)
from datasets.dataset import get_dataloaders, get_test_dataloader

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

ALL_MODELS = [
    "vanilla",
    "attunet",
    "resunet",
    "polar_unet",
    "transunet",
    "beal",
    "dofe",
]


# ==============================================================================
# MODEL FACTORY
# ==============================================================================


def build_model(model_name):
    """Instantiate the requested baseline model with 2 output channels."""
    model_name = model_name.lower()

    if model_name == "vanilla":
        from models.baselines.vanilla import VanillaUNet

        return VanillaUNet(in_channels=3, out_channels=2, features=FEATURES)

    elif model_name == "attunet":
        from models.baselines.attunet import AttentionUNet

        return AttentionUNet(in_channels=3, out_channels=2, features=FEATURES)

    elif model_name == "resunet":
        from models.baselines.resunet import ResUNet

        return ResUNet(in_channels=3, out_channels=2, features=FEATURES)

    elif model_name == "polar_unet":
        from models.baselines.polar_unet import PolarUNet

        return PolarUNet(
            in_channels=3,
            out_channels=2,
            image_size=IMAGE_SIZE,
            n_theta=N_THETA,
            n_rho=N_RHO,
            features=FEATURES,
        )

    elif model_name == "transunet":
        from models.baselines.transunet import TransUNet

        return TransUNet(
            in_channels=3,
            out_channels=2,
            img_size=IMAGE_SIZE,
            cnn_features=[64, 128, 256],
            d_model=512,
            num_heads=8,
            num_layers=6,
            use_grad_checkpoint=True,
        )

    elif model_name == "beal":
        from models.baselines.beal import BEAL

        return BEAL(in_channels=3, out_channels=2)

    elif model_name == "dofe":
        from models.baselines.dofe import DoFE

        return DoFE(in_channels=3, out_channels=2, n_domains=N_PSEUDO_DOMAINS)

    else:
        raise ValueError(
            f"Unknown model: {model_name}. Choose from: {', '.join(ALL_MODELS)}"
        )


def build_loss_fn(model_name, device):
    """Build the appropriate loss function for the given model."""
    if model_name == "beal":
        from models.baselines.beal_loss import BEALLoss

        return BEALLoss().to(device)
    elif model_name == "dofe":
        from models.baselines.dofe_loss import DoFELoss

        return DoFELoss().to(device)
    else:
        return BaselineLoss().to(device)


# ==============================================================================
# METRICS
# ==============================================================================


def dice_score(logits_channel, targets, threshold=0.5):
    p = (torch.sigmoid(logits_channel) > threshold).float()
    t = targets.float()
    inter = (p * t).sum()
    return (2.0 * inter + 1.0) / (p.sum() + t.sum() + 1.0)


def iou_score(logits_channel, targets, threshold=0.5):
    p = (torch.sigmoid(logits_channel) > threshold).float()
    t = targets.float()
    inter = (p * t).sum()
    union = p.sum() + t.sum() - inter
    return (inter + 1.0) / (union + 1.0)


# ==============================================================================
# TRAINING LOOPS
# ==============================================================================


def train_one_epoch(
    model, loader, criterion, optimizer, scaler, device, batch_scheduler=None
):
    """Standard training loop for vanilla/attunet/resunet/polar_unet/transunet."""
    model.train()
    running = {"total": 0.0}
    cup_dice_sum = 0.0
    disc_dice_sum = 0.0
    n_batches = 0
    nan_count = 0

    for batch in loader:
        images = batch["image"].to(device)
        cup_mask = batch["cup_mask"].to(device)
        disc_mask = batch["disc_mask"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(images)  # (B, 2, H, W)
            loss, loss_dict = criterion(logits, cup_mask, disc_mask)

        if not torch.isfinite(loss):
            nan_count += 1
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

        running["total"] += loss_dict["total"]
        with torch.no_grad():
            cup_dice_sum += dice_score(logits[:, 0:1], cup_mask).item()
            disc_dice_sum += dice_score(logits[:, 1:2], disc_mask).item()
        n_batches += 1

    N = max(n_batches, 1)
    return {
        "total": running["total"] / N,
        "cup_dice": cup_dice_sum / N,
        "disc_dice": disc_dice_sum / N,
    }


def train_one_epoch_beal(
    model,
    discriminator,
    loader,
    criterion,
    disc_loss_fn,
    optimizer_g,
    optimizer_d,
    scaler,
    device,
    batch_scheduler=None,
):
    """BEAL training loop with alternating generator/discriminator updates."""
    model.train()
    discriminator.train()
    running = {"total": 0.0}
    cup_dice_sum = 0.0
    disc_dice_sum = 0.0
    n_batches = 0

    for batch in loader:
        images = batch["image"].to(device)
        cup_mask = batch["cup_mask"].to(device)
        disc_mask = batch["disc_mask"].to(device)

        # === Step 1: Update Discriminator ===================================
        optimizer_d.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=(device.type == "cuda")):
            with torch.no_grad():
                model_out = model.forward_with_boundary(images)
                seg_logits = model_out["seg_logits"]
                pred_masks = torch.sigmoid(seg_logits).detach()

            gt_masks = torch.cat([cup_mask, disc_mask], dim=1)  # (B, 2, H, W)

            d_real = discriminator(gt_masks)
            d_fake = discriminator(pred_masks)
            d_loss = disc_loss_fn(d_real, d_fake)

        if torch.isfinite(d_loss):
            scaler.scale(d_loss).backward()
            scaler.unscale_(optimizer_d)
            nn.utils.clip_grad_norm_(
                discriminator.parameters(), max_norm=GRAD_CLIP_MAX_NORM
            )
            scaler.step(optimizer_d)
            scaler.update()
        else:
            optimizer_d.zero_grad(set_to_none=True)

        # === Step 2: Update Generator ======================================
        optimizer_g.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=(device.type == "cuda")):
            model_out = model.forward_with_boundary(images)
            seg_logits = model_out["seg_logits"]
            pred_masks = torch.sigmoid(seg_logits)

            # Get discriminator score for generator adversarial loss
            d_score = discriminator(pred_masks)

            loss, loss_dict = criterion(
                model_out, cup_mask, disc_mask, disc_score=d_score
            )

        if not torch.isfinite(loss):
            optimizer_g.zero_grad(set_to_none=True)
            if batch_scheduler is not None:
                batch_scheduler.step()
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer_g)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
        scaler.step(optimizer_g)
        scaler.update()

        if batch_scheduler is not None:
            batch_scheduler.step()

        running["total"] += loss_dict["total"]
        with torch.no_grad():
            cup_dice_sum += dice_score(seg_logits[:, 0:1], cup_mask).item()
            disc_dice_sum += dice_score(seg_logits[:, 1:2], disc_mask).item()
        n_batches += 1

    N = max(n_batches, 1)
    return {
        "total": running["total"] / N,
        "cup_dice": cup_dice_sum / N,
        "disc_dice": disc_dice_sum / N,
    }


def train_one_epoch_dofe(
    model, loader, criterion, optimizer, scaler, device, batch_scheduler=None
):
    """DoFE training loop with domain classification."""
    model.train()
    running = {"total": 0.0}
    cup_dice_sum = 0.0
    disc_dice_sum = 0.0
    n_batches = 0

    for batch in loader:
        images = batch["image"].to(device)
        cup_mask = batch["cup_mask"].to(device)
        disc_mask = batch["disc_mask"].to(device)
        domain_labels = batch.get("domain")
        if domain_labels is not None:
            domain_labels = domain_labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=(device.type == "cuda")):
            model_out = model.forward_with_domain(images)
            seg_logits = model_out["seg_logits"]
            loss, loss_dict = criterion(
                model_out, cup_mask, disc_mask, domain_labels=domain_labels
            )

        if not torch.isfinite(loss):
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

        running["total"] += loss_dict["total"]
        with torch.no_grad():
            cup_dice_sum += dice_score(seg_logits[:, 0:1], cup_mask).item()
            disc_dice_sum += dice_score(seg_logits[:, 1:2], disc_mask).item()
        n_batches += 1

    N = max(n_batches, 1)
    return {
        "total": running["total"] / N,
        "cup_dice": cup_dice_sum / N,
        "disc_dice": disc_dice_sum / N,
    }


@torch.no_grad()
def validate(model, loader, criterion, device, model_name="standard"):
    """Validation for all models. Uses standard forward() for consistency."""
    model.eval()
    running = {"total": 0.0}
    cup_dice_sum = 0.0
    disc_dice_sum = 0.0
    cup_iou_sum = 0.0
    disc_iou_sum = 0.0
    n_batches = 0

    # Use the matching validation loss
    val_criterion = BaselineLoss().to(device)

    for batch in loader:
        images = batch["image"].to(device)
        cup_mask = batch["cup_mask"].to(device)
        disc_mask = batch["disc_mask"].to(device)

        logits = model(images)  # All models return (B, 2, H, W) from forward()
        loss, loss_dict = val_criterion(logits, cup_mask, disc_mask)
        if not torch.isfinite(loss):
            continue

        running["total"] += loss_dict["total"]
        cup_dice_sum += dice_score(logits[:, 0:1], cup_mask).item()
        disc_dice_sum += dice_score(logits[:, 1:2], disc_mask).item()
        cup_iou_sum += iou_score(logits[:, 0:1], cup_mask).item()
        disc_iou_sum += iou_score(logits[:, 1:2], disc_mask).item()
        n_batches += 1

    N = max(n_batches, 1)
    return {
        "total": running["total"] / N,
        "cup_dice": cup_dice_sum / N,
        "disc_dice": disc_dice_sum / N,
        "cup_iou": cup_iou_sum / N,
        "disc_iou": disc_iou_sum / N,
    }


# ==============================================================================
# CHECKPOINT
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
    print(f"  -> Saved: {path} (cup_dice={best_dice:.4f})")


def _save_resume(
    model, optimizer, epoch, best_dice, scaler, model_name, extra_state=None
):
    path = os.path.join(CHECKPOINT_DIR, f"{model_name}_resume.pth")
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_cup_dice": best_dice,
    }
    if extra_state:
        state.update(extra_state)
    torch.save(state, path)


# ==============================================================================
# MAIN
# ==============================================================================


def train(model_name, resume=False):
    warnings.filterwarnings("ignore", message=".*lr_scheduler.*")
    warnings.filterwarnings("ignore", message=".*ShiftScaleRotate.*")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device(DEVICE)
    ckpt_dir = os.path.join(CHECKPOINT_DIR, model_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Data loading (with pseudo-domain support for DoFE) ────────────────
    pseudo_domain_labels = None
    if model_name == "dofe":
        from pseudo_domains import load_pseudo_domain_labels
        from config import DATA_CSV

        cache_path = os.path.join(os.path.dirname(__file__), "pseudo_domains.npy")
        pseudo_domain_labels = load_pseudo_domain_labels(
            DATA_CSV, N_PSEUDO_DOMAINS, cache_path=cache_path
        )

    print("Loading data...")
    train_loader, val_loader = get_dataloaders(
        BATCH_SIZE, NUM_WORKERS, pseudo_domain_labels=pseudo_domain_labels
    )

    # ── Model ─────────────────────────────────────────────────────────────
    print(f"Building {model_name}...")
    model = build_model(model_name).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {total_params:,}")

    # ── Loss & Optimiser ──────────────────────────────────────────────────
    criterion = build_loss_fn(model_name, device)
    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    # BEAL-specific: discriminator + its optimiser
    discriminator = None
    optimizer_d = None
    disc_loss_fn = None
    if model_name == "beal":
        from beal import Discriminator
        from beal_loss import DiscriminatorLoss

        discriminator = Discriminator(in_channels=2).to(device)
        optimizer_d = optim.Adam(
            discriminator.parameters(), lr=DISC_LR, betas=(0.5, 0.999)
        )
        disc_loss_fn = DiscriminatorLoss()
        disc_params = sum(
            p.numel() for p in discriminator.parameters() if p.requires_grad
        )
        print(f"  Discriminator params: {disc_params:,}")

    best_cup_dice = 0.0
    start_epoch = 0

    if resume:
        rpath = os.path.join(CHECKPOINT_DIR, f"{model_name}_resume.pth")
        if os.path.isfile(rpath):
            ckpt = torch.load(rpath, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scaler.load_state_dict(ckpt["scaler_state_dict"])
            best_cup_dice = ckpt["best_cup_dice"]
            start_epoch = ckpt["epoch"] + 1
            if discriminator is not None and "disc_state_dict" in ckpt:
                discriminator.load_state_dict(ckpt["disc_state_dict"])
            if optimizer_d is not None and "optimizer_d_state_dict" in ckpt:
                optimizer_d.load_state_dict(ckpt["optimizer_d_state_dict"])
            print(f"  Resuming from epoch {start_epoch}")

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

    if HAS_WANDB:
        wandb.init(
            project="baselines",
            name=model_name,
            config={
                "model": model_name,
                "epochs": NUM_EPOCHS,
                "lr": MAX_LR,
            },
            mode="offline",
        )

    # Determine loss description
    if model_name == "beal":
        loss_desc = "BCE + Dice + Boundary + Adversarial (BEAL)"
    elif model_name == "dofe":
        loss_desc = "BCE + Dice + Domain Classification (DoFE)"
    else:
        loss_desc = "BCE + Dice (standard)"

    print(f"\n{'=' * 60}")
    print(f"  Training {model_name} — {NUM_EPOCHS} epochs")
    print(f"  Loss: {loss_desc}")
    print(f"{'=' * 60}")

    for epoch in range(start_epoch, NUM_EPOCHS):
        t0 = time.time()

        # ── Route to correct training function ────────────────────────────
        if model_name == "beal":
            train_m = train_one_epoch_beal(
                model,
                discriminator,
                train_loader,
                criterion,
                disc_loss_fn,
                optimizer,
                optimizer_d,
                scaler,
                device,
                batch_scheduler=scheduler,
            )
        elif model_name == "dofe":
            train_m = train_one_epoch_dofe(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler,
                device,
                batch_scheduler=scheduler,
            )
        else:
            train_m = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler,
                device,
                batch_scheduler=scheduler,
            )

        val_m = validate(model, val_loader, criterion, device, model_name=model_name)
        dt = time.time() - t0

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"[E{epoch + 1:02d}/{NUM_EPOCHS}] "
            f"lr={lr:.2e} | "
            f"loss={val_m['total']:.4f} | "
            f"cup={val_m['cup_dice']:.4f} "
            f"disc={val_m['disc_dice']:.4f} | "
            f"{dt:.1f}s"
        )

        if val_m["cup_dice"] > best_cup_dice:
            best_cup_dice = val_m["cup_dice"]
            _save_checkpoint(
                model,
                optimizer,
                epoch,
                best_cup_dice,
                os.path.join(ckpt_dir, "best_model.pth"),
            )

        # Build extra state for resume
        extra = {}
        if discriminator is not None:
            extra["disc_state_dict"] = discriminator.state_dict()
        if optimizer_d is not None:
            extra["optimizer_d_state_dict"] = optimizer_d.state_dict()

        _save_resume(
            model,
            optimizer,
            epoch,
            best_cup_dice,
            scaler,
            model_name,
            extra_state=extra,
        )

    _save_checkpoint(
        model,
        optimizer,
        NUM_EPOCHS - 1,
        best_cup_dice,
        os.path.join(ckpt_dir, "final_model.pth"),
    )

    print(f"\nDone! Best cup Dice: {best_cup_dice:.4f}")

    # Test
    _evaluate_test(model, model_name, criterion, device, ckpt_dir)

    if HAS_WANDB:
        wandb.finish()


def _evaluate_test(model, model_name, criterion, device, ckpt_dir):
    """Full evaluation: held-out test + external generalization (DRISHTI, RIM)."""
    from inference import evaluate, print_results
    from dataset import get_external_dataloader

    best_path = os.path.join(ckpt_dir, "best_model.pth")
    if not os.path.isfile(best_path):
        return

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    # ── Held-out test set ────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  FULL EVALUATION — {model_name}")
    print(f"{'=' * 70}")

    test_loader = get_test_dataloader(BATCH_SIZE, NUM_WORKERS)
    r = evaluate(model, test_loader, device, f"{model_name} Test")
    print_results(r)

    # ── External generalization (zero-shot) ──────────────────────────────
    ext_datasets = [
        ("RIM", "../Map/Corrected_testrim.csv"),
        ("Dristi", "../Map/Corrected_DristiTest.csv"),
    ]
    for name, csv_path in ext_datasets:
        abs_path = os.path.join(os.path.dirname(__file__), csv_path)
        if not os.path.isfile(abs_path):
            print(f"[SKIP] {name}: not found {abs_path}")
            continue
        print(f"\n>>> External generalization: {name}")
        loader = get_external_dataloader(abs_path, BATCH_SIZE, NUM_WORKERS)
        r = evaluate(model, loader, device, f"{model_name} {name}")
        print_results(r)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline Training")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=ALL_MODELS,
        help="Which baseline model to train",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(args.model, resume=args.resume)
