# config.py
#!/usr/bin/env python3
"""
Ablation Study Configuration — NPS-Net Component Ablation.

Shared configuration for all ablation variants (B2, B3, B4).
Identical hyperparameters to the full NPS-Net V3.1 for fair comparison.

The only differences across variants are:
  - Which architectural components are active
  - Which loss terms are active
  - Where checkpoints are saved
"""

import os
import torch

# ==============================================================================
# DATA
# ==============================================================================
DATA_CSV = r"/home/rojan/Desktop/phoenix/Map/Glaucoma_Classification.csv"

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15

# ==============================================================================
# MODEL — Polar Grid
# ==============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 512
N_CHANNELS = 3

# Polar sampling resolution
N_THETA = 360  # angular samples
N_RHO = 256  # radial samples

# Encoder feature channels (polar UNet) — IDENTICAL across all variants
ENCODER_FEATURES = [64, 128, 256, 512]

# ==============================================================================
# DENSE MONOTONE PATH
# ==============================================================================
TAU = 0.03  # sigmoid temperature for rendering

# ==============================================================================
# BAND-ANCHORED DISC HEAD  (B5 only — included for reference)
# ==============================================================================
DISC_R_MIN = 0.28
DISC_R_MAX = 0.42
DISC_TAU_MIN = 0.03
DISC_TAU_MAX = 0.10
DISC_BAND_WIDTH = 0.08
DISC_USE_EDGE = True

# ==============================================================================
# SHAPE PRIOR BRANCH  (B4+ only)
# ==============================================================================
SHAPE_FEATURES = 128
SOFTARGMAX_TEMPERATURE = 0.5
TAU_SHAPE = 0.03
PRIOR_SCALE_INIT = 0.1

# ==============================================================================
# BOUNDARY DISTRIBUTION TARGETS
# ==============================================================================
SIGMA_Q = 1.5

# ==============================================================================
# TRAINING — IDENTICAL ACROSS ALL VARIANTS
# ==============================================================================
BATCH_SIZE = 4
NUM_WORKERS = 4
NUM_EPOCHS = 80
MAX_LR = 3e-4  # OneCycleLR peak learning rate

# Staged training schedule (used by B4; B2/B3 ignore these)
STAGE_A_END = 20
STAGE_B_END = 30

# ==============================================================================
# LOSS WEIGHTS — IDENTICAL TO FULL MODEL
# ==============================================================================
# Always active (B2, B3, B4):
LAMBDA_CART = 1.0  # (1) Cartesian mask: BCE + Dice
LAMBDA_POLAR = 0.7  # (2) Polar mask: BCE + Dice
LAMBDA_RIM = 0.5  # (3) Rim-profile loss on dense-mask radii

# Active from Stage B (B4 only):
LAMBDA_DIST = 0.3  # (4) Shape distribution: soft cross-entropy
LAMBDA_SHAPE = 0.5  # (5) Shape radial regression: SmoothL1

# Active from Stage C (B4 only):
LAMBDA_CONS = 0.3  # (6) Confidence-weighted consistency

# Active from Stage B (B4 only):
LAMBDA_SMOOTH = 0.05  # (7) Circular smoothness (shape prior only)

# SmoothL1 delta
SMOOTHL1_DELTA = 0.01

# ==============================================================================
# GENERAL
# ==============================================================================
GRAD_CLIP_MAX_NORM = 1.0
SEED = 42
WEIGHT_DECAY = 1e-4

# ==============================================================================
# PATHS / LOGGING — VARIANT-AWARE
# ==============================================================================
ABLATION_DIR = os.path.dirname(os.path.abspath(__file__))

# Checkpoint directories per variant (root-level checkpoints/)
CHECKPOINT_DIRS = {
    "b2": os.path.join(ABLATION_DIR, "checkpoints", "b2"),
    "b3": os.path.join(ABLATION_DIR, "checkpoints", "b3"),
    "b4": os.path.join(ABLATION_DIR, "checkpoints", "b4"),
}

# B1 checkpoints (from baselines/)
B1_CHECKPOINT_DIR = os.path.join(ABLATION_DIR, "checkpoints", "baselines")

PREDICTIONS_DIR = os.path.join(ABLATION_DIR, "predictions")
WANDB_PROJECT = "nps_net_ablation"
WANDB_ENTITY = None


def get_checkpoint_dir(variant):
    """Return checkpoint directory for a given ablation variant."""
    if variant in CHECKPOINT_DIRS:
        os.makedirs(CHECKPOINT_DIRS[variant], exist_ok=True)
        return CHECKPOINT_DIRS[variant]
    raise ValueError(
        f"Unknown variant: {variant}. Expected one of {list(CHECKPOINT_DIRS.keys())}"
    )
