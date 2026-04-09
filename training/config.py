# config.py
"""
Shared configuration for all baseline models (B1–B4).

Models: VanillaUNet, AttentionUNet, ResUNet, TransUNet, PolarUNet
Loss: BCE + Dice (standard, no special shape losses)
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
# MODEL
# ==============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 512
N_CHANNELS = 3
FEATURES = [64, 128, 256, 512]

# Polar-UNet specific
N_THETA = 360
N_RHO = 256

# NPS-Net specific
ENCODER_FEATURES = [64, 128, 256, 512]
SHAPE_FEATURES = 128
SOFTARGMAX_TEMPERATURE = 0.5
TAU_SHAPE = 0.03
PRIOR_SCALE_INIT = 0.1

# ==============================================================================
# TRAINING
# ==============================================================================
BATCH_SIZE = 4
NUM_WORKERS = 4
NUM_EPOCHS = 50
MAX_LR = 3e-4

# ==============================================================================
# LOSS
# ==============================================================================
LAMBDA_CUP = 2.0  # Cup gets 2× disc weight (harder target)
LAMBDA_DISC = 1.0

# BEAL-specific loss weights
LAMBDA_BND = 0.5  # Boundary attention loss weight
LAMBDA_ADV = 0.01  # Adversarial generator loss weight
DISC_LR = 1e-4  # Discriminator learning rate (separate from generator)

# DoFE-specific
LAMBDA_DOM = 0.1  # Domain classification loss weight
N_PSEUDO_DOMAINS = 4  # Number of pseudo-domain clusters

# ==============================================================================
# GENERAL
# ==============================================================================
GRAD_CLIP_MAX_NORM = 1.0
SEED = 42
WEIGHT_DECAY = 1e-4

# ==============================================================================
# PATHS
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "baselines")
PREDICTIONS_DIR = "predictions"

# Ablation checkpoint directories
ABLATION_DIR = os.path.join(PROJECT_ROOT)
CHECKPOINT_DIRS = {
    "b2": os.path.join(PROJECT_ROOT, "checkpoints", "b2"),
    "b3": os.path.join(PROJECT_ROOT, "checkpoints", "b3"),
    "b4": os.path.join(PROJECT_ROOT, "checkpoints", "b4"),
}

B1_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "baselines")


def get_checkpoint_dir(variant):
    """Return checkpoint directory for a given ablation variant."""
    if variant in CHECKPOINT_DIRS:
        os.makedirs(CHECKPOINT_DIRS[variant], exist_ok=True)
        return CHECKPOINT_DIRS[variant]
    raise ValueError(
        f"Unknown variant: {variant}. Expected one of {list(CHECKPOINT_DIRS.keys())}"
    )
