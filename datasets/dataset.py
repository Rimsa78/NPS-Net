# dataset.py
#!/usr/bin/env python3
"""
Ablation Study — Dataset.

Identical to ThreeSixty/dataset.py. Self-contained copy so the ablation
folder is fully independent.

Reads Corrected_Training.csv (columns: Raw, Disk, Cup, CDR, Type, Class).
Each sample returns Cartesian masks PLUS polar-domain ground truth:

    image           : (3, H, W) CLAHE-enhanced, normalised
    cup_mask        : (1, H, W) binary
    disc_mask       : (1, H, W) binary
    r_c_gt          : (N_θ,) GT cup radii
    r_d_gt          : (N_θ,) GT disc radii
    alpha_gt        : (N_θ,) GT cup/disc ratio α = r_c / r_d
    q_d             : (N_ρ, N_θ) Gaussian-smoothed disc-boundary distribution
    q_alpha         : (N_ρ, N_θ) Gaussian-smoothed alpha distribution
    Y_c_polar_gt    : (N_ρ, N_θ) binary polar cup mask
    Y_d_polar_gt    : (N_ρ, N_θ) binary polar disc mask
    cdr             : scalar float
    label           : int (glaucoma class index)
"""

import math
import os
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from config import (
    DATA_CSV, IMAGE_SIZE, N_THETA, N_RHO, SIGMA_Q,
    TRAIN_RATIO, VALID_RATIO, TEST_RATIO, SEED,
)


# ==============================================================================
# HELPERS
# ==============================================================================

def clahe_enhance(image, clip=2.0, grid=8):
    """Apply CLAHE on the L channel of LAB, return 3-channel BGR."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def load_and_resize(path, size, is_mask=False):
    """Load image/mask and resize to (size, size)."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if is_mask else cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {path}")
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    return cv2.resize(img, (size, size), interpolation=interp)


# ==============================================================================
# POLAR GT COMPUTATION
# ==============================================================================

def compute_gt_radii(mask_binary, n_theta, n_rho, image_size):
    """Extract GT radial boundary function from a binary mask."""
    H = W = image_size
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0
    R = min(H, W) / 2.0

    thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    rhos = np.linspace(0, 1, n_rho)

    r_gt = np.zeros(n_theta, dtype=np.float32)

    for i, theta in enumerate(thetas):
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        px = cx + R * rhos * cos_t
        py = cy + R * rhos * sin_t

        px_int = np.clip(np.round(px).astype(int), 0, W - 1)
        py_int = np.clip(np.round(py).astype(int), 0, H - 1)

        samples = mask_binary[py_int, px_int]

        fg_indices = np.where(samples > 0.5)[0]
        if len(fg_indices) > 0:
            r_gt[i] = rhos[fg_indices[-1]]
        else:
            r_gt[i] = 0.0

    return r_gt


def compute_alpha_gt(r_c_gt, r_d_gt, eps=1e-6):
    """Compute cup/disc ratio target α*(θ)."""
    alpha_gt = r_c_gt / (r_d_gt + eps)
    return np.clip(alpha_gt, 0.0, 1.0).astype(np.float32)


def compute_soft_distribution(r_gt, n_rho, sigma_q):
    """Compute Gaussian-smoothed target distribution."""
    rhos = np.linspace(0, 1, n_rho)

    diff_sq = (rhos[:, None] - r_gt[None, :]) ** 2

    bin_width = 1.0 / (n_rho - 1)
    sigma_norm = sigma_q * bin_width

    log_q = -diff_sq / (2.0 * sigma_norm ** 2)

    log_q_max = log_q.max(axis=0, keepdims=True)
    q = np.exp(log_q - log_q_max)
    q = q / (q.sum(axis=0, keepdims=True) + 1e-10)

    return q.astype(np.float32)


def compute_polar_gt_masks(r_c_gt, r_d_gt, n_rho):
    """Construct binary polar GT masks."""
    rhos = np.linspace(0, 1, n_rho)

    Y_c_polar = (rhos[:, None] <= r_c_gt[None, :]).astype(np.float32)
    Y_d_polar = (rhos[:, None] <= r_d_gt[None, :]).astype(np.float32)

    return Y_c_polar, Y_d_polar


# ==============================================================================
# LABEL ENCODING
# ==============================================================================

LABEL_MAP = {'Healthy': 0, 'Glaucoma': 1}


# ==============================================================================
# AUGMENTATIONS
# ==============================================================================

def get_augmentation(is_train=True):
    """Return augmentation pipeline."""
    try:
        import albumentations as A
    except ImportError:
        return None

    if is_train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                               rotate_limit=15, p=0.5,
                               border_mode=cv2.BORDER_CONSTANT),
            A.RandomBrightnessContrast(brightness_limit=0.15,
                                        contrast_limit=0.15, p=0.4),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GaussNoise(p=0.2),
        ], additional_targets={'mask0': 'mask', 'mask1': 'mask'})
    return None


# ==============================================================================
# DATASET
# ==============================================================================

class NPSDataset(Dataset):
    """Dataset for ablation study.

    Returns all GT targets needed by any variant:
        - q_d, q_alpha: for shape prior distribution supervision (B4)
        - r_c_gt, r_d_gt: for rim, radial, and consistency losses
        - Y_c_polar_gt, Y_d_polar_gt: for dense polar mask supervision
        - cup_mask, disc_mask: for Cartesian mask supervision
    """

    def __init__(self, df, is_train=True, image_size=IMAGE_SIZE,
                 n_theta=N_THETA, n_rho=N_RHO, sigma_q=SIGMA_Q):
        self.df = df.reset_index(drop=True)
        self.is_train = is_train
        self.image_size = image_size
        self.n_theta = n_theta
        self.n_rho = n_rho
        self.sigma_q = sigma_q
        self.aug = get_augmentation(is_train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        raw_path  = row['Raw']
        disk_path = row['Disk']
        cup_path  = row['Cup']

        image     = load_and_resize(raw_path, self.image_size, is_mask=False)
        disc_mask = load_and_resize(disk_path, self.image_size, is_mask=True)
        cup_mask  = load_and_resize(cup_path, self.image_size, is_mask=True)

        image = clahe_enhance(image)

        if self.aug is not None and self.is_train:
            augmented = self.aug(image=image, mask0=cup_mask, mask1=disc_mask)
            image     = augmented['image']
            cup_mask  = augmented['mask0']
            disc_mask = augmented['mask1']

        cdr = float(row['CDR']) if pd.notna(row.get('CDR', None)) else 0.5
        label = LABEL_MAP.get(row.get('Type', 'Healthy'), 0)

        cup_mask_bin  = (cup_mask > 127).astype(np.float32)
        disc_mask_bin = (disc_mask > 127).astype(np.float32)

        r_c_gt = compute_gt_radii(
            cup_mask_bin, self.n_theta, self.n_rho, self.image_size)
        r_d_gt = compute_gt_radii(
            disc_mask_bin, self.n_theta, self.n_rho, self.image_size)

        # Ensure nesting in GT
        r_d_gt = np.maximum(r_d_gt, r_c_gt)

        alpha_gt = compute_alpha_gt(r_c_gt, r_d_gt)

        q_d     = compute_soft_distribution(r_d_gt,   self.n_rho, self.sigma_q)
        q_alpha = compute_soft_distribution(alpha_gt, self.n_rho, self.sigma_q)

        Y_c_polar_gt, Y_d_polar_gt = compute_polar_gt_masks(
            r_c_gt, r_d_gt, self.n_rho)

        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)

        return {
            'image':         image,
            'cup_mask':      torch.from_numpy(cup_mask_bin).unsqueeze(0),
            'disc_mask':     torch.from_numpy(disc_mask_bin).unsqueeze(0),
            'r_c_gt':        torch.from_numpy(r_c_gt),
            'r_d_gt':        torch.from_numpy(r_d_gt),
            'alpha_gt':      torch.from_numpy(alpha_gt),
            'q_d':           torch.from_numpy(q_d),
            'q_alpha':       torch.from_numpy(q_alpha),
            'Y_c_polar_gt':  torch.from_numpy(Y_c_polar_gt),
            'Y_d_polar_gt':  torch.from_numpy(Y_d_polar_gt),
            'cdr':           torch.tensor(cdr, dtype=torch.float32),
            'label':         label,
        }


# ==============================================================================
# DATA LOADING UTILITIES
# ==============================================================================

def load_csv():
    """Load and validate the training CSV."""
    csv_path = DATA_CSV
    df = pd.read_csv(csv_path)

    required = ['Raw', 'Disk', 'Cup', 'CDR', 'Type']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    valid_mask = (
        df['Raw'].apply(lambda x: os.path.isfile(str(x))) &
        df['Disk'].apply(lambda x: os.path.isfile(str(x))) &
        df['Cup'].apply(lambda x: os.path.isfile(str(x)))
    )
    n_before = len(df)
    df = df[valid_mask].reset_index(drop=True)
    n_after = len(df)
    if n_before != n_after:
        print(f"[dataset] Filtered {n_before - n_after} rows with missing files. "
              f"Remaining: {n_after}")

    return df


def get_dataloaders(batch_size, num_workers=4):
    """Create train/val DataLoaders with stratified split."""
    df = load_csv()

    val_test_ratio = VALID_RATIO + TEST_RATIO
    train_df, valtest_df = train_test_split(
        df, test_size=val_test_ratio,
        stratify=df['Type'], random_state=SEED,
    )
    test_relative = TEST_RATIO / val_test_ratio
    val_df, test_df = train_test_split(
        valtest_df, test_size=test_relative,
        stratify=valtest_df['Type'], random_state=SEED,
    )
    print(f"[dataset] Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    train_ds = NPSDataset(train_df, is_train=True)
    val_ds   = NPSDataset(val_df,   is_train=False)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )

    return train_loader, val_loader


def get_test_dataloader(batch_size=4, num_workers=4):
    """Return test DataLoader from the held-out test split."""
    df = load_csv()

    val_test_ratio = VALID_RATIO + TEST_RATIO
    _, valtest_df = train_test_split(
        df, test_size=val_test_ratio,
        stratify=df['Type'], random_state=SEED,
    )
    test_relative = TEST_RATIO / val_test_ratio
    _, test_df = train_test_split(
        valtest_df, test_size=test_relative,
        stratify=valtest_df['Type'], random_state=SEED,
    )
    print(f"[dataset] Test split: {len(test_df)} samples")

    test_ds = NPSDataset(test_df, is_train=False)
    return torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )


def get_external_dataloader(csv_path, batch_size=4, num_workers=4):
    """Load an external CSV and return a DataLoader for inference."""
    df = pd.read_csv(csv_path)

    required = ['Raw', 'Disk', 'Cup', 'CDR', 'Type']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    valid_mask = (
        df['Raw'].apply(lambda x: os.path.isfile(str(x))) &
        df['Disk'].apply(lambda x: os.path.isfile(str(x))) &
        df['Cup'].apply(lambda x: os.path.isfile(str(x)))
    )
    n_before = len(df)
    df = df[valid_mask].reset_index(drop=True)
    if len(df) < n_before:
        print(f"[dataset] External CSV: filtered {n_before - len(df)} missing rows")
    print(f"[dataset] External CSV: {len(df)} samples from {os.path.basename(csv_path)}")

    ds = NPSDataset(df, is_train=False)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
