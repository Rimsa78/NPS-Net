# dataset.py
"""
Shared dataset for all baseline models.

Simple: returns image + cup_mask + disc_mask + cdr + label.
No polar GT targets (those are NPS-Net specific).
"""

import os
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from config import (
    DATA_CSV, IMAGE_SIZE,
    TRAIN_RATIO, VALID_RATIO, TEST_RATIO, SEED,
)


def clahe_enhance(image, clip=2.0, grid=8):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def load_and_resize(path, size, is_mask=False):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if is_mask else cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {path}")
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    return cv2.resize(img, (size, size), interpolation=interp)


LABEL_MAP = {'Healthy': 0, 'Glaucoma': 1}


def get_augmentation(is_train=True):
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


class BaselineDataset(Dataset):
    """Simple dataset returning image + cup/disc masks.

    Optionally includes pseudo-domain labels (for DoFE training).
    """

    def __init__(self, df, is_train=True, image_size=IMAGE_SIZE,
                 pseudo_domain_labels=None):
        self.df = df.reset_index(drop=True)
        self.is_train = is_train
        self.image_size = image_size
        self.aug = get_augmentation(is_train)
        self.pseudo_domain_labels = pseudo_domain_labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image     = load_and_resize(row['Raw'], self.image_size, is_mask=False)
        disc_mask = load_and_resize(row['Disk'], self.image_size, is_mask=True)
        cup_mask  = load_and_resize(row['Cup'], self.image_size, is_mask=True)

        image = clahe_enhance(image)

        if self.aug is not None and self.is_train:
            augmented = self.aug(image=image, mask0=cup_mask, mask1=disc_mask)
            image     = augmented['image']
            cup_mask  = augmented['mask0']
            disc_mask = augmented['mask1']

        cdr = float(row['CDR']) if pd.notna(row.get('CDR', None)) else 0.5
        label = LABEL_MAP.get(row.get('Type', 'Healthy'), 0)

        cup_mask  = (cup_mask > 127).astype(np.float32)
        disc_mask = (disc_mask > 127).astype(np.float32)

        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)

        sample = {
            'image':     image,
            'cup_mask':  torch.from_numpy(cup_mask).unsqueeze(0),
            'disc_mask': torch.from_numpy(disc_mask).unsqueeze(0),
            'cdr':       torch.tensor(cdr, dtype=torch.float32),
            'label':     label,
        }

        # Pseudo-domain label for DoFE training (optional)
        if self.pseudo_domain_labels is not None:
            sample['domain'] = torch.tensor(
                self.pseudo_domain_labels[idx], dtype=torch.long)

        return sample


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_csv():
    df = pd.read_csv(DATA_CSV)
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
    if n_before != len(df):
        print(f"[dataset] Filtered {n_before - len(df)} rows. Remaining: {len(df)}")
    return df


def get_dataloaders(batch_size, num_workers=4, pseudo_domain_labels=None):
    df = load_csv()
    val_test_ratio = VALID_RATIO + TEST_RATIO
    train_df, valtest_df = train_test_split(
        df, test_size=val_test_ratio, stratify=df['Type'], random_state=SEED)
    test_relative = TEST_RATIO / val_test_ratio
    val_df, _ = train_test_split(
        valtest_df, test_size=test_relative, stratify=valtest_df['Type'], random_state=SEED)
    print(f"[dataset] Train: {len(train_df)}, Val: {len(val_df)}")

    # Map pseudo-domain labels to train/val indices if provided
    train_pd = None
    if pseudo_domain_labels is not None:
        train_indices = train_df.index.tolist()
        train_pd = pseudo_domain_labels[train_indices]

    train_ds = BaselineDataset(train_df, is_train=True,
                               pseudo_domain_labels=train_pd)
    val_ds   = BaselineDataset(val_df, is_train=False)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False)
    return train_loader, val_loader


def get_test_dataloader(batch_size=4, num_workers=4):
    df = load_csv()
    val_test_ratio = VALID_RATIO + TEST_RATIO
    _, valtest_df = train_test_split(
        df, test_size=val_test_ratio, stratify=df['Type'], random_state=SEED)
    test_relative = TEST_RATIO / val_test_ratio
    _, test_df = train_test_split(
        valtest_df, test_size=test_relative, stratify=valtest_df['Type'], random_state=SEED)
    print(f"[dataset] Test: {len(test_df)}")

    test_ds = BaselineDataset(test_df, is_train=False)
    return torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False)


def get_external_dataloader(csv_path, batch_size=4, num_workers=4):
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
        print(f"[dataset] External: filtered {n_before - len(df)} missing rows")
    print(f"[dataset] External: {len(df)} samples from {os.path.basename(csv_path)}")

    ds = BaselineDataset(df, is_train=False)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False)
