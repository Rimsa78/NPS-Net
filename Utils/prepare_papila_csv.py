"""
Prepare the cropped Papila dataset for the baseline inference pipeline.

Steps:
    1. Split each cropped mask (.png) into separate disc and cup binary masks (.png)
    2. Compute vertical CDR from the masks
    3. Generate a CSV file (Raw, Disk, Cup, CDR, Type) compatible with
       Comparision/dataset.py

Mask encoding (Papila):
    255 = background
    128 = optic disc (rim)
      0 = optic cup

Disc = all pixels < 255  (cup + disc rim)
Cup  = all pixels == 0

Output:
    Dataset/Papila/cropped_disc_masks/   — binary disc masks (white on black)
    Dataset/Papila/cropped_cup_masks/    — binary cup masks (white on black)
    Map/Corrected_papila.csv             — CSV index file
"""

import os
import glob
import csv

import numpy as np
from PIL import Image


BASE_DIR = os.path.join("Dataset", "Papila")
CROPPED_IMAGES = os.path.join(BASE_DIR, "cropped_images")
CROPPED_MASKS = os.path.join(BASE_DIR, "cropped_masks")
OUT_DISC = os.path.join(BASE_DIR, "cropped_disc_masks")
OUT_CUP = os.path.join(BASE_DIR, "cropped_cup_masks")
CSV_PATH = os.path.join("Map", "Corrected_papila.csv")

# Absolute base for CSV paths (to match the project convention)
ABS_BASE = os.path.abspath(".")

os.makedirs(OUT_DISC, exist_ok=True)
os.makedirs(OUT_CUP, exist_ok=True)


def compute_vcdr(cup_arr, disc_arr):
    """Compute vertical cup-to-disc ratio from binary masks."""
    cup_rows = np.where(cup_arr.any(axis=1))[0]
    disc_rows = np.where(disc_arr.any(axis=1))[0]
    if len(disc_rows) == 0:
        return 0.0
    disc_h = disc_rows[-1] - disc_rows[0] + 1
    if len(cup_rows) == 0:
        return 0.0
    cup_h = cup_rows[-1] - cup_rows[0] + 1
    return cup_h / disc_h


def main():
    mask_paths = sorted(glob.glob(os.path.join(CROPPED_MASKS, "*.png")))
    print(f"Found {len(mask_paths)} cropped masks")

    rows = []
    for mask_path in mask_paths:
        stem = os.path.splitext(os.path.basename(mask_path))[0]

        # Corresponding cropped image
        img_path = os.path.join(CROPPED_IMAGES, stem + ".jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(CROPPED_IMAGES, stem + ".png")
        if not os.path.exists(img_path):
            print(f"  [SKIP] No image for {stem}")
            continue

        # Load mask
        mask = np.array(Image.open(mask_path))
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]

        # Create binary disc mask (everything that's not background)
        disc_mask = (mask < 255).astype(np.uint8) * 255

        # Create binary cup mask (value == 0 in original)
        cup_mask = (mask == 0).astype(np.uint8) * 255

        # Save
        disc_out = os.path.join(OUT_DISC, stem + ".png")
        cup_out = os.path.join(OUT_CUP, stem + ".png")
        Image.fromarray(disc_mask).save(disc_out)
        Image.fromarray(cup_mask).save(cup_out)

        # CDR
        cdr = compute_vcdr(cup_mask > 0, disc_mask > 0)

        # Use absolute paths (matching project convention)
        abs_img = os.path.join(ABS_BASE, img_path)
        abs_disc = os.path.join(ABS_BASE, disc_out)
        abs_cup = os.path.join(ABS_BASE, cup_out)

        rows.append({
            'Raw': abs_img,
            'Disk': abs_disc,
            'Cup': abs_cup,
            'CDR': f"{cdr:.6f}",
            'Type': 'Unknown',
        })

    # Write CSV
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Raw', 'Disk', 'Cup', 'CDR', 'Type'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. {len(rows)} entries written to {CSV_PATH}")
    print(f"Disc masks → {OUT_DISC}")
    print(f"Cup masks  → {OUT_CUP}")


if __name__ == "__main__":
    main()
