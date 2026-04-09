"""
Crop REFUGE fundus images and masks around the optic disc region.

The crop area is 2x the size of the optic disc bounding box, centred on the disc centre.
Both the raw image and its corresponding mask are cropped with identical coordinates.

Mask encoding (REFUGE / RIGA convention):
    255 = background
    128 = optic disc (rim between cup and disc boundary)
      0 = optic cup

The full disc region is defined as all pixels with value < 255 (i.e. both 128 and 0),
since the cup lies inside the disc.

Output:
    Dataset/REFUGE/cropped_images/  -- cropped fundus images (.jpg)
    Dataset/REFUGE/cropped_masks/   -- cropped segmentation masks (.bmp)
"""

import os
import glob
import numpy as np
from PIL import Image


# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.join("Dataset", "REFUGE")
IMAGE_DIR = os.path.join(BASE_DIR, "images")
MASK_DIR = os.path.join(BASE_DIR, "mask")
OUT_IMAGE_DIR = os.path.join(BASE_DIR, "cropped_images")
OUT_MASK_DIR = os.path.join(BASE_DIR, "cropped_masks")

os.makedirs(OUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUT_MASK_DIR, exist_ok=True)


def get_disc_bbox(mask_arr: np.ndarray):
    """
    Return the bounding box (min_row, min_col, max_row, max_col) of the full
    optic disc region.  In REFUGE masks the disc region is every pixel whose
    value is *not* 255 (background).  This includes both the disc rim (128)
    and the cup (0).
    """
    disc_region = mask_arr < 255  # True for disc + cup pixels
    non_zero = np.argwhere(disc_region)  # (N, 2) array of (row, col)
    if len(non_zero) == 0:
        return None
    min_row, min_col = non_zero.min(axis=0)
    max_row, max_col = non_zero.max(axis=0)
    return min_row, min_col, max_row, max_col


def compute_crop_box(bbox, img_h: int, img_w: int, scale: float = 2.0):
    """
    Given a bounding box of the disc, compute a *square* crop box that is
    ``scale`` times the disc size, centred on the disc centre.  The result
    is clamped to the image boundaries.

    Returns (top, left, bottom, right) — pixel coordinates suitable for
    PIL.Image.crop((left, top, right, bottom)).
    """
    min_row, min_col, max_row, max_col = bbox

    disc_h = max_row - min_row
    disc_w = max_col - min_col

    # Use the larger dimension so the crop is square
    disc_size = max(disc_h, disc_w)
    crop_size = int(disc_size * scale)

    # Centre of the disc
    center_row = (min_row + max_row) // 2
    center_col = (min_col + max_col) // 2

    half = crop_size // 2

    top = center_row - half
    left = center_col - half
    bottom = top + crop_size
    right = left + crop_size

    # ── Clamp to image boundaries ────────────────────────────────────────
    if top < 0:
        bottom -= top          # shift bottom by the same amount
        top = 0
    if left < 0:
        right -= left
        left = 0
    if bottom > img_h:
        top -= (bottom - img_h)
        bottom = img_h
    if right > img_w:
        left -= (right - img_w)
        right = img_w

    # Final clamp (in case the crop is larger than the image itself)
    top = max(top, 0)
    left = max(left, 0)
    bottom = min(bottom, img_h)
    right = min(right, img_w)

    return top, left, bottom, right


def main():
    mask_paths = sorted(glob.glob(os.path.join(MASK_DIR, "*.bmp")))
    print(f"Found {len(mask_paths)} mask files in {MASK_DIR}")

    processed = 0
    skipped = 0

    for mask_path in mask_paths:
        stem = os.path.splitext(os.path.basename(mask_path))[0]  # e.g. "T0002"

        # Find the corresponding image (.jpg or .png)
        img_path = os.path.join(IMAGE_DIR, stem + ".jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(IMAGE_DIR, stem + ".png")
        if not os.path.exists(img_path):
            print(f"  [SKIP] No image found for mask {stem}")
            skipped += 1
            continue

        # Load mask and image
        mask = Image.open(mask_path)
        mask_arr = np.array(mask)
        img = Image.open(img_path)
        img_arr = np.array(img)

        # Sanity check: dimensions must match
        if mask_arr.shape[:2] != img_arr.shape[:2]:
            print(f"  [WARN] Size mismatch for {stem}: "
                  f"image {img_arr.shape[:2]} vs mask {mask_arr.shape[:2]}")

        # If mask is multi-channel, use first channel
        if len(mask_arr.shape) == 3:
            mask_arr = mask_arr[:, :, 0]

        # Get disc bounding box (pixels < 255)
        bbox = get_disc_bbox(mask_arr)
        if bbox is None:
            print(f"  [SKIP] No disc region in mask {stem}")
            skipped += 1
            continue

        img_h, img_w = mask_arr.shape[:2]
        top, left, bottom, right = compute_crop_box(bbox, img_h, img_w, scale=2.0)

        # Crop using PIL (left, upper, right, lower)
        cropped_img = img.crop((left, top, right, bottom))
        cropped_mask = mask.crop((left, top, right, bottom))

        # Save
        out_img_path = os.path.join(OUT_IMAGE_DIR, stem + ".jpg")
        out_mask_path = os.path.join(OUT_MASK_DIR, stem + ".bmp")

        cropped_img.save(out_img_path)
        cropped_mask.save(out_mask_path)

        disc_h = bbox[2] - bbox[0]
        disc_w = bbox[3] - bbox[1]
        crop_h = bottom - top
        crop_w = right - left
        print(f"  [OK] {stem} | disc bbox ({disc_h}x{disc_w}) "
              f"| crop ({crop_h}x{crop_w}) | saved")
        processed += 1

    print(f"\nDone. Processed: {processed}, Skipped: {skipped}")
    print(f"Cropped images → {OUT_IMAGE_DIR}")
    print(f"Cropped masks  → {OUT_MASK_DIR}")


if __name__ == "__main__":
    main()
