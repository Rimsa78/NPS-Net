"""
Crop Papila fundus images and segmentation masks around the optic disc.

The Papila dataset stores segmentations as polygon contour text files
(x, y coordinate pairs) with two experts per structure:
    {image_id}_{cup|disc}_{exp1|exp2}.txt

Processing pipeline:
    1. Load both expert contours for cup and disc
    2. Average the two expert polygons (point-wise interpolation)
    3. Render filled polygons into a single combined mask:
           255 = background
           128 = disc rim  (inside disc, outside cup)
             0 = cup       (inside cup)
    4. Find the disc bounding box from the combined mask
    5. Crop both image and mask at 2× the disc bounding-box size
    6. Save to output folders

Output:
    Dataset/Papila/cropped_images/  — cropped fundus images (.jpg)
    Dataset/Papila/cropped_masks/   — cropped combined masks (.png)
"""

import os
import re
import glob
from collections import defaultdict

import numpy as np
import cv2
from PIL import Image


# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.join("Dataset", "Papila")
IMAGE_DIR = os.path.join(BASE_DIR, "FundusImages")
CONTOUR_DIR = os.path.join(BASE_DIR, "ExpertsSegmentations", "Contours")
OUT_IMAGE_DIR = os.path.join(BASE_DIR, "cropped_images")
OUT_MASK_DIR = os.path.join(BASE_DIR, "cropped_masks")

os.makedirs(OUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUT_MASK_DIR, exist_ok=True)


# ==============================================================================
# CONTOUR PARSING
# ==============================================================================

def load_contour(txt_path):
    """Load a polygon contour from a Papila-format text file.

    Each line: '   x_float   y_float'
    Returns: (N, 2) float array of (x, y) coordinates.
    """
    points = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                x, y = float(parts[0]), float(parts[1])
                points.append([x, y])
    return np.array(points, dtype=np.float64)


def _signed_area(pts):
    """Compute signed area via the shoelace formula.
    Positive = counter-clockwise, negative = clockwise."""
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def _ensure_ccw(pts):
    """Ensure a polygon has counter-clockwise winding."""
    if _signed_area(pts) < 0:
        return pts[::-1].copy()
    return pts


def _align_start(pts, ref_start):
    """Rotate polygon vertices so the one closest to ref_start comes first."""
    dists = np.sqrt(((pts - ref_start) ** 2).sum(axis=1))
    start_idx = dists.argmin()
    return np.roll(pts, -start_idx, axis=0)


def _resample(pts, n_points=200):
    """Resample a polygon to n_points uniformly-spaced points along arc-length."""
    closed = np.vstack([pts, pts[:1]])
    diffs = np.diff(closed, axis=0)
    seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    cum_length = np.concatenate([[0], np.cumsum(seg_lengths)])
    total = cum_length[-1]
    if total < 1e-6:
        return pts[:n_points] if len(pts) >= n_points else pts
    target_lengths = np.linspace(0, total, n_points, endpoint=False)
    resampled = np.zeros((n_points, 2))
    for i, t in enumerate(target_lengths):
        idx = np.searchsorted(cum_length, t, side='right') - 1
        idx = min(idx, len(closed) - 2)
        seg_start = cum_length[idx]
        seg_end = cum_length[idx + 1]
        seg_len = seg_end - seg_start
        if seg_len < 1e-8:
            resampled[i] = closed[idx]
        else:
            frac = (t - seg_start) / seg_len
            resampled[i] = closed[idx] + frac * (closed[idx + 1] - closed[idx])
    return resampled


def average_contours(contour1, contour2):
    """Average two polygon contours by:
    1. Ensuring both have the same winding direction (CCW)
    2. Aligning their starting points
    3. Resampling to the same number of points
    4. Computing point-wise midpoints
    """
    n = 200

    # Ensure consistent CCW winding
    c1 = _ensure_ccw(contour1)
    c2 = _ensure_ccw(contour2)

    # Resample c1 first
    r1 = _resample(c1, n)

    # Align c2's start to c1's top-most point, then resample
    c2 = _align_start(c2, r1[0])
    r2 = _resample(c2, n)

    return (r1 + r2) / 2.0


def contour_to_mask(contour, shape):
    """Render a polygon contour as a filled binary mask."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    pts = contour.astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 255)
    return mask


# ==============================================================================
# CATALOGUE CONTOUR FILES
# ==============================================================================

def catalogue_contours(contour_dir):
    """Build a dict: image_id → {structure → {expert → file_path}}."""
    pattern = re.compile(r'(RET\d+[A-Z]{2})_(cup|disc)_(exp\d+)\.txt')
    catalogue = defaultdict(lambda: defaultdict(dict))

    for fname in sorted(os.listdir(contour_dir)):
        m = pattern.match(fname)
        if m:
            img_id, structure, expert = m.groups()
            catalogue[img_id][structure][expert] = os.path.join(contour_dir, fname)

    return catalogue


# ==============================================================================
# CROP LOGIC (same as REFUGE)
# ==============================================================================

def get_disc_bbox(mask):
    """Return (min_row, min_col, max_row, max_col) of the disc region."""
    non_bg = mask < 255  # disc + cup
    rows_cols = np.argwhere(non_bg)
    if len(rows_cols) == 0:
        return None
    min_row, min_col = rows_cols.min(axis=0)
    max_row, max_col = rows_cols.max(axis=0)
    return min_row, min_col, max_row, max_col


def compute_crop_box(bbox, img_h, img_w, scale=2.0):
    """Square crop centred on disc, ``scale`` times larger."""
    min_row, min_col, max_row, max_col = bbox
    disc_h = max_row - min_row
    disc_w = max_col - min_col
    disc_size = max(disc_h, disc_w)
    crop_size = int(disc_size * scale)

    center_row = (min_row + max_row) // 2
    center_col = (min_col + max_col) // 2
    half = crop_size // 2

    top = center_row - half
    left = center_col - half
    bottom = top + crop_size
    right = left + crop_size

    if top < 0:
        bottom -= top
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

    top = max(top, 0)
    left = max(left, 0)
    bottom = min(bottom, img_h)
    right = min(right, img_w)

    return top, left, bottom, right


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    catalogue = catalogue_contours(CONTOUR_DIR)
    print(f"Found contours for {len(catalogue)} images")

    processed = 0
    skipped = 0

    for img_id in sorted(catalogue.keys()):
        structs = catalogue[img_id]

        # Check we have both cup and disc
        if 'cup' not in structs or 'disc' not in structs:
            print(f"  [SKIP] {img_id}: missing cup or disc contour")
            skipped += 1
            continue

        # Find the fundus image
        img_path = os.path.join(IMAGE_DIR, img_id + ".jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(IMAGE_DIR, img_id + ".png")
        if not os.path.exists(img_path):
            print(f"  [SKIP] {img_id}: no fundus image found")
            skipped += 1
            continue

        # Load image to get dimensions
        img = Image.open(img_path)
        img_w, img_h = img.size  # PIL (width, height)
        shape = (img_h, img_w)

        # ── Load and average disc contours ───────────────────────────────
        disc_experts = structs['disc']
        disc_contours = [load_contour(disc_experts[exp])
                         for exp in sorted(disc_experts.keys())]
        if len(disc_contours) == 2:
            disc_avg = average_contours(disc_contours[0], disc_contours[1])
        else:
            disc_avg = disc_contours[0]

        # ── Load and average cup contours ────────────────────────────────
        cup_experts = structs['cup']
        cup_contours = [load_contour(cup_experts[exp])
                        for exp in sorted(cup_experts.keys())]
        if len(cup_contours) == 2:
            cup_avg = average_contours(cup_contours[0], cup_contours[1])
        else:
            cup_avg = cup_contours[0]

        # ── Render filled binary masks ───────────────────────────────────
        disc_mask = contour_to_mask(disc_avg, shape)
        cup_mask = contour_to_mask(cup_avg, shape)

        # ── Build combined mask (REFUGE encoding) ────────────────────────
        # 255 = background, 128 = disc rim, 0 = cup
        combined = np.full(shape, 255, dtype=np.uint8)
        combined[disc_mask > 0] = 128    # disc rim
        combined[cup_mask > 0] = 0       # cup (overwrites disc rim inside cup)

        # ── Compute disc bounding box and 2x crop ────────────────────────
        bbox = get_disc_bbox(combined)
        if bbox is None:
            print(f"  [SKIP] {img_id}: no disc region found")
            skipped += 1
            continue

        top, left, bottom, right = compute_crop_box(bbox, img_h, img_w, scale=2.0)

        # Crop using PIL (left, top, right, bottom)
        cropped_img = img.crop((left, top, right, bottom))
        cropped_mask = Image.fromarray(combined).crop((left, top, right, bottom))

        # Save
        out_img = os.path.join(OUT_IMAGE_DIR, img_id + ".jpg")
        out_mask = os.path.join(OUT_MASK_DIR, img_id + ".png")
        cropped_img.save(out_img)
        cropped_mask.save(out_mask)

        disc_h = bbox[2] - bbox[0]
        disc_w = bbox[3] - bbox[1]
        crop_h = bottom - top
        crop_w = right - left
        print(f"  [OK] {img_id} | disc ({disc_h}x{disc_w}) "
              f"| crop ({crop_h}x{crop_w}) | saved")
        processed += 1

    print(f"\nDone. Processed: {processed}, Skipped: {skipped}")
    print(f"Cropped images → {OUT_IMAGE_DIR}")
    print(f"Cropped masks  → {OUT_MASK_DIR}")


if __name__ == "__main__":
    main()
