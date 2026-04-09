# pseudo_domains.py
"""
Pseudo-domain assignment for DoFE training.

Since we train on a single source domain, we create pseudo-domains via
K-means clustering on image-level statistics (LAB colour space: mean L, a, b
+ contrast). This gives ~K clusters that capture visual style variability.

Usage:
    from pseudo_domains import load_pseudo_domain_labels
    labels = load_pseudo_domain_labels(csv_path, n_domains=4, cache_path='pseudo_domains.npy')
"""

import os
import numpy as np
import cv2
import pandas as pd
from sklearn.cluster import KMeans


def compute_image_stats(image_path, target_size=256):
    """Compute LAB colour statistics for one image.

    Returns: [mean_L, mean_a, mean_b, std_L, contrast]
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return np.zeros(5, dtype=np.float32)

    # Resize for speed
    img = cv2.resize(img, (target_size, target_size))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    return np.array([
        L.mean(), a.mean(), b.mean(),
        L.std(),
        img.astype(np.float32).std(),  # global contrast proxy
    ], dtype=np.float32)


def assign_pseudo_domains(csv_path, n_domains=4, seed=42):
    """Compute pseudo-domain labels for all images in the CSV.

    Args:
        csv_path: path to data CSV with 'Raw' column
        n_domains: number of pseudo-domain clusters
        seed: random seed for K-means

    Returns:
        labels: np.ndarray of shape (N,) with int labels in [0, n_domains)
        stats: np.ndarray of shape (N, 5) feature matrix
    """
    df = pd.read_csv(csv_path)
    valid_mask = df['Raw'].apply(lambda x: os.path.isfile(str(x)))
    df = df[valid_mask].reset_index(drop=True)

    n = len(df)
    stats = np.zeros((n, 5), dtype=np.float32)

    for i in range(n):
        stats[i] = compute_image_stats(df.iloc[i]['Raw'])

    # Normalise features
    mu = stats.mean(axis=0, keepdims=True)
    sigma = stats.std(axis=0, keepdims=True) + 1e-8
    stats_norm = (stats - mu) / sigma

    # K-means clustering
    km = KMeans(n_clusters=n_domains, random_state=seed, n_init=10)
    labels = km.fit_predict(stats_norm)

    return labels.astype(np.int64), stats


def load_pseudo_domain_labels(csv_path, n_domains=4, cache_path=None, seed=42):
    """Load or compute pseudo-domain labels.

    Args:
        csv_path: path to data CSV
        n_domains: number of pseudo-domain clusters
        cache_path: if provided, cache labels to/from this .npy file
        seed: random seed

    Returns:
        labels: np.ndarray of shape (N,) with int labels in [0, n_domains)
    """
    if cache_path and os.path.isfile(cache_path):
        labels = np.load(cache_path)
        print(f"[pseudo_domains] Loaded cached labels from {cache_path} "
              f"({len(labels)} samples, {len(np.unique(labels))} domains)")
        return labels

    print(f"[pseudo_domains] Computing pseudo-domain labels (K={n_domains}) ...")
    labels, stats = assign_pseudo_domains(csv_path, n_domains, seed)

    if cache_path:
        np.save(cache_path, labels)
        print(f"[pseudo_domains] Cached → {cache_path}")

    # Report cluster sizes
    for k in range(n_domains):
        count = (labels == k).sum()
        print(f"  Domain {k}: {count} samples ({100*count/len(labels):.1f}%)")

    return labels
