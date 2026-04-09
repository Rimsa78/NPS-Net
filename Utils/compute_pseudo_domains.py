#!/usr/bin/env python3
"""
Pre-compute pseudo-domain labels for DoFE training.

Clusters training images from the main CSV into K pseudo-domains based on
LAB colour statistics. Results are cached to pseudo_domains.npy.

Usage:
    cd Comparision/
    python compute_pseudo_domains.py
    python compute_pseudo_domains.py --n-domains 4 --out pseudo_domains.npy
"""

import argparse
import os

from config import DATA_CSV
from pseudo_domains import load_pseudo_domain_labels


def main():
    parser = argparse.ArgumentParser(
        description='Pre-compute pseudo-domain labels for DoFE')
    parser.add_argument('--csv', default=DATA_CSV, help='Path to data CSV')
    parser.add_argument('--n-domains', type=int, default=4)
    parser.add_argument('--out', default='pseudo_domains.npy')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    labels = load_pseudo_domain_labels(
        csv_path=args.csv,
        n_domains=args.n_domains,
        cache_path=args.out,
        seed=args.seed,
    )
    print(f"\n[done] {len(labels)} labels saved to {args.out}")


if __name__ == '__main__':
    main()
