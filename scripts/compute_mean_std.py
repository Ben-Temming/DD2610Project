import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


def compute_mean_std(images_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-channel mean and std over all RGB images in images_dir.

    Returns mean and std as arrays of shape (3,), in [0,1] scale.
    """
    img_paths = []
    for root, _, files in os.walk(images_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                img_paths.append(Path(root) / f)

    if not img_paths:
        raise FileNotFoundError(f"No images found under {images_dir}")

    # Accumulate sums and squared sums per channel
    n_pixels_total = 0
    sum_c = np.zeros(3, dtype=np.float64)
    sumsq_c = np.zeros(3, dtype=np.float64)

    for p in img_paths:
        img = Image.open(p).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC in [0,1]
        # reshape to (N, 3)
        h, w, _ = arr.shape
        n = h * w
        n_pixels_total += n
        flat = arr.reshape(-1, 3)
        sum_c += flat.sum(axis=0)
        sumsq_c += (flat ** 2).sum(axis=0)

    mean = sum_c / n_pixels_total
    var = (sumsq_c / n_pixels_total) - (mean ** 2)
    std = np.sqrt(np.maximum(var, 0.0))
    return mean, std


def main():
    parser = argparse.ArgumentParser(description="Compute dataset mean/std for Normalize")
    parser.add_argument(
        "--images-dir",
        type=str,
        required=True,
        help="Path to images directory",
    )
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    mean, std = compute_mean_std(images_dir)
    print("Mean:", mean.tolist())
    print("Std:", std.tolist())
    

if __name__ == "__main__":
    main()
