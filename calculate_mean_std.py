import numpy as np
from PIL import Image
import os
from tqdm import tqdm

# Config
DATA_ROOT = "data/cyclone_standard"
TRAIN_LIST = os.path.join(DATA_ROOT, "train.txt")

print(f"Reading train list from {TRAIN_LIST}...")

# 1. Get list of image paths
image_paths = []
with open(TRAIN_LIST, 'r') as f:
    for line in f:
        path = line.strip().split()[0]  # "images/cyclone_00001.jpg"
        full_path = os.path.join(DATA_ROOT, path)
        image_paths.append(full_path)

print(f"Found {len(image_paths)} training images. Computing stats...")

# 2. Counters
pixel_num = 0
channel_sum = np.zeros(3)
channel_sq_sum = np.zeros(3)

for path in tqdm(image_paths):
    try:
        # Load and convert to 0-1 range numpy array
        img = Image.open(path).convert('RGB')
        img_np = np.array(img) / 255.0

        # Reshape to (Pixels, 3)
        pixels = img_np.reshape(-1, 3)

        # Accumulate
        channel_sum += pixels.sum(axis=0)
        channel_sq_sum += (pixels ** 2).sum(axis=0)
        pixel_num += pixels.shape[0]

    except Exception as e:
        print(f"Error reading {path}: {e}")

# 3. Final Calculation
rgb_mean = channel_sum / pixel_num
rgb_std = np.sqrt((channel_sq_sum / pixel_num) - (rgb_mean ** 2))

print("\n--- RESULTS FOR TRANSFORMS.PY ---")
print(f'mean["cyclone_standard"] = [{rgb_mean[0]:.8f}, {rgb_mean[1]:.8f}, {rgb_mean[2]:.8f}]')
print(f'std["cyclone_standard"] = [{rgb_std[0]:.8f}, {rgb_std[1]:.8f}, {rgb_std[2]:.8f}]')