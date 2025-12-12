import h5py
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

# CONFIG
H5_PATH = "archive/Cyclone_Images.h5"
NPY_PATH = "archive/Cyclone_Labels h5.npy"
OUTPUT_DIR = "data/cyclone_standard"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
TRAIN_LIST = os.path.join(OUTPUT_DIR, "train.txt")
TEST_LIST = os.path.join(OUTPUT_DIR, "test.txt")

# 1. Setup
os.makedirs(IMAGES_DIR, exist_ok=True)
print("Loading labels...")
raw_labels = np.load(NPY_PATH, allow_pickle=True)
intensities = raw_labels[:, 5].astype(np.float32)
max_val = 168.0

# 2. Conversion Loop
print(f"Converting images to {IMAGES_DIR}...")
with h5py.File(H5_PATH, 'r') as f:
    images = f['Images']
    total_len = len(images)
    split_idx = int(0.8 * total_len)  # 80/20 Split Point

    with open(TRAIN_LIST, 'w') as f_train, open(TEST_LIST, 'w') as f_test:
        for idx in tqdm(range(total_len)):
            # A. Process Image
            img_data = images[idx]
            img_data = img_data[:, :, :3]

            if img_data.dtype != np.uint8:
                if img_data.max() <= 1.0:
                    img_data = (img_data * 255).astype(np.uint8)
                else:
                    img_data = img_data.astype(np.uint8)

            img = Image.fromarray(img_data, mode='RGB')
            filename = f"cyclone_{idx:05d}.jpg"
            img.save(os.path.join(IMAGES_DIR, filename))

            # B. Prepare Line: "images/filename.jpg normalized_target"
            # norm_target = intensities[idx] / max_val
            raw_target = intensities[idx]
            line = f"images/{filename} {raw_target:.6f}\n"

            # C. Write to correct file
            if idx < split_idx:
                f_train.write(line)
            else:
                f_test.write(line)

print("\nDone!")
print(f"Train set: {split_idx} images -> {TRAIN_LIST}")
print(f"Test set:  {total_len - split_idx} images -> {TEST_LIST}")