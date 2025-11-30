import h5py
import numpy as np
from tqdm import tqdm

# Paths
H5_PATH = "archive/Cyclone_Images.h5"
BATCH_SIZE = 1000


def calculate_stats():
    print(f"Reading from {H5_PATH}...")

    with h5py.File(H5_PATH, 'r') as f:
        # Find key
        key = list(f.keys())[0]
        dset = f[key]
        total_len = len(dset)

        # Used for mean and sd calculations, length 3 for 3 channels
        channel_sum = np.zeros(3, dtype=np.float64)
        channel_sq_sum = np.zeros(3, dtype=np.float64)
        num_pixels = 0

        # Process in batches
        for i in tqdm(range(0, total_len, BATCH_SIZE)):
            # 1. Read Batch
            batch = dset[i: i + BATCH_SIZE]

            # 2. Slice to RGB (first 3 channels)
            batch = batch[:, :, :, :3]

            # 3. Normalize to [0, 1] Float
            batch = batch.astype(np.float64)
            if batch.max() > 1.0:
                batch /= 255.0

            # 4. Reshape to (N_pixels, 3) to sum easily
            pixels = batch.reshape(-1, 3)

            # 5. Accumulate
            channel_sum += pixels.sum(axis=0)
            channel_sq_sum += (pixels ** 2).sum(axis=0)
            num_pixels += pixels.shape[0]

        # 6. Final Calculation
        mean = channel_sum / num_pixels
        std = np.sqrt((channel_sq_sum / num_pixels) - (mean ** 2))

        print(f'mean["cyclone"] = [{mean[0]:.8f}, {mean[1]:.8f}, {mean[2]:.8f}]')
        print(f'std["cyclone"] = [{std[0]:.8f}, {std[1]:.8f}, {std[2]:.8f}]')


if __name__ == "__main__":
    calculate_stats()