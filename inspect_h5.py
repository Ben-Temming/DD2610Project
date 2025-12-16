import h5py
import numpy as np


"""
This file helps to figure out the structure of the Cyclone dataset as the Kaggle
description wasn't super helpful
"""

# Paths to the files
H5_PATH = "archive/Cyclone_Images.h5"
NPY_PATH = "archive/Cyclone_Labels h5.npy"

print(f"--- Inspecting {H5_PATH} ---")
try:
    with h5py.File(H5_PATH, 'r') as f:
        print("Keys inside the H5 file:", list(f.keys()))
        key = list(f.keys())[0]
        print(f"Shape of data inside '{key}':", f[key].shape)
except Exception as e:
    print("Error reading H5:", e)

print(f"\n--- Inspecting {NPY_PATH} ---")
try:
    labels = np.load(NPY_PATH, allow_pickle=True)
    print("Labels shape:", labels.shape)
    print("First 5 labels:", labels[:5])

    # Extract Intensity Column (Index 5)
    wind_speeds = labels[:, 5].astype(np.float32)

    max_val = wind_speeds.max()
    min_val = wind_speeds.min()

    print(f"\n--- RESULTS ---")
    print(f"Total Cyclones: {len(wind_speeds)}")
    print(f"Max Intensity:  {max_val} knots")
    print(f"Min Intensity:  {min_val} knots")
except Exception as e:
    print("Error reading NPY:", e)