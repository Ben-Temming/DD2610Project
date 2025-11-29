import h5py
import numpy as np
import os

# Paths to the files
H5_PATH = "archive/Cyclone_Images.h5"
NPY_PATH = "archive/Cyclone_Labels h5.npy"

print(f"--- Inspecting {H5_PATH} ---")
try:
    with h5py.File(H5_PATH, 'r') as f:
        print("Keys inside the H5 file:", list(f.keys()))
        # Let's assume the key is the first one found
        key = list(f.keys())[0]
        print(f"Shape of data inside '{key}':", f[key].shape)
except Exception as e:
    print("Error reading H5:", e)

print(f"\n--- Inspecting {NPY_PATH} ---")
try:
    labels = np.load(NPY_PATH, allow_pickle=True)
    print("Labels shape:", labels.shape)
    print("First 5 labels:", labels[:5])
except Exception as e:
    print("Error reading NPY:", e)