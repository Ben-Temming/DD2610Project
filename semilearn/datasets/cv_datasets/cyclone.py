# semilearn/datasets/cv_datasets/cyclone.py
import numpy as np
import h5py
from torchvision.datasets.vision import VisionDataset
from PIL import Image


class CYCLONE(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None, download=False, data=None,
                 targets=None):
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.h5_path = f"{root}/Cyclone_Images.h5"
        self.npy_path = f"{root}/Cyclone_Labels h5.npy"
        self.dataset_key = 'Images'

        # 1. Handling Data Source
        if data is not None:
            self.indices = data
            self.labels = targets
        else:
            raw_labels = np.load(self.npy_path, allow_pickle=True)
            full_labels = raw_labels[:, 5].astype(np.float32)

            with h5py.File(self.h5_path, 'r') as f:
                total_len = len(f[self.dataset_key])

            split_idx = int(0.8 * total_len)
            if split == 'train':
                self.indices = np.arange(0, split_idx)
                self.labels = full_labels[0:split_idx]
            else:
                self.indices = np.arange(split_idx, total_len)
                self.labels = full_labels[split_idx:]

        # 2. Compatibility Attributes
        self._file_paths = self.indices
        self._labels = self.labels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        with h5py.File(self.h5_path, 'r') as f:
            img_data = f[self.dataset_key][real_idx]

        label = self.labels[idx]

        # 4 channels -> 3 channels RGB
        img_data = img_data[:, :, :3]
        if img_data.dtype != np.uint8:
            if img_data.max() <= 1.0:
                img_data = (img_data * 255).astype(np.uint8)
            else:
                img_data = img_data.astype(np.uint8)

        image = Image.fromarray(img_data, mode='RGB')

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label