# semilearn/datasets/cv_datasets/cyclone.py
import numpy as np
import h5py
from torchvision.datasets.vision import VisionDataset
from PIL import Image


class CYCLONE(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None,
                 download=False, data=None, targets=None,
                 is_ulb=False, strong_transform=None, alg=None):
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.alg = alg
        self.is_ulb = is_ulb
        self.strong_transform = strong_transform

        self.h5_path = f"{root}/Cyclone_Images.h5"
        self.npy_path = f"{root}/Cyclone_Labels h5.npy"
        self.dataset_key = 'Images'

        # Normalize intensity
        # Found using the inspect_h5.py
        self.max_intensity = 168

        if data is not None:
            self.indices = data
            self.labels = targets
        else:
            raw_labels = np.load(self.npy_path, allow_pickle=True)
            full_labels = raw_labels[:, 5].astype(np.float32)

            # Normalize labels immediately upon loading
            # full_labels = full_labels / self.max_intensity
            # (Wait, better to normalize in __getitem__ so we can inspect real values if needed,
            # but for RankUp stability, let's normalize the source array if it's the raw load)

            with h5py.File(self.h5_path, 'r') as f:
                total_len = len(f[self.dataset_key])

            split_idx = int(0.8 * total_len)
            if split == 'train':
                self.indices = np.arange(0, split_idx)
                self.labels = full_labels[0:split_idx]
            else:
                self.indices = np.arange(split_idx, total_len)
                self.labels = full_labels[split_idx:]

        self._file_paths = self.indices
        self._labels = self.labels
        self.targets = self.labels
        self.data = self.indices

    def __len__(self):
        return len(self.indices)

    def __sample__(self, idx):
        real_idx = self.indices[idx]
        with h5py.File(self.h5_path, 'r') as f:
            img_data = f[self.dataset_key][real_idx]

        label = self.labels[idx]

        # Normalize label
        if label > 1.0:
            label = label / self.max_intensity

        img_data = img_data[:, :, :3]
        if img_data.dtype != np.uint8:
            if img_data.max() <= 1.0:
                img_data = (img_data * 255).astype(np.uint8)
            else:
                img_data = img_data.astype(np.uint8)

        image = Image.fromarray(img_data, mode='RGB')
        return image, label

    def __getitem__(self, idx):
        img, target = self.__sample__(idx)

        if self.transform is None:
            from torchvision import transforms
            return {"x_lb": transforms.ToTensor()(img), "y_lb": target}

        data_dict = {
            "idx_lb": idx,
            "x_lb": self.transform(img),
            "y_lb": target,
            "idx_ulb": idx,
            "x_ulb_w": self.transform(img),
            # MixMatch needs a second weak augmentation:
            "x_ulb_w_2": self.transform(img),
            # RankUp needs strong augmentation:
            "x_ulb_s": self.strong_transform(img) if self.strong_transform else None,
        }

        if not self.is_ulb:
            return {
                "idx_lb": data_dict["idx_lb"],
                "x_lb": data_dict["x_lb"],
                "y_lb": data_dict["y_lb"]
            }
        else:
            # Select based on algorithm
            if self.alg == "mixmatch":
                return {
                    "idx_ulb": data_dict["idx_ulb"],
                    "x_ulb_w": data_dict["x_ulb_w"],
                    "x_ulb_w_2": data_dict["x_ulb_w_2"]
                }
            elif self.alg == "rankup" or self.alg == "rankuprda":
                return {
                    "idx_ulb": data_dict["idx_ulb"],
                    "x_ulb_w": data_dict["x_ulb_w"],
                    "x_ulb_s": data_dict["x_ulb_s"]
                }
            else:
                # Default/Fallback
                return {
                    "idx_ulb": data_dict["idx_ulb"],
                    "x_ulb_w": data_dict["x_ulb_w"]
                }