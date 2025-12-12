# Copyright (c) 2024 Pin-Yen Huang.
# Licensed under the MIT License.
# Code in this file is adapted from pytorch/pytorch
# https://github.com/pytorch/vision/blob/main/torchvision/datasets/food101.py

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import PIL.Image

from torchvision.datasets.utils import verify_str_arg
from torchvision.datasets.vision import VisionDataset

import kagglehub # pip install kagglehub
import shutil


class CROWD_COUNTING(VisionDataset):
    """
    https://www.kaggle.com/datasets/fmena14/crowd-counting/data

    2000 images of crowds with labels indicating the number of people in each image.
    250 labeled, the rest used as unlabeled data.

    Dataset structure:
        crowd-counting/
            frames/
                frames/
                    seq000001.jpg
                    seq000002.jpg
                    ...
            labels.csv
            labels.npy
            images.npy

    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default) and ``"test"``.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again. Default is False.
    """


    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._split = verify_str_arg(split, "split", ("train", "test"))
        self._base_folder = Path(self.root) / "crowd_counting"
        # self._meta_folder = self._base_folder
        self._images_folder = self._base_folder / "frames" / "frames"
        self._labels_file = self._base_folder / "labels.csv"

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")


        # read labels from numpy file from structured dataset
        csv_file = pd.read_csv(self._labels_file)
        # first column = index -> name image integer with 6 digits = "seq{index:06d}.jpg"
        self._file_paths = csv_file.iloc[:, 0].astype(int).apply(lambda x: self._images_folder / f"seq_{x:06d}.jpg").to_numpy(dtype="object")
        # second column = label
        self._labels = csv_file.iloc[:, 1].astype(str).to_numpy(dtype=np.float32)

        # self._labels = np.load(self._base_folder / "labels.npy")
        # self._file_paths = np.load(self._base_folder / "images.npy")
        self.print_info()


    def __len__(self) -> int:
        return len(self._file_paths)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image_file, label = self._file_paths[idx], self._labels[idx]
        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def extra_repr(self) -> str:
        return f"split={self._split}"

    def _check_exists(self) -> bool:
        return self._images_folder.exists() and self._images_folder.is_dir() \
            and (self._labels_file).exists() 

    def _download(self) -> None:
        if self._check_exists():
            return
        
        # download dataset from kaggle to cache
        path = kagglehub.dataset_download("fmena14/crowd-counting")
        # copy dataset from cache to self._base_folder
        shutil.copytree(path, self._base_folder, dirs_exist_ok=True)
        print(f"CrowdCounting dataset downloaded to {path} and copied to {self._base_folder}.")
        # a copy of the dataset is saved in kagglehub's cache directory

    def print_info(self):
        # print some info
        num_samples = self.__len__()
        print(f"CrowdCounting dataset contains {num_samples} samples.")
        shape_image = PIL.Image.open(self._file_paths[0]).size
        print(f"Image shape: {shape_image[0]}x{shape_image[1]}x{shape_image[2]}")
