# Copyright (c) Microsoft Corporation.
# Modifications Copyright (c) 2024 Pin-Yen Huang.
# Licensed under the MIT License.

from semilearn.datasets import cv_datasets
from semilearn.datasets.utils import split_ssl_data, load_image_files
import os
import numpy as np

from .datasetbase import BasicDataset, ImagePathDataset
from .augmentation import get_val_transforms, get_weak_transforms, get_strong_transforms


def get_cv_dataset(args, alg, dataset_name, num_labels, data_dir="./data", include_lb_to_ulb=True):
    # 1. Setup Transforms
    transform_weak = get_weak_transforms(crop_size=args.img_size, crop_ratio=args.crop_ratio, dataset_name=dataset_name)
    transform_strong = get_strong_transforms(crop_size=args.img_size, crop_ratio=args.crop_ratio,
                                             dataset_name=dataset_name)
    transform_val = get_val_transforms(crop_size=args.img_size, dataset_name=dataset_name)

    # 2. LOAD DATA
    if dataset_name == "cyclone_standard":
        root = os.path.join(data_dir, "cyclone_standard")

        def load_txt(filename):
            paths = []
            targets = []
            with open(os.path.join(root, filename), 'r') as f:
                for line in f:
                    p, t = line.strip().split()
                    paths.append(os.path.join(root, p))
                    targets.append(float(t))
            return np.array(paths), np.array(targets, dtype=np.float32)

        # Load TRAIN (for Labeled + Unlabeled)
        train_data, train_targets = load_txt("train.txt")

        # Load TEST (for Evaluation only)
        test_data, test_targets = load_txt("test.txt")
        ImageDataset = ImagePathDataset

    else:
        # Version from the original code for the UTKFace dataset
        dataset = getattr(cv_datasets, dataset_name.upper())

        train_dataset = dataset(data_dir, split="train", download=True)
        train_paths, train_targets = train_dataset._file_paths, train_dataset._labels

        test_dataset = dataset(data_dir, split="test", download=True)
        test_paths, test_targets = test_dataset._file_paths, test_dataset._labels

        if args.preload:
            train_data = load_image_files(train_paths)
            test_data = load_image_files(test_paths)
            ImageDataset = BasicDataset
        else:
            train_data = train_paths
            test_data = test_paths
            ImageDataset = ImagePathDataset

    # 3. Create Evaluation Dataset
    eval_dset = ImageDataset(alg, test_data, test_targets, transform_val, False, None)
    test_dset = None

    if alg == "fullysupervised":
        lb_dset = ImageDataset(alg, train_data, train_targets, transform_weak, False, transform_strong)
        return lb_dset, None, eval_dset, test_dset

    # 4. Split Labeled / Unlabeled
    lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(
        args,
        train_data,
        train_targets,
        lb_num_labels=num_labels,
        ulb_num_labels=args.ulb_num_labels,
        include_lb_to_ulb=include_lb_to_ulb,
    )

    # 5. Create Final Datasets
    lb_dset = ImageDataset(alg, lb_data, lb_targets, transform_weak, False, transform_strong)
    ulb_dset = ImageDataset(alg, ulb_data, ulb_targets, transform_weak, True, transform_strong)

    if alg == "supervised":
        ulb_dset = None

    return lb_dset, ulb_dset, eval_dset, test_dset