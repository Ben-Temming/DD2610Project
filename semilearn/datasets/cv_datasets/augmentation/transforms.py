from torchvision import transforms

from .randaugment import RandAugment


mean, std = {}, {}
mean["utkface"] = [0.59632254, 0.45671629, 0.39103324]
std["utkface"] = [0.25907077, 0.23132719, 0.22686818]
mean["cyclone"] = [0.70588239, 0.68448650, 0.25788743]
std["cyclone"] = [0.25630702, 0.20469663, 0.24108123]
mean["cyclone_standard"] = [0.71674829, 0.68610356, 0.24847185]
std["cyclone_standard"] = [0.24776029, 0.20396079, 0.22555258]


def get_val_transforms(crop_size, dataset_name):
    return transforms.Compose(
        [
            transforms.Resize(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean[dataset_name.lower()],
                std[dataset_name.lower()],
            ),
        ]
    )


def get_weak_transforms(crop_size, crop_ratio, dataset_name):
    return transforms.Compose(
        [
            transforms.Resize(crop_size),
            transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean[dataset_name.lower()], std[dataset_name.lower()]),
        ]
    )


def get_strong_transforms(crop_size, crop_ratio, dataset_name):

    if dataset_name == "cyclone" or dataset_name == "cyclone_standard":

        # We define a custom pipeline that ONLY does geometry.
        # NO ColorJitter, NO Brightness, NO Contrast.
        return transforms.Compose([
            transforms.Resize(crop_size),

            # Random Crop (Standard)
            transforms.RandomResizedCrop(
                crop_size,
                scale=(0.2, 1.0)  # Zooming in is safe (it's just a closer look at the storm)
            ),

            # Horizontal Flip (Safe - storms look similar mirrored)
            transforms.RandomHorizontalFlip(),

            # --- THE KEY GEOMETRIC AUGMENTATIONS ---

            # Rotation: Storms can be oriented any way
            transforms.RandomRotation(degrees=180),

            # Affine: Shearing/Stretching slightly
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                shear=10
            ),

            transforms.ToTensor(),

            # Normalize with YOUR calculated stats (from the 80% split)
            transforms.Normalize(
                mean=mean[dataset_name],
                std=std[dataset_name]
            )
        ])

    return transforms.Compose(
        [
            transforms.Resize(crop_size),
            transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            RandAugment(3, 5),
            transforms.ToTensor(),
            transforms.Normalize(mean[dataset_name.lower()], std[dataset_name.lower()]),
        ]
    )
