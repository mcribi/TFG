import os
import numpy as np
import torch
import nibabel as nib
from tqdm import tqdm
from datasets import LungCTWithMaskDataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd,
    ResizeWithPadOrCropd, RandFlipd, RandAffined, RandGaussianNoised, ToTensord, ResizeD, TransposeD
)
from monai.data import DataLoader
from datasets import LungCTDataset, LungCTWithMaskDataset
from transforms import Windowingd, MultiWindowingd, CombineMaskAndImage,DebugShaped
import traceback

def save_npy(output_path, array):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path + ".npy", array)

def save_nifti(output_path, array, affine=np.eye(4)):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nib.save(nib.Nifti1Image(array, affine), output_path + ".nii.gz")

def run_preprocessing_configs(image_dir, mask_dir, output_base_dir, configs_dict):
    for config_name, config in configs_dict.items():
        print(f"\n🚀 Procesando configuración: {config_name}")
        base_output_dir = os.path.join(output_base_dir, config_name)

        # Validar formatos
        save_formats = config.get("save_format", ["npy"])
        if isinstance(save_formats, str):
            save_formats = [save_formats]
        for fmt in save_formats:
            assert fmt in ["npy", "nifti"], f"❌ Formato no soportado: {fmt}"

        # Dataset
        dataset = LungCTWithMaskDataset(
            image_dir=image_dir,
            mask_dir=mask_dir,
            transform=config.get("transforms"),
            use_monai_io=config.get("use_monai_io", True),
            combine_strategy=config.get("combine_strategy", None)
        )

        save_separately = config.get("save_separately", False)

        for i in tqdm(range(len(dataset)), desc=f"[{config_name}]"):
            try:
                item = dataset[i]
                fname = os.path.basename(dataset.files[i]).replace('.nii.gz', '').replace('.nii', '')

                if dataset.use_monai_io:
                    image, mask, label = item  #item["image"], item.get("label", dataset[i][1])
                    # mask = item.get("mask", None)
                else:
                    image, mask, label = item
                    # mask = None

                def save_to_all_formats(array, subfolder, base_filename):
                    for fmt in save_formats:
                        folder = os.path.join(base_output_dir, fmt, subfolder)
                        full_path = os.path.join(folder, base_filename)
                        if fmt == "npy":
                            save_npy(full_path, array)
                        elif fmt == "nifti":
                            save_nifti(full_path, array)

                if save_separately and mask is not None:
                    save_to_all_formats(image.squeeze().numpy(), "images", fname)
                    save_to_all_formats(mask.squeeze().numpy(), "masks", fname)
                else:
                    combined = image.numpy() if isinstance(image, torch.Tensor) else image
                    save_to_all_formats(combined, "combined", fname)

            except Exception as e:
                print(f"⚠️ Error procesando {dataset.files[i]}:")
                traceback.print_exc()


transforms_small = Compose([
    LoadImaged(keys=["image","mask"]),
    DebugShaped(keys=["image", "mask"], name="EnsureChannelFirstd"),
    EnsureChannelFirstd(keys=["image", "mask"]),
    DebugShaped(keys=["image", "mask"], name="TransposeD"),
    TransposeD(keys=["image", "mask"], indices=(0, 3, 1, 2)),    # [1, D, H, W]
    DebugShaped(keys=["image", "mask"], name="ResizeD"),
    ResizeD(keys=["image", "mask"], spatial_size=(128, 256, 256), mode="trilinear", align_corners=True),
    DebugShaped(keys=["image", "mask"], name="ToTensord"),
    ToTensord(keys=["image", "mask"])
])

transforms_small_hu_m600_1500 = Compose([
    LoadImaged(keys=["image","mask"]),
    DebugShaped(keys=["image", "mask"], name="EnsureChannelFirstd"),
    EnsureChannelFirstd(keys=["image", "mask"]),
    DebugShaped(keys=["image", "mask"], name="TransposeD"),
    TransposeD(keys=["image", "mask"], indices=(0, 3, 1, 2)),    # [1, D, H, W]
    DebugShaped(keys=["image", "mask"], name="ResizeD"),
    ResizeD(keys=["image", "mask"], spatial_size=(128, 256, 256), mode="trilinear", align_corners=True),
    DebugShaped(keys=["image", "mask"], name="Windowingd"),
    Windowingd(keys=["image"], window_center=-600, window_width=1500),
    DebugShaped(keys=["image", "mask"], name="ToTensord"),
    ToTensord(keys=["image", "mask"])
])

transforms_small_hu_m300_1400 = Compose([
    LoadImaged(keys=["image","mask"]),
    DebugShaped(keys=["image", "mask"], name="EnsureChannelFirstd"),
    EnsureChannelFirstd(keys=["image", "mask"]),
    DebugShaped(keys=["image", "mask"], name="TransposeD"),
    TransposeD(keys=["image", "mask"], indices=(0, 3, 1, 2)),    # [1, D, H, W]
    DebugShaped(keys=["image", "mask"], name="ResizeD"),
    ResizeD(keys=["image", "mask"], spatial_size=(128, 256, 256), mode="trilinear", align_corners=True),
    DebugShaped(keys=["image", "mask"], name="Windowingd"),
    Windowingd(keys=["image"], window_center=-300, window_width=1400),
    DebugShaped(keys=["image", "mask"], name="ToTensord"),
    ToTensord(keys=["image", "mask"])
])

transforms_small_multiwindowing = Compose([
    LoadImaged(keys=["image","mask"]),
    DebugShaped(keys=["image", "mask"], name="EnsureChannelFirstd"),
    EnsureChannelFirstd(keys=["image", "mask"]),
    DebugShaped(keys=["image", "mask"], name="TransposeD"),
    TransposeD(keys=["image", "mask"], indices=(0, 3, 1, 2)),    # [1, D, H, W]
    DebugShaped(keys=["image", "mask"], name="ResizeD"),
    ResizeD(keys=["image", "mask"], spatial_size=(128, 256, 256), mode="trilinear", align_corners=True),
    DebugShaped(keys=["image", "mask"], name="Multiwindowingd"),
    MultiWindowingd(
        keys=["image"],
        windows=[(-600, 1500), (40, 400), (-160, 600)]  # Pulmón, mediastino, consolidación
    ),
    DebugShaped(keys=["image", "mask"], name="ToTensord"),
    ToTensord(keys=["image", "mask"])
])

transforms_medium = Compose([
    LoadImaged(keys=["image","mask"]),
    DebugShaped(keys=["image", "mask"], name="EnsureChannelFirstd"),
    EnsureChannelFirstd(keys=["image", "mask"]),
    DebugShaped(keys=["image", "mask"], name="TransposeD"),
    TransposeD(keys=["image", "mask"], indices=(0, 3, 1, 2)),    # [1, D, H, W]
    DebugShaped(keys=["image", "mask"], name="ResizeD"),
    ResizeD(keys=["image", "mask"], spatial_size=(256, 512, 512), mode="trilinear", align_corners=True),
    DebugShaped(keys=["image", "mask"], name="ToTensord"),
    ToTensord(keys=["image", "mask"])
])

transforms_medium_hu_m600_1500 = Compose([
    LoadImaged(keys=["image","mask"]),
    DebugShaped(keys=["image", "mask"], name="EnsureChannelFirstd"),
    EnsureChannelFirstd(keys=["image", "mask"]),
    DebugShaped(keys=["image", "mask"], name="TransposeD"),
    TransposeD(keys=["image", "mask"], indices=(0, 3, 1, 2)),    # [1, D, H, W]
    DebugShaped(keys=["image", "mask"], name="ResizeD"),
    ResizeD(keys=["image", "mask"], spatial_size=(256, 512, 512), mode="trilinear", align_corners=True),
    DebugShaped(keys=["image", "mask"], name="Windowingd"),
    Windowingd(keys=["image"], window_center=-600, window_width=1500),
    DebugShaped(keys=["image", "mask"], name="ToTensord"),
    ToTensord(keys=["image", "mask"])
])

transforms_medium_hu_m300_1400 = Compose([
    LoadImaged(keys=["image","mask"]),
    DebugShaped(keys=["image", "mask"], name="EnsureChannelFirstd"),
    EnsureChannelFirstd(keys=["image", "mask"]),
    DebugShaped(keys=["image", "mask"], name="TransposeD"),
    TransposeD(keys=["image", "mask"], indices=(0, 3, 1, 2)),    # [1, D, H, W]
    DebugShaped(keys=["image", "mask"], name="ResizeD"),
    ResizeD(keys=["image", "mask"], spatial_size=(256, 512, 512), mode="trilinear", align_corners=True),
    DebugShaped(keys=["image", "mask"], name="Windowingd"),
    Windowingd(keys=["image"], window_center=-300, window_width=1400),
    DebugShaped(keys=["image", "mask"], name="ToTensord"),
    ToTensord(keys=["image", "mask"])
])

transforms_medium_multiwindowing = Compose([
    LoadImaged(keys=["image","mask"]),
    DebugShaped(keys=["image", "mask"], name="EnsureChannelFirstd"),
    EnsureChannelFirstd(keys=["image", "mask"]),
    DebugShaped(keys=["image", "mask"], name="TransposeD"),
    TransposeD(keys=["image", "mask"], indices=(0, 3, 1, 2)),    # [1, D, H, W]
    DebugShaped(keys=["image", "mask"], name="ResizeD"),
    ResizeD(keys=["image", "mask"], spatial_size=(256, 512, 512), mode="trilinear", align_corners=True),
    DebugShaped(keys=["image", "mask"], name="Multiwindowingd"),
    MultiWindowingd(
        keys=["image"],
        windows=[(-600, 1500), (40, 400), (-160, 600)]  # Pulmón, mediastino, consolidación
    ),
    DebugShaped(keys=["image", "mask"], name="ToTensord"),
    ToTensord(keys=["image", "mask"])
])

configs = {
    "resize_small": {
        "transforms": transforms_small,
        "save_separately": True,
        "save_format": ["npy", "nifti"],
    },

    "resize_small_hu_m600_1500_separadas": {
        "transforms": transforms_small_hu_m600_1500,
        "save_separately": True,
        "save_format": ["npy", "nifti"]
    },

    "resize_small_hu_m300_1400_separadas": {
        "transforms": transforms_small_hu_m300_1400,
        "save_separately": True,
        "save_format": ["npy", "nifti"]
    },

    "resize_medium": {
        "transforms": transforms_medium,
        "save_separately": True,
        "save_format": ["npy", "nifti"],
    },

    "resize_medium_hu_m600_1500_separadas": {
        "transforms": transforms_medium_hu_m600_1500,
        "save_separately": True,
        "save_format": ["npy", "nifti"]
    },

    "resize_medium_hu_m300_1400_separadas": {
        "transforms": transforms_medium_hu_m300_1400,
        "save_separately": True,
        "save_format": ["npy", "nifti"]
    },

    "resize_small_multiwindowing_separadas": {
        "transforms": transforms_small_multiwindowing,
        "save_separately": True,
        "save_format": ["npy", "nifti"]
    },

    "resize_medium_multiwindowing_separadas": {
        "transforms": transforms_medium_multiwindowing,
        "save_separately": True,
        "save_format": ["npy", "nifti"]
    }
}

run_preprocessing_configs(
    image_dir="../data/4_1julio/datos_nifti",
    mask_dir="../data/4_1julio/segmentaciones_nodulos",
    output_base_dir="../data/4_1julio/preprocesamientos",
    configs_dict=configs
)