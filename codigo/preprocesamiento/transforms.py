
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd,
    ResizeWithPadOrCropd, RandFlipd, RandAffined, RandGaussianNoised, ToTensord, ResizeD, TransposeD,
)
from monai.transforms.transform import MapTransform
import numpy as np

class Windowingd(MapTransform):
    def __init__(self, keys, window_center, window_width):
        super().__init__(keys)
        self.center = window_center
        self.width = window_width

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            lower = self.center - self.width / 2
            upper = self.center + self.width / 2
            img = np.clip(img, lower, upper)
            img = (img - lower) / (upper - lower)  # scale to [0, 1]
            d[key] = img
        return d
    
from monai.transforms import MapTransform
import numpy as np

class MultiWindowingd(MapTransform):
    def __init__(self, keys, windows):
        """
        Aplica múltiples ventanas HU a una imagen y las apila como canales.

        Args:
            keys: claves a transformar, típicamente ["image"]
            windows: lista de tuplas (center, width)
        """
        super().__init__(keys)
        self.windows = windows

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            meta = d.get(f"{key}_meta_dict", {})

            # Asegurar que img tiene forma [C, D, H, W] o [D, H, W]
            if img.ndim == 4 and img.shape[0] == 1:
                img = img[0]  # Eliminar canal singleton
            elif img.ndim != 3:
                raise ValueError(f"Esperada imagen 3D, pero se obtuvo shape {img.shape}")

            out_channels = []
            for center, width in self.windows:
                hu_min = center - width / 2
                hu_max = center + width / 2
                clipped = np.clip(img, hu_min, hu_max)
                normalized = (clipped - hu_min) / (hu_max - hu_min)
                out_channels.append(normalized.astype(np.float32))

            stacked = np.stack(out_channels, axis=0)  # [C=3, D, H, W]
            d[key] = stacked

            # Actualizar metadata para MONAI
            meta["spatial_shape"] = list(stacked.shape[1:])
            meta["original_channel_dim"] = 0
            d[f"{key}_meta_dict"] = meta
        return d


# Tamaño final uniforme para todos los volúmenes
target_shape = (128, 256, 256)

from monai.transforms import MapTransform

class DebugShaped(MapTransform):
    def __init__(self, keys, name):
        super().__init__(keys)
        self.name = name

    def __call__(self, data):
        for key in self.keys:
            img = data[key]
            print(f"[DEBUG] Shape before {self.name} ({key}): {img.shape}")
        return data


class CombineMaskAndImage(MapTransform):
    def __init__(self, image_key="image", mask_key="mask", mode="mask_zero_out", default_value=-2000):
        super().__init__([image_key, mask_key])
        self.image_key = image_key
        self.mask_key = mask_key
        self.mode = mode
        self.default_value = default_value

    def __call__(self, data):
        d = dict(data)
        img = d[self.image_key]
        mask = d[self.mask_key]

        # Ambos deberían venir como [1, D, H, W] tras EnsureChannelFirstd
        if not (img.ndim == 4 and mask.ndim == 4):
            raise ValueError(f"Esperado [1, D, H, W] pero se obtuvo img={img.shape}, mask={mask.shape}")

        if self.mode == "stack":
            d[self.image_key] = np.concatenate([img, mask], axis=0)  # [2, D, H, W]

        elif self.mode == "mask_zero_out":
            img_masked = np.where(mask[0] > 0, img[0], self.default_value)
            d[self.image_key] = img_masked[None, ...]  # vuelve a [1, D, H, W]

        return d


CT_train_transforms = Compose([
    LoadImaged(keys=["image"]),  # (H, W, D)
    EnsureChannelFirstd(keys=["image"]),    # (1, H, W, D)             
    TransposeD(keys=["image"], indices=(0, 3, 1, 2)),    # [1, D, H, W]
    # DebugShaped(keys=["image"]),
    ResizeD(keys=["image"], spatial_size=target_shape, mode="trilinear", align_corners=True), # (1, 128, 256, 256)
    # Windowingd(keys=["image"], window_center=-600, window_width=1500),
    RandAffined(
    keys=["image"],
        prob=0.5,
        rotate_range=(0.15, 0.15, 0.15),   # ~3º en cada eje
        scale_range=(0.15, 0.15, 0.15),    # hasta ±5% de zoom
        translate_range=(5, 5, 5),         # opcional, en voxels
        mode="bilinear",
        padding_mode="border"
    ),
    MultiWindowingd(
        keys=["image"],
        windows=[(-600, 1500), (40, 400), (-160, 600)]  # Pulmón, mediastino, consolidación  # (C, 128, 256, 256)
    ),
    #ResizeWithPadOrCropd(keys=["image"], spatial_size=target_shape),
    #RandFlipd(keys=["image"], spatial_axis=0, prob=0.5),
    # RandAffined(
    #     keys=["image"],
    #     prob=0.5,
    #     rotate_range=[0.1, 0.1, 0.1],
    #     scale_range=[0.1, 0.1, 0.1],
    #     mode='bilinear'
    # ),
    #RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.01),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    ToTensord(keys=["image"]),
])


CT_val_transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    TransposeD(keys=["image"], indices=(0, 3, 1, 2)),    # [1, D, H, W]
    ResizeD(keys=["image"], spatial_size=target_shape, mode="trilinear", align_corners=True),
    # Windowingd(keys=["image"], window_center=-600, window_width=1500),
    MultiWindowingd(
        keys=["image"],
        windows=[(-600, 1500), (40, 400), (-160, 600)]  # Pulmón, mediastino, consolidación
    ),
    # ResizeWithPadOrCropd(keys=["image"], spatial_size=target_shape),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    ToTensord(keys=["image"]),
])