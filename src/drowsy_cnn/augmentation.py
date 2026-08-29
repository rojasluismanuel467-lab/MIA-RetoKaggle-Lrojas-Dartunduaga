"""Pipelines de data augmentation con albumentations.

Diseño: función `build_transforms(aug_level)` produce el `A.Compose` apropiado
según el nivel de aug. Todas las transformaciones respetan bbox via `bbox_params`.

Estrategias:
    - none:              solo resize + normalize (baseline, val/test)
    - basic:             + HorizontalFlip + RandomBrightnessContrast
    - strong:            + ShiftScaleRotate + HueSat + GaussNoise + MotionBlur
    - cabin_realistic:   strong + shadows + sun flare + gamma amplio + oclusiones
                         (para robustez a variabilidad temporal + túneles + gafas/gorras)

Fuentes:
    - Albumentations docs oficiales (bbox_params, safe transforms)
    - Chollet cap 8 (aug moderna)
    - DrowsyDetectNet 2024 (CLAHE + brightness estándar)
    - Yu et al. 2019 arxiv:1910.09722 (condition-adaptive para día/noche/gafas)
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

from .config import IMAGENET_MEAN, IMAGENET_STD, TRAIN_MEAN, TRAIN_STD


def build_transforms(
    input_size: int = 224,
    aug_level: str = "basic",
    use_imagenet_stats: bool = True,
) -> A.Compose:
    """Compone el pipeline de aug + normalización + tensor conversion.

    Args:
        input_size: tamaño del resize final (cuadrado).
        aug_level: nivel de aug (none | basic | strong | cabin_realistic).
        use_imagenet_stats: normalizar con stats ImageNet (True para pretrained)
            o stats calculadas sobre train (False para CustomCNN).

    Returns:
        A.Compose con `bbox_params(format='pascal_voc', min_visibility=0.3)` que
        transforma consistentemente la imagen y su bbox asociada.
    """
    mean = IMAGENET_MEAN if use_imagenet_stats else TRAIN_MEAN
    std = IMAGENET_STD if use_imagenet_stats else TRAIN_STD

    ops: list = _build_ops(aug_level)
    ops += [
        A.Resize(input_size, input_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]
    return A.Compose(
        ops,
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_labels"],
            min_visibility=0.3,
        ),
    )


def _build_ops(aug_level: str) -> list:
    """Construye la lista de operaciones según nivel. Sin resize/normalize/tensor."""
    ops: list = []

    if aug_level in {"basic", "strong", "cabin_realistic"}:
        ops += [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        ]

    if aug_level in {"strong", "cabin_realistic"}:
        ops += [
            A.Affine(translate_percent=(-0.05, 0.05), scale=(0.9, 1.1), rotate=(-10, 10), p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
            A.GaussNoise(std_range=(0.04, 0.15), p=0.3),
            A.MotionBlur(blur_limit=5, p=0.3),
        ]

    if aug_level == "cabin_realistic":
        # Requisitos explícitos del proyecto (ver memory/project_cabin_conditions.md):
        # 1. Variabilidad temporal (hora del día): gamma amplio + brightness extremo
        # 2. Transiciones bruscas (túneles): shadow intenso + brightness extremo
        # 3. Oclusiones (gafas, gorras): CoarseDropout localizado
        ops += [
            # 1. Iluminación temporal
            A.RandomGamma(gamma_limit=(60, 160), p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.3, p=0.4),
            A.CLAHE(clip_limit=3.0, p=0.3),
            A.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.05, p=0.3),
            # 2. Túneles y contraluz
            A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_limit=(1, 3), p=0.4),
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 0.5), num_flare_circles_range=(1, 3), src_radius=100, p=0.15
            ),
            # 3. Oclusiones (gafas, gorras, mascarillas)
            A.CoarseDropout(
                num_holes_range=(1, 3), hole_height_range=(12, 32), hole_width_range=(16, 48), p=0.4
            ),
            # Robustez perceptual
            A.ImageCompression(quality_range=(75, 95), p=0.3),
            A.Downscale(scale_range=(0.75, 0.95), p=0.2),
        ]

    return ops


def build_transforms_crop(input_size: int = 224, aug_level: str = "strong") -> A.Compose:
    """Variante para el CropClassifier del §18 (no lleva bbox — solo crop de la img).

    Args:
        input_size: tamaño resize.
        aug_level: nivel de aug (none | basic | strong | cabin_realistic).

    Returns:
        A.Compose sin bbox_params (aplica solo a la imagen del crop).
    """
    ops = _build_ops(aug_level)
    ops += [
        A.Resize(input_size, input_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]
    return A.Compose(ops)


__all__ = ["build_transforms", "build_transforms_crop"]
