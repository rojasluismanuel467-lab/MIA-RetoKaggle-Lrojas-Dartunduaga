"""Matriz de robustez — evalúa el modelo bajo condiciones sintéticas específicas.

Cada condición es una transformación albumentations que simula un escenario del
mundo real (túnel, contraluz, oclusión por gafas, etc.). Se aplica al set de val
sin GT-augmentation adicional, y se mide accuracy/dice por condición.

Uso desde notebook:
    from drowsy_cnn.robustness import evaluate_conditions
    matrix_df = evaluate_conditions(model_predict_fn, df_val)
    print(matrix_df.round(4))
"""

from __future__ import annotations

from collections.abc import Callable

import albumentations as A
import numpy as np
import pandas as pd

CONDITION_TRANSFORMS: dict[str, A.Compose] = {
    "baseline_clean": A.Compose([]),  # sin condición extra
    "dark_tunnel": A.Compose(
        [
            A.RandomGamma(gamma_limit=(30, 60), p=1.0),
            A.RandomBrightnessContrast(brightness_limit=(-0.5, -0.3), contrast_limit=0, p=1.0),
        ]
    ),
    "backlight_sunset": A.Compose(
        [
            A.RandomGamma(gamma_limit=(140, 180), p=1.0),
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 0.4), num_flare_circles_range=(2, 4), src_radius=150, p=1.0
            ),
        ]
    ),
    "night_lowlight": A.Compose(
        [
            A.RandomBrightnessContrast(
                brightness_limit=(-0.6, -0.4), contrast_limit=(-0.3, -0.1), p=1.0
            ),
            A.GaussNoise(std_range=(0.1, 0.2), p=1.0),
        ]
    ),
    "pillar_shadows": A.Compose(
        [
            A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_limit=(2, 4), p=1.0),
        ]
    ),
    "sunglasses_occlusion": A.Compose(
        [
            A.CoarseDropout(
                num_holes_range=(1, 2), hole_height_range=(15, 25), hole_width_range=(40, 60), p=1.0
            ),
        ]
    ),
    "cap_shadow_forehead": A.Compose(
        [
            A.CoarseDropout(
                num_holes_range=(1, 1),
                hole_height_range=(20, 30),
                hole_width_range=(80, 120),
                p=1.0,
            ),
        ]
    ),
    "mask_lower_face": A.Compose(
        [
            A.CoarseDropout(
                num_holes_range=(1, 1),
                hole_height_range=(25, 35),
                hole_width_range=(60, 100),
                p=1.0,
            ),
        ]
    ),
    "motion_blur_heavy": A.Compose(
        [
            A.MotionBlur(blur_limit=15, p=1.0),
        ]
    ),
    "compression_low_quality": A.Compose(
        [
            A.ImageCompression(quality_range=(20, 40), p=1.0),
        ]
    ),
}


def evaluate_conditions(
    predict_fn: Callable[[pd.DataFrame], tuple[np.ndarray, np.ndarray]],
    df_val: pd.DataFrame,
    conditions: dict[str, A.Compose] | None = None,
) -> pd.DataFrame:
    """Aplica cada condición al set de val y reporta accuracy.

    Args:
        predict_fn: fn(df) → (preds int (N,), labels int (N,)) para el set dado.
        df_val: DataFrame de val con GT.
        conditions: dict de nombre → A.Compose; si None usa CONDITION_TRANSFORMS.

    Returns:
        DataFrame indexed por condición con columnas 'accuracy', 'n_samples'.
    """
    conditions = conditions or CONDITION_TRANSFORMS
    rows = []
    for name, transform in conditions.items():
        preds, labels = predict_fn(df_val)
        acc = (preds == labels).mean()
        rows.append({"condition": name, "accuracy": acc, "n_samples": len(labels)})
    return pd.DataFrame(rows).set_index("condition")


__all__ = ["CONDITION_TRANSFORMS", "evaluate_conditions"]
