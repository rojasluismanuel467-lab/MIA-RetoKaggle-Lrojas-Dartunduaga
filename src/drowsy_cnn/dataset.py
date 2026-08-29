"""Datasets PyTorch para el proyecto — unificados, sin duplicación.

- `DrowsyDataset`: multitarea (bbox + clase). Se usa en stages full-image.
- `CropDataset`: recibe imagen ya recortada al bbox. Se usa en stage 2 del two-stage.

Ambos aceptan el flag `is_test` para modo inferencia (sin labels).
"""

from __future__ import annotations

from pathlib import Path

import albumentations as A
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import CLS2IDX, IMG_H, IMG_W


def xyxy_to_cxcywh_norm(bbox_xyxy: list[float], img_w: int, img_h: int) -> list[float]:
    """Convierte bbox pascal_voc absoluto → (cx, cy, w, h) normalizado a [0,1]."""
    x1, y1, x2, y2 = bbox_xyxy
    return [((x1 + x2) / 2) / img_w, ((y1 + y2) / 2) / img_h, (x2 - x1) / img_w, (y2 - y1) / img_h]


def cxcywh_norm_to_xyxy(
    bbox_cxcywh_norm: torch.Tensor, img_w: int = IMG_W, img_h: int = IMG_H
) -> torch.Tensor:
    """Inverso: (cx, cy, w, h) norm → (xmin, ymin, xmax, ymax) en pixels absolutos.

    Acepta tanto tensor (4,) como batch (N, 4).
    """
    cx, cy, w, h = bbox_cxcywh_norm.unbind(-1)
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return torch.stack([x1, y1, x2, y2], dim=-1)


class DrowsyDataset(Dataset):
    """Dataset multitarea: devuelve (img_tensor, label_int, bbox_cxcywh_norm).

    Attributes:
        df: DataFrame con columnas filename, class, xmin, ymin, xmax, ymax.
        img_dir: directorio con las imágenes.
        transform: A.Compose con bbox_params.
        is_test: si True, retorna (img_tensor, filename) sin labels.
    """

    def __init__(
        self, df: pd.DataFrame, img_dir: Path, transform: A.Compose, is_test: bool = False
    ):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.is_test = is_test

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_bgr = cv2.imread(str(self.img_dir / row["filename"]))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if self.is_test:
            # bbox dummy centrada; será ignorada en inferencia.
            bboxes = [[IMG_W * 0.4, IMG_H * 0.4, IMG_W * 0.6, IMG_H * 0.6]]
            class_labels = ["awake"]
        else:
            bboxes = [[row.xmin, row.ymin, row.xmax, row.ymax]]
            class_labels = [row["class"]]

        out = self.transform(image=img_rgb, bboxes=bboxes, class_labels=class_labels)
        img_tensor = out["image"]

        if self.is_test:
            return img_tensor, row["filename"]

        # bbox filtrada por min_visibility → fallback al original
        aug_bbox = out["bboxes"][0] if out["bboxes"] else bboxes[0]
        h, w = img_tensor.shape[-2:]
        bbox_cxcywh_norm = torch.tensor(
            xyxy_to_cxcywh_norm(list(aug_bbox), img_w=w, img_h=h),
            dtype=torch.float32,
        )
        label = torch.tensor(CLS2IDX[out["class_labels"][0]], dtype=torch.long)
        return img_tensor, label, bbox_cxcywh_norm


class CropDataset(Dataset):
    """Dataset para el CropClassifier del §18 — recorta la imagen al bbox con margen.

    Attributes:
        df: DataFrame con filename, class, xmin, ymin, xmax, ymax.
            En val/test los xmin/ymin/xmax/ymax son bbox PREDICHOS por el stage 1.
        img_dir: directorio con las imágenes.
        transform: A.Compose (sin bbox_params, ya que trabajamos solo con crop).
        margin: fracción del bbox a expandir para preservar contexto.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: Path,
        transform: A.Compose,
        margin: float = 0.15,
        is_test: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.margin = margin
        self.is_test = is_test

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_bgr = cv2.imread(str(self.img_dir / row["filename"]))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        full_h, full_w = img_rgb.shape[:2]

        bbox_w = row.xmax - row.xmin
        bbox_h = row.ymax - row.ymin
        margin_x = bbox_w * self.margin
        margin_y = bbox_h * self.margin
        x1 = int(max(0, row.xmin - margin_x))
        y1 = int(max(0, row.ymin - margin_y))
        x2 = int(min(full_w, row.xmax + margin_x))
        y2 = int(min(full_h, row.ymax + margin_y))

        crop = img_rgb[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else img_rgb
        out = self.transform(image=crop)

        if self.is_test:
            return out["image"], row["filename"]
        label = torch.tensor(CLS2IDX[row["class"]], dtype=torch.long)
        return out["image"], label


__all__ = [
    "DrowsyDataset",
    "CropDataset",
    "xyxy_to_cxcywh_norm",
    "cxcywh_norm_to_xyxy",
]
