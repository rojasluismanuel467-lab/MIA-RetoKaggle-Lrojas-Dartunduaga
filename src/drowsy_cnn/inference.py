"""Inferencia — predict con TTA hflip, ensemble multi-modelo, multi-scale, two-stage.

Todas las funciones son puras (no mutan estado global) y devuelven arrays numpy
listos para escribir el submission CSV.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .augmentation import build_transforms
from .checkpoints import load_checkpoint
from .config import CKPT_DIR, DEVICE, IDX2CLS, IMG_DIR, IMG_H, IMG_W, ExpConfig
from .dataset import DrowsyDataset, cxcywh_norm_to_xyxy
from .models import CustomCNN, build_pretrained


def load_model_from_result(exp: dict) -> tuple[torch.nn.Module, bool]:
    """Reconstruye modelo desde un dict result de train_one_config.

    Returns:
        (model_on_device, use_imagenet_stats).
    """
    cfg = ExpConfig(**exp["cfg"])
    if cfg.model_kind == "custom":
        model = CustomCNN(dropout=cfg.dropout_head)
        use_imagenet_stats = False
    else:
        model = build_pretrained(cfg.model_kind, freeze=False, dropout_head=cfg.dropout_head)
        use_imagenet_stats = True
    load_checkpoint(model, exp["ckpt"], DEVICE)
    return model.to(DEVICE), use_imagenet_stats


@torch.no_grad()
def predict_with_tta(
    model: torch.nn.Module,
    df_test: pd.DataFrame,
    input_size: int = 224,
    use_imagenet_stats: bool = True,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Predice sobre df_test con TTA horizontal flip.

    Args:
        model: modelo con forward → (logits, bbox_cxcywh_norm).
        df_test: DataFrame con columna filename.
        input_size: resize target.
        use_imagenet_stats: normalización.
        batch_size: batch size.

    Returns:
        DataFrame con filename, class, xmin, ymin, xmax, ymax (formato Kaggle).
    """
    model.eval()
    transform = build_transforms(input_size, "none", use_imagenet_stats)
    dataset = DrowsyDataset(df_test, IMG_DIR, transform, is_test=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_logits, all_bbox, all_files = [], [], []
    for img_bchw, filenames in loader:
        img_bchw = img_bchw.to(DEVICE)
        logits_1, bbox_1 = model(img_bchw)
        logits_2, bbox_2 = model(torch.flip(img_bchw, dims=[-1]))
        bbox_2_deflipped = bbox_2.clone()
        bbox_2_deflipped[:, 0] = 1.0 - bbox_2[:, 0]  # cx flip
        all_logits.append((logits_1 + logits_2).cpu() / 2)
        all_bbox.append((bbox_1 + bbox_2_deflipped).cpu() / 2)
        all_files.extend(filenames)

    logits_all = torch.cat(all_logits)
    bbox_cxcywh_norm_all = torch.cat(all_bbox)
    xyxy = cxcywh_norm_to_xyxy(bbox_cxcywh_norm_all, IMG_W, IMG_H)
    xyxy[:, 0::2].clamp_(0, IMG_W)
    xyxy[:, 1::2].clamp_(0, IMG_H)

    return pd.DataFrame(
        {
            "filename": all_files,
            "class": [IDX2CLS[p.item()] for p in logits_all.argmax(1)],
            "xmin": xyxy[:, 0].round().int().tolist(),
            "ymin": xyxy[:, 1].round().int().tolist(),
            "xmax": xyxy[:, 2].round().int().tolist(),
            "ymax": xyxy[:, 3].round().int().tolist(),
        }
    )


@torch.no_grad()
def ensemble_predict(
    model_list: list[torch.nn.Module],
    df: pd.DataFrame,
    input_size: int = 224,
    use_imagenet_stats: bool = True,
    batch_size: int = 16,
    with_bbox_true: bool = False,
) -> dict:
    """Ensemble por promedio de logits + bbox con TTA hflip.

    Args:
        model_list: lista de modelos (todos con misma firma multitarea).
        df: DataFrame de eval (train/val) o test (is_test según with_bbox_true).
        input_size: resize.
        use_imagenet_stats: normalización.
        batch_size: batch size.
        with_bbox_true: si True, df tiene bbox GT y devuelve labels/bbox_true.

    Returns:
        dict con probs (N,2), preds (N,), bbox_norm (N,4), y opcionalmente
        labels + bbox_true, o filenames si is_test.
    """
    transform = build_transforms(input_size, "none", use_imagenet_stats)
    dataset = DrowsyDataset(df, IMG_DIR, transform, is_test=not with_bbox_true)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_probs, all_bbox = [], []
    all_files, all_labels, all_bbox_true = [], [], []
    for batch in loader:
        if with_bbox_true:
            img_bchw, label_b, bbox_true_b4 = batch
            all_labels.append(label_b)
            all_bbox_true.append(bbox_true_b4)
        else:
            img_bchw, filenames = batch
            all_files.extend(filenames)
        img_bchw = img_bchw.to(DEVICE)
        img_flipped = torch.flip(img_bchw, dims=[-1])

        batch_logits, batch_bbox = [], []
        for m in model_list:
            m.eval()
            logits_1, bbox_1 = m(img_bchw)
            logits_2, bbox_2 = m(img_flipped)
            bbox_2_deflipped = bbox_2.clone()
            bbox_2_deflipped[:, 0] = 1.0 - bbox_2[:, 0]
            batch_logits.append((logits_1 + logits_2) / 2)
            batch_bbox.append((bbox_1 + bbox_2_deflipped) / 2)

        logits_avg = torch.stack(batch_logits).mean(0)
        bbox_avg = torch.stack(batch_bbox).mean(0)
        all_probs.append(F.softmax(logits_avg, dim=1).cpu())
        all_bbox.append(bbox_avg.cpu())

    probs = torch.cat(all_probs).numpy()
    bbox = torch.cat(all_bbox).numpy()
    out = {"probs": probs, "preds": probs.argmax(1), "bbox_norm": bbox}
    if with_bbox_true:
        out["labels"] = torch.cat(all_labels).numpy()
        out["bbox_true"] = torch.cat(all_bbox_true).numpy()
    else:
        out["filenames"] = all_files
    return out


@torch.no_grad()
def stage1_predict_bboxes(
    model_list: list[torch.nn.Module], df: pd.DataFrame, is_test: bool = False
) -> np.ndarray:
    """Stage 1 del two-stage: predice bboxes (xmin,ymin,xmax,ymax) en pixels absolutos.

    Args:
        model_list: modelos que hacen bbox regression (mega ensemble).
        df: DataFrame con filenames.
        is_test: True si df no tiene labels.

    Returns:
        (N, 4) array numpy con xyxy en pixels absolutos, clipped a IMG_W×IMG_H.
    """
    accum_bbox = 0.0
    for m in model_list:
        m.eval()
        transform = build_transforms(224, "none", use_imagenet_stats=True)
        dataset = DrowsyDataset(df, IMG_DIR, transform, is_test=is_test)
        loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
        batch_bboxes = []
        for batch in loader:
            img_bchw = batch[0].to(DEVICE)
            _, bbox_1 = m(img_bchw)
            _, bbox_2 = m(torch.flip(img_bchw, dims=[-1]))
            bbox_2_deflipped = bbox_2.clone()
            bbox_2_deflipped[:, 0] = 1.0 - bbox_2[:, 0]
            batch_bboxes.append(((bbox_1 + bbox_2_deflipped) / 2).cpu())
        accum_bbox = accum_bbox + torch.cat(batch_bboxes).numpy()

    bbox_cxcywh_norm_avg = accum_bbox / len(model_list)
    xyxy = cxcywh_norm_to_xyxy(torch.tensor(bbox_cxcywh_norm_avg), IMG_W, IMG_H)
    xyxy[:, 0::2].clamp_(0, IMG_W)
    xyxy[:, 1::2].clamp_(0, IMG_H)
    return xyxy.numpy()


def load_stage1_models(names_and_kinds: list[tuple[str, str]]) -> list[torch.nn.Module]:
    """Carga backbones del stage 1 (mega ensemble) desde CKPT_DIR.

    Args:
        names_and_kinds: lista de (nombre_ckpt_sin_ext, backbone_kind).

    Returns:
        Lista de modelos cargados en DEVICE. Los que no existen se saltan.
    """
    loaded = []
    for name, kind in names_and_kinds:
        ckpt = CKPT_DIR / f"{name}.pt"
        if not ckpt.exists():
            continue
        model = build_pretrained(kind, freeze=False)
        load_checkpoint(model, ckpt, DEVICE)
        loaded.append(model.to(DEVICE))
    return loaded


__all__ = [
    "load_model_from_result",
    "predict_with_tta",
    "ensemble_predict",
    "stage1_predict_bboxes",
    "load_stage1_models",
]
