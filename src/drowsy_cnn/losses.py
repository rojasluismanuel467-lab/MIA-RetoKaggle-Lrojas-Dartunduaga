"""Losses + métricas + early stopping.

Referencias:
    - Chollet cap 12 (IoU implementation)
    - torchvision.ops.generalized_box_iou_loss (GIoU)
    - DETR paper 2020 arxiv:2005.12872 (loss combinada SmoothL1 + GIoU)
    - Keras EarlyStopping docs
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torchvision.ops import generalized_box_iou_loss

from .dataset import cxcywh_norm_to_xyxy


def bbox_iou(
    pred_cxcywh: torch.Tensor, true_cxcywh: torch.Tensor, eps: float = 1e-7
) -> torch.Tensor:
    """IoU batched entre pares de bboxes en formato (cx, cy, w, h).

    Args:
        pred_cxcywh: (N, 4) predicciones.
        true_cxcywh: (N, 4) targets.
        eps: numerical stability.

    Returns:
        (N,) IoU values ∈ [0, 1].
    """

    def _to_xyxy(bbox: torch.Tensor) -> torch.Tensor:
        cx, cy, w, h = bbox.unbind(-1)
        return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)

    pred_xyxy = _to_xyxy(pred_cxcywh)
    true_xyxy = _to_xyxy(true_cxcywh)
    x1 = torch.max(pred_xyxy[..., 0], true_xyxy[..., 0])
    y1 = torch.max(pred_xyxy[..., 1], true_xyxy[..., 1])
    x2 = torch.min(pred_xyxy[..., 2], true_xyxy[..., 2])
    y2 = torch.min(pred_xyxy[..., 3], true_xyxy[..., 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_pred = (pred_xyxy[..., 2] - pred_xyxy[..., 0]) * (pred_xyxy[..., 3] - pred_xyxy[..., 1])
    area_true = (true_xyxy[..., 2] - true_xyxy[..., 0]) * (true_xyxy[..., 3] - true_xyxy[..., 1])
    return inter / (area_pred + area_true - inter + eps)


def bbox_dice(pred_cxcywh: torch.Tensor, true_cxcywh: torch.Tensor) -> torch.Tensor:
    """Dice coefficient = 2·IoU / (1 + IoU) — monotónica en IoU.

    Args:
        pred_cxcywh: (N, 4).
        true_cxcywh: (N, 4).

    Returns:
        (N,) Dice values ∈ [0, 1].
    """
    iou = bbox_iou(pred_cxcywh, true_cxcywh)
    return 2 * iou / (1 + iou)


class MultitaskLoss(nn.Module):
    """L = w_cls·CE + w_bbox·(SmoothL1 + λ_giou·GIoU) — combo estilo DETR.

    Args:
        w_cls: peso de la loss de clasificación.
        w_bbox: peso global de la loss de bbox (aplicado a SmoothL1 + λ·GIoU).
        lambda_giou: peso relativo de GIoU dentro de la loss de bbox.
        class_weights: opcional (2,) para desbalance de clases.
        label_smoothing: 0-1, regulariza confianza (Müller et al. 2019).
    """

    def __init__(
        self,
        w_cls: float = 1.0,
        w_bbox: float = 5.0,
        lambda_giou: float = 2.0,
        class_weights: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.w_cls = w_cls
        self.w_bbox = w_bbox
        self.lambda_giou = lambda_giou
        self.ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        self.smooth_l1 = nn.SmoothL1Loss()

    def forward(
        self,
        logits: torch.Tensor,
        bbox_pred_cxcywh: torch.Tensor,
        labels: torch.Tensor,
        bbox_true_cxcywh: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        loss_cls = self.ce(logits, labels)
        loss_l1 = self.smooth_l1(bbox_pred_cxcywh, bbox_true_cxcywh)

        # GIoU necesita xyxy — trabajamos en [0,1] (scale-invariant)
        pred_xyxy = cxcywh_norm_to_xyxy(bbox_pred_cxcywh, img_w=1, img_h=1)
        true_xyxy = cxcywh_norm_to_xyxy(bbox_true_cxcywh, img_w=1, img_h=1)
        loss_giou = generalized_box_iou_loss(pred_xyxy, true_xyxy, reduction="mean")

        loss_bbox = loss_l1 + self.lambda_giou * loss_giou
        total = self.w_cls * loss_cls + self.w_bbox * loss_bbox

        return {
            "total": total,
            "cls": loss_cls,
            "bbox": loss_bbox,
            "l1": loss_l1,
            "giou": loss_giou,
        }


class EarlyStopping:
    """Detiene entrenamiento si la métrica monitoreada no mejora en `patience` epochs.

    Args:
        patience: número de epochs sin mejora antes de parar.
        min_delta: mejora mínima para contar como "mejora".
        mode: 'min' para pérdida, 'max' para métricas positivas (acc, dice).
    """

    def __init__(self, patience: int = 12, min_delta: float = 1e-3, mode: str = "min"):
        if mode not in {"min", "max"}:
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = math.inf if mode == "min" else -math.inf
        self.counter = 0

    def step(self, value: float) -> bool:
        """Registra un valor. Returns True si hay que parar."""
        improved = (self.mode == "min" and value < self.best - self.min_delta) or (
            self.mode == "max" and value > self.best + self.min_delta
        )
        if improved:
            self.best = value
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


__all__ = ["bbox_iou", "bbox_dice", "MultitaskLoss", "EarlyStopping"]
