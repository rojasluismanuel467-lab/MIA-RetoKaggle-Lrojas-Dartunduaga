"""Smoke tests — verifican que el pipeline no está roto.

Ejecutar con: `pytest tests/`
"""

from __future__ import annotations

import pytest
import torch

from drowsy_cnn.augmentation import build_transforms, build_transforms_crop
from drowsy_cnn.config import CLASSES, IMG_H, IMG_W, ExpConfig
from drowsy_cnn.dataset import cxcywh_norm_to_xyxy, xyxy_to_cxcywh_norm
from drowsy_cnn.losses import EarlyStopping, MultitaskLoss, bbox_dice, bbox_iou
from drowsy_cnn.models import CropClassifier, CustomCNN, build_pretrained


# =============================================================================
# Config + dataclasses
# =============================================================================
def test_expconfig_defaults() -> None:
    cfg = ExpConfig(name="test")
    assert cfg.model_kind == "custom"
    assert cfg.input_size == 224
    assert cfg.seed == 42


def test_classes_alphabetical() -> None:
    """awake < drowsy — orden alfabético fijo."""
    assert CLASSES == ("awake", "drowsy")


# =============================================================================
# Bbox conversions — round-trip debe ser identidad
# =============================================================================
def test_bbox_roundtrip() -> None:
    original_xyxy = [500, 300, 1000, 800]
    cxcywh = xyxy_to_cxcywh_norm(original_xyxy, IMG_W, IMG_H)
    recovered = cxcywh_norm_to_xyxy(torch.tensor(cxcywh), IMG_W, IMG_H).tolist()
    for a, b in zip(original_xyxy, recovered):
        assert abs(a - b) < 1e-3


# =============================================================================
# IoU / Dice — bboxes idénticas → 1.0
# =============================================================================
def test_iou_identical_is_one() -> None:
    bbox = torch.tensor([[0.5, 0.5, 0.3, 0.4]])
    assert bbox_iou(bbox, bbox).item() == pytest.approx(1.0, abs=1e-5)


def test_dice_relation_to_iou() -> None:
    """Dice = 2·IoU / (1 + IoU) — monotonic en IoU."""
    pred = torch.tensor([[0.5, 0.5, 0.3, 0.4]])
    true = torch.tensor([[0.6, 0.5, 0.3, 0.4]])
    iou = bbox_iou(pred, true).item()
    dice = bbox_dice(pred, true).item()
    assert dice == pytest.approx(2 * iou / (1 + iou), abs=1e-5)


# =============================================================================
# Modelos — forward no crashea y devuelve shapes correctas
# =============================================================================
def test_custom_cnn_forward() -> None:
    model = CustomCNN()
    x = torch.randn(2, 3, 224, 224)
    logits, bbox = model(x)
    assert logits.shape == (2, 2)
    assert bbox.shape == (2, 4)
    assert (bbox >= 0).all() and (bbox <= 1).all(), "bbox debe estar en [0,1] (Sigmoid)"


@pytest.mark.parametrize("name", ["resnet18", "mobilenet_v3_small", "efficientnet_b0"])
def test_pretrained_backbones_forward(name: str) -> None:
    model = build_pretrained(name, freeze=True)
    logits, bbox = model(torch.randn(1, 3, 224, 224))
    assert logits.shape == (1, 2)
    assert bbox.shape == (1, 4)


def test_crop_classifier_forward() -> None:
    model = CropClassifier()
    logits = model(torch.randn(2, 3, 224, 224))
    assert logits.shape == (2, 2)


def test_build_pretrained_invalid() -> None:
    with pytest.raises(ValueError, match="desconocido"):
        build_pretrained("nonexistent_backbone")


# =============================================================================
# Loss — output shapes y positivo
# =============================================================================
def test_multitask_loss_returns_positive_scalar() -> None:
    loss_fn = MultitaskLoss()
    logits = torch.randn(4, 2)
    bbox_pred = torch.rand(4, 4)
    labels = torch.randint(0, 2, (4,))
    bbox_true = torch.rand(4, 4)
    losses = loss_fn(logits, bbox_pred, labels, bbox_true)
    for key in ["total", "cls", "bbox", "l1", "giou"]:
        assert key in losses
        assert losses[key].ndim == 0, f"{key} debe ser scalar"
        assert torch.isfinite(losses[key]), f"{key} no debe ser NaN/Inf"


# =============================================================================
# EarlyStopping — behavior
# =============================================================================
def test_early_stopping_triggers_after_patience() -> None:
    stopper = EarlyStopping(patience=3, min_delta=1e-3, mode="min")
    assert not stopper.step(1.0)  # initial best
    assert not stopper.step(0.9)  # improved
    for _ in range(3):
        stop = stopper.step(0.9)  # no improvement
    assert stop


def test_early_stopping_invalid_mode() -> None:
    with pytest.raises(ValueError, match="mode"):
        EarlyStopping(mode="invalid")


# =============================================================================
# Augmentations — pipeline compila y produce tensor
# =============================================================================
@pytest.mark.parametrize("aug_level", ["none", "basic", "strong", "cabin_realistic"])
def test_augmentation_pipeline(aug_level: str) -> None:
    import numpy as np

    transform = build_transforms(224, aug_level, use_imagenet_stats=True)
    # Simular imagen + bbox
    img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    out = transform(image=img, bboxes=[[50, 50, 200, 200]], class_labels=["awake"])
    assert out["image"].shape == (3, 224, 224)


def test_crop_transform_pipeline() -> None:
    import numpy as np

    transform = build_transforms_crop(224, "cabin_realistic")
    img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    out = transform(image=img)
    assert out["image"].shape == (3, 224, 224)
