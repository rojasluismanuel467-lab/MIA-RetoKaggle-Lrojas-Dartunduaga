"""Modelos CNN: CustomCNN (desde cero), MultitaskWrapper (backbone preentrenado), CropClassifier.

Todas las arquitecturas emiten 2 outputs cuando aplica:
    logits ∈ (B, 2)     — clasificación awake/drowsy
    bbox   ∈ (B, 4)     — (cx, cy, w, h) normalizado en [0, 1] via Sigmoid

Referencias:
    - Raschka cap 14 (patrón Conv-BN-ReLU-Pool-Dropout)
    - He et al. 2015 arxiv:1502.01852 (Kaiming init)
    - Chollet cap 9 (GlobalAveragePooling)
    - Géron cap 14 (multitask classification + localization)
    - Girshick 2014 arxiv:1311.2524 (R-CNN, base del two-stage)
"""

from __future__ import annotations

import timm
import torch
from torch import nn
from torchvision import models


class CustomCNN(nn.Module):
    """CNN multitarea desde cero — Punto 1 de la rúbrica.

    Arquitectura:
        4× [Conv3x3(pad=1) → BN → ReLU → MaxPool2 → Dropout] canales 32→64→128→256
        → GlobalAveragePooling → head_cls: Linear(2), head_bbox: Linear(4)+Sigmoid
    """

    def __init__(self, n_classes: int = 2, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            self._conv_block(3, 32, dropout),
            self._conv_block(32, 64, dropout),
            self._conv_block(64, 128, dropout),
            self._conv_block(128, 256, dropout),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head_cls = nn.Linear(256, n_classes)
        self.head_bbox = nn.Sequential(nn.Linear(256, 4), nn.Sigmoid())
        self._init_weights()

    @staticmethod
    def _conv_block(in_c: int, out_c: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )

    def _init_weights(self) -> None:
        """He/Kaiming init para conv+linear tras ReLU (Géron cap 11)."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(x)
        pooled = self.pool(features).flatten(1)
        return self.head_cls(pooled), self.head_bbox(pooled)


class MultitaskWrapper(nn.Module):
    """Wrapper que añade 2 heads (cls + bbox) sobre cualquier backbone que produce features."""

    def __init__(
        self, backbone: nn.Module, feat_dim: int, n_classes: int = 2, dropout_head: float = 0.3
    ):
        super().__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_head)
        self.head_cls = nn.Linear(feat_dim, n_classes)
        self.head_bbox = nn.Sequential(nn.Linear(feat_dim, 4), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        if features.ndim == 4:
            features = self.pool(features).flatten(1)
        features = self.dropout(features)
        return self.head_cls(features), self.head_bbox(features)


class CropClassifier(nn.Module):
    """EfficientNet-B0 puro para clasificación binaria en el two-stage (§18).

    Recibe el CROP del bbox como input — no necesita head de bbox.
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            num_classes=0,
            global_pool="",
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.backbone.num_features, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.pool(self.backbone(x)).flatten(1)
        return self.head(self.dropout(features))


def build_pretrained(
    name: str, n_classes: int = 2, dropout_head: float = 0.3, freeze: bool = True
) -> MultitaskWrapper:
    """Factory para backbones preentrenados con heads multitarea.

    Args:
        name: uno de {resnet18, mobilenet_v3_small, efficientnet_b0}.
        n_classes: número de clases para la head_cls.
        dropout_head: dropout antes de las heads.
        freeze: si True, backbone.parameters().requires_grad = False (fase A).

    Returns:
        MultitaskWrapper con el backbone envuelto.

    Raises:
        ValueError si `name` no es soportado.
    """
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feat_dim = m.fc.in_features
        m.fc = nn.Identity()
    elif name == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        feat_dim = m.classifier[0].in_features
        m.classifier = nn.Identity()
    elif name == "efficientnet_b0":
        m = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0, global_pool="")
        feat_dim = m.num_features
    else:
        raise ValueError(
            f"Backbone desconocido: {name!r}. "
            "Soportados: resnet18, mobilenet_v3_small, efficientnet_b0",
        )

    if freeze:
        for p in m.parameters():
            p.requires_grad = False

    return MultitaskWrapper(m, feat_dim=feat_dim, n_classes=n_classes, dropout_head=dropout_head)


def unfreeze_last_stage(model: MultitaskWrapper, backbone_name: str) -> None:
    """Descongela el último stage del backbone para fase B fine-tuning.

    Args:
        model: MultitaskWrapper con backbone congelado.
        backbone_name: nombre del backbone (resnet18 | mobilenet_v3_small | efficientnet_b0).
    """
    if backbone_name == "resnet18":
        for p in model.backbone.layer4.parameters():
            p.requires_grad = True
    elif backbone_name == "mobilenet_v3_small":
        for p in model.backbone.features[-3:].parameters():
            p.requires_grad = True
    elif backbone_name == "efficientnet_b0":
        for p in model.backbone.blocks[-2:].parameters():
            p.requires_grad = True


__all__ = [
    "CustomCNN",
    "MultitaskWrapper",
    "CropClassifier",
    "build_pretrained",
    "unfreeze_last_stage",
]
