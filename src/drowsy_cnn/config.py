"""Configuración global del proyecto — constantes, paths y dataclass de experimentos.

Detección automática de entorno Kaggle vs local, para que el mismo notebook corra
sin cambios en ambos sitios.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch


# =============================================================================
# Detección del entorno + rutas
# =============================================================================
def detect_kaggle() -> bool:
    """True si el código corre en un kernel Kaggle."""
    return Path("/kaggle/input").exists() or "KAGGLE_KERNEL_RUN_TYPE" in os.environ


IS_KAGGLE = detect_kaggle()

if IS_KAGGLE:
    # Layout Kaggle: input read-only, working para outputs.
    _kaggle_inputs = list(Path("/kaggle/input").iterdir()) if Path("/kaggle/input").exists() else []
    DATA_DIR = _kaggle_inputs[0] if _kaggle_inputs else Path("/kaggle/input")
    IMG_DIR = DATA_DIR / "images"
    OUT_DIR = Path("/kaggle/working")
    CKPT_DIR = Path("/kaggle/working/checkpoints")
else:
    ROOT = Path(__file__).resolve().parents[2]  # …/MIA-RetoKaggle-Lrojas-Dartunduaga
    DATA_DIR = ROOT / "data"
    IMG_DIR = DATA_DIR / "images"
    OUT_DIR = ROOT / "outputs"
    CKPT_DIR = ROOT / "checkpoints"

for _d in (OUT_DIR, CKPT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Constantes del problema
# =============================================================================
IMG_W, IMG_H = 1920, 1080

CLASSES: tuple[str, ...] = ("awake", "drowsy")
N_CLASSES = len(CLASSES)
CLS2IDX: dict[str, int] = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS: dict[int, str] = {i: c for c, i in CLS2IDX.items()}

# Stats de normalización — ImageNet (usar con backbones preentrenados)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Stats sobre train set (usar con CustomCNN from scratch)
TRAIN_MEAN = [0.4752, 0.4592, 0.4563]
TRAIN_STD = [0.2500, 0.2351, 0.2157]

SEED = 42


# =============================================================================
# Device + optimización de recursos
# =============================================================================
def setup_device_and_threads(num_threads: int | None = None) -> torch.device:
    """Selecciona device y configura threading CPU para máximo throughput.

    Args:
        num_threads: número de threads torch (default = todos los cores disponibles).

    Returns:
        torch.device — cuda si hay GPU, cpu si no.
    """
    n = num_threads or os.cpu_count() or 1
    torch.set_num_threads(n)
    if hasattr(torch.backends, "mkldnn"):
        torch.backends.mkldnn.enabled = True  # Intel MKL-DNN aceleración CPU
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


DEVICE = setup_device_and_threads()


# =============================================================================
# ExpConfig — un experimento = un dict de hiperparámetros
# =============================================================================
@dataclass
class ExpConfig:
    """Configuración completa de un experimento de entrenamiento.

    Cambiar cualquier campo aquí = cambiar el experimento. Todos los experimentos
    del pipeline son llamadas a `training.train_one_config(cfg)` con distinto ExpConfig.
    """

    name: str
    model_kind: str = "custom"  # custom | resnet18 | mobilenet_v3_small | efficientnet_b0
    input_size: int = 224
    batch_size: int = 32
    lr_head: float = 1e-3
    lr_backbone: float = 1e-4
    weight_decay: float = 5e-4
    optimizer: str = "adamw"  # adam | adamw | sgd
    scheduler: str = "cosine"  # cosine | step | none
    bbox_loss_weight: float = 5.0
    lambda_giou: float = 2.0
    aug_level: str = "basic"  # none | basic | strong | cabin_realistic
    dropout_head: float = 0.3
    use_class_weights: bool = True
    freeze_epochs: int = 10  # fase A pretrained
    finetune_epochs: int = 20  # fase B pretrained
    total_epochs: int = 30  # CustomCNN (una sola fase)
    early_stopping_patience: int = 12
    seed: int = SEED
    num_workers: int = 0  # 0 evita issues de fork en nbconvert; localmente subir a 4-8
    # Extensiones §16-18
    label_smoothing: float = 0.0
    full_finetune: bool = False  # descongelar TODO el backbone (¡catastrophic forgetting risk!)
    lr_backbone_full: float = 2e-5


@dataclass
class ExperimentResult:
    """Resultado devuelto por train_one_config."""

    name: str
    cfg: dict
    history: list[dict] = field(default_factory=list)
    best: dict = field(default_factory=dict)
    ckpt: str = ""


__all__ = [
    "IS_KAGGLE",
    "DATA_DIR",
    "IMG_DIR",
    "OUT_DIR",
    "CKPT_DIR",
    "IMG_W",
    "IMG_H",
    "CLASSES",
    "N_CLASSES",
    "CLS2IDX",
    "IDX2CLS",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "TRAIN_MEAN",
    "TRAIN_STD",
    "SEED",
    "DEVICE",
    "ExpConfig",
    "ExperimentResult",
    "detect_kaggle",
    "setup_device_and_threads",
]
