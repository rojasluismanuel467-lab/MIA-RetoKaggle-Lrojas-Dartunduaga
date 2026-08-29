"""Bucle de entrenamiento reutilizable — `train_one_config(cfg)` es el corazón del pipeline.

Todos los experimentos de los puntos 2-6 de la rúbrica son llamadas a esta función
con distinto ExpConfig. Resultados se acumulan en un tracker externo.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .augmentation import build_transforms
from .config import CKPT_DIR, CLASSES, DEVICE, IMG_DIR, N_CLASSES, ExpConfig
from .dataset import DrowsyDataset
from .losses import EarlyStopping, MultitaskLoss, bbox_dice
from .models import CustomCNN, build_pretrained, unfreeze_last_stage


def compute_class_weights(df_train: pd.DataFrame) -> torch.Tensor:
    """Pesos inversamente proporcionales a frecuencia (Raschka cap 6)."""
    n_total = len(df_train)
    counts = df_train["class"].value_counts().to_dict()
    weights = [n_total / (N_CLASSES * counts[c]) for c in CLASSES]
    return torch.tensor(weights, dtype=torch.float32)


def make_optimizer(
    model: nn.Module, cfg: ExpConfig, param_groups: list | None = None
) -> torch.optim.Optimizer:
    """Factory de optimizer — Adam | AdamW | SGD(momentum)."""
    if param_groups is None:
        param_groups = [{"params": [p for p in model.parameters() if p.requires_grad]}]
    if cfg.optimizer == "adam":
        return torch.optim.Adam(param_groups, lr=cfg.lr_head, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(param_groups, lr=cfg.lr_head, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(
            param_groups, lr=cfg.lr_head, momentum=0.9, weight_decay=cfg.weight_decay
        )
    raise ValueError(f"Optimizer desconocido: {cfg.optimizer!r}")


def make_scheduler(
    optimizer: torch.optim.Optimizer, cfg: ExpConfig, n_epochs: int
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Factory de scheduler — Cosine | Step | None."""
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    if cfg.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    return None


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: MultitaskLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """Un epoch de entrenamiento — devuelve dict con loss/cls/bbox/acc/dice promedio."""
    model.train()
    totals = dict.fromkeys(["loss", "cls", "bbox", "acc", "dice"], 0.0)
    n_samples = 0
    for img_bchw, label_b, bbox_b4 in loader:
        img_bchw = img_bchw.to(device)
        label_b = label_b.to(device)
        bbox_b4 = bbox_b4.to(device)

        logits_bc, bbox_pred_b4 = model(img_bchw)
        losses = loss_fn(logits_bc, bbox_pred_b4, label_b, bbox_b4)

        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()

        batch_size = img_bchw.size(0)
        n_samples += batch_size
        totals["loss"] += losses["total"].item() * batch_size
        totals["cls"] += losses["cls"].item() * batch_size
        totals["bbox"] += losses["bbox"].item() * batch_size
        totals["acc"] += (logits_bc.argmax(1) == label_b).float().sum().item()
        totals["dice"] += bbox_dice(bbox_pred_b4, bbox_b4).sum().item()

    return {k: v / n_samples for k, v in totals.items()}


@torch.no_grad()
def eval_epoch(
    model: nn.Module, loader: DataLoader, loss_fn: MultitaskLoss, device: torch.device
) -> dict[str, float]:
    """Validación en un epoch — mismos returns que train_epoch."""
    model.eval()
    totals = dict.fromkeys(["loss", "cls", "bbox", "acc", "dice"], 0.0)
    n_samples = 0
    for img_bchw, label_b, bbox_b4 in loader:
        img_bchw = img_bchw.to(device)
        label_b = label_b.to(device)
        bbox_b4 = bbox_b4.to(device)

        logits_bc, bbox_pred_b4 = model(img_bchw)
        losses = loss_fn(logits_bc, bbox_pred_b4, label_b, bbox_b4)

        batch_size = img_bchw.size(0)
        n_samples += batch_size
        totals["loss"] += losses["total"].item() * batch_size
        totals["cls"] += losses["cls"].item() * batch_size
        totals["bbox"] += losses["bbox"].item() * batch_size
        totals["acc"] += (logits_bc.argmax(1) == label_b).float().sum().item()
        totals["dice"] += bbox_dice(bbox_pred_b4, bbox_b4).sum().item()

    return {k: v / n_samples for k, v in totals.items()}


def train_one_config(
    cfg: ExpConfig, df_train_split: pd.DataFrame, df_val_split: pd.DataFrame, verbose: bool = True
) -> dict:
    """Entrena un experimento completo según cfg. Corazón del pipeline.

    Args:
        cfg: configuración del experimento.
        df_train_split: split de entrenamiento (filename, class, xmin, ymin, xmax, ymax).
        df_val_split: split de validación.
        verbose: si True, imprime métricas por epoch.

    Returns:
        dict con name, cfg, history (list per epoch), best (dict), ckpt (path).
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    use_imagenet_stats = cfg.model_kind != "custom"
    tr_tf = build_transforms(cfg.input_size, cfg.aug_level, use_imagenet_stats)
    val_tf = build_transforms(cfg.input_size, "none", use_imagenet_stats)

    tr_ds = DrowsyDataset(df_train_split, IMG_DIR, tr_tf)
    val_ds = DrowsyDataset(df_val_split, IMG_DIR, val_tf)
    tr_ld = DataLoader(
        tr_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True
    )
    val_ld = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )

    if cfg.model_kind == "custom":
        model = CustomCNN(dropout=cfg.dropout_head)
    else:
        model = build_pretrained(cfg.model_kind, dropout_head=cfg.dropout_head, freeze=True)
    model = model.to(DEVICE)

    class_weights = (
        compute_class_weights(df_train_split).to(DEVICE) if cfg.use_class_weights else None
    )
    loss_fn = MultitaskLoss(
        w_cls=1.0,
        w_bbox=cfg.bbox_loss_weight,
        lambda_giou=cfg.lambda_giou,
        class_weights=class_weights,
        label_smoothing=cfg.label_smoothing,
    )

    history: list[dict] = []
    best = dict(val_loss=math.inf, val_acc=0.0, val_dice=0.0, epoch=-1)
    ckpt_path = CKPT_DIR / f"{cfg.name}.pt"

    def _run_phase(
        n_epochs: int,
        phase_label: str,
        optimizer: torch.optim.Optimizer,
        scheduler,
        early_stopper: EarlyStopping,
    ) -> None:
        nonlocal best
        for ep in range(n_epochs):
            tr_metrics = train_epoch(model, tr_ld, loss_fn, optimizer, DEVICE)
            val_metrics = eval_epoch(model, val_ld, loss_fn, DEVICE)
            if scheduler is not None:
                scheduler.step()
            row = dict(
                phase=phase_label,
                epoch=ep,
                **{f"train_{k}": v for k, v in tr_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            )
            history.append(row)
            if verbose:
                print(
                    f"  [{phase_label} ep{ep + 1:>2}/{n_epochs}] "
                    f"train loss={tr_metrics['loss']:.3f} acc={tr_metrics['acc']:.3f} "
                    f"dice={tr_metrics['dice']:.3f}  |  "
                    f"val loss={val_metrics['loss']:.3f} acc={val_metrics['acc']:.3f} "
                    f"dice={val_metrics['dice']:.3f}"
                )
            if val_metrics["loss"] < best["val_loss"]:
                best = dict(
                    val_loss=val_metrics["loss"],
                    val_acc=val_metrics["acc"],
                    val_dice=val_metrics["dice"],
                    epoch=len(history) - 1,
                )
                torch.save(model.state_dict(), ckpt_path)
            if early_stopper.step(val_metrics["loss"]):
                if verbose:
                    print(
                        f"  ↳ early stopping @ ep{ep + 1} (patience={cfg.early_stopping_patience})"
                    )
                break

    if cfg.model_kind == "custom":
        opt = make_optimizer(model, cfg)
        sched = make_scheduler(opt, cfg, cfg.total_epochs)
        stopper = EarlyStopping(patience=cfg.early_stopping_patience)
        _run_phase(cfg.total_epochs, "train", opt, sched, stopper)
    elif cfg.full_finetune:
        for p in model.backbone.parameters():
            p.requires_grad = True
        opt = torch.optim.AdamW(
            [
                {"params": model.backbone.parameters(), "lr": cfg.lr_backbone_full},
                {
                    "params": list(model.head_cls.parameters())
                    + list(model.head_bbox.parameters()),
                    "lr": cfg.lr_head,
                },
            ],
            weight_decay=cfg.weight_decay,
        )
        sched = make_scheduler(opt, cfg, cfg.finetune_epochs)
        stopper = EarlyStopping(patience=cfg.early_stopping_patience)
        _run_phase(cfg.finetune_epochs, "full-ft", opt, sched, stopper)
    else:
        opt_a = make_optimizer(model, cfg)
        sched_a = make_scheduler(opt_a, cfg, cfg.freeze_epochs)
        _run_phase(
            cfg.freeze_epochs,
            "A(feat)",
            opt_a,
            sched_a,
            EarlyStopping(patience=cfg.early_stopping_patience),
        )
        unfreeze_last_stage(model, cfg.model_kind)
        opt_b = torch.optim.AdamW(
            [
                {
                    "params": [p for p in model.backbone.parameters() if p.requires_grad],
                    "lr": cfg.lr_backbone,
                },
                {
                    "params": list(model.head_cls.parameters())
                    + list(model.head_bbox.parameters()),
                    "lr": cfg.lr_head,
                },
            ],
            weight_decay=cfg.weight_decay,
        )
        sched_b = make_scheduler(opt_b, cfg, cfg.finetune_epochs)
        _run_phase(
            cfg.finetune_epochs,
            "B(fine)",
            opt_b,
            sched_b,
            EarlyStopping(patience=cfg.early_stopping_patience),
        )

    return dict(name=cfg.name, cfg=asdict(cfg), history=history, best=best, ckpt=str(ckpt_path))


__all__ = [
    "train_one_config",
    "train_epoch",
    "eval_epoch",
    "compute_class_weights",
    "make_optimizer",
    "make_scheduler",
]
