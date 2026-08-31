"""Compatibilidad y persistencia de checkpoints del entrenamiento."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def checkpoint_state(payload: Any) -> Any:
    """Extrae pesos tanto del formato nuevo como del formato histórico."""
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    return payload


def load_checkpoint(model: nn.Module, path: str | Path, device: torch.device) -> dict:
    """Carga un checkpoint nuevo o un state_dict antiguo en `model`."""
    payload = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint_state(payload))
    return payload if isinstance(payload, dict) else {}


def save_checkpoint(
    model: nn.Module,
    path: str | Path,
    cfg: dict,
    best: dict,
    history: list[dict],
) -> None:
    """Guarda pesos junto con configuración, métricas e historial."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "cfg": cfg,
            "best": best,
            "history": history,
        },
        path,
    )


__all__ = ["checkpoint_state", "load_checkpoint", "save_checkpoint"]
