"""Integración opcional con MLflow para registrar experimentos localmente."""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any

from .config import (
    MLFLOW_ENABLED,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_LOG_MODELS,
    MLFLOW_TRACKING_URI,
    PROJECT_ROOT,
)


def _import_mlflow():
    if not MLFLOW_ENABLED:
        return None
    try:
        import mlflow
    except ImportError as exc:  # pragma: no cover - depende del entorno opcional
        warnings.warn(f"MLflow está habilitado pero no está instalado: {exc}")
        return None
    return mlflow


def configure_mlflow() -> Any | None:
    """Configura el tracking local y devuelve el módulo MLflow, si está disponible."""
    mlflow = _import_mlflow()
    if mlflow is None:
        return None

    tracking_uri = MLFLOW_TRACKING_URI
    if tracking_uri.startswith("file:") and not tracking_uri.startswith("file://"):
        tracking_path = (PROJECT_ROOT / tracking_uri.removeprefix("file:")).resolve()
        tracking_uri = tracking_path.as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    return mlflow


def _safe_params(params: dict[str, Any]) -> dict[str, str]:
    return {str(key): str(value)[:500] for key, value in params.items()}


def _finite_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    result = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            result[str(key)] = number
    return result


def log_experiment_result(result: dict[str, Any]) -> bool:
    """Registra parámetros, métricas, historial y checkpoint de un experimento.

    MLflow no debe interrumpir el entrenamiento: cualquier error de tracking se
    reporta como warning y devuelve False.
    """
    mlflow = configure_mlflow()
    if mlflow is None:
        return False

    try:
        name = str(result.get("name", "unnamed"))
        cfg = result.get("cfg", {})
        best = result.get("best", {})
        history = result.get("history", [])
        checkpoint = Path(result["ckpt"]) if result.get("ckpt") else None

        with mlflow.start_run(run_name=name):
            mlflow.log_params(_safe_params(cfg))
            mlflow.set_tags(
                {
                    "project": "aaiv-2026-ii-taller-cnn-miaa-mcd",
                    "device": str(result.get("device", cfg.get("device", "auto"))),
                    "checkpoint_reused": str(result.get("checkpoint_reused", False)),
                }
            )
            mlflow.log_metrics(
                _finite_metrics(
                    {
                        "best_val_loss": best.get("val_loss"),
                        "best_val_acc": best.get("val_acc"),
                        "best_val_dice": best.get("val_dice"),
                        "best_epoch": best.get("epoch"),
                    }
                )
            )

            for step, row in enumerate(history):
                mlflow.log_metrics(
                    _finite_metrics(
                        {
                            key: value
                            for key, value in row.items()
                            if key not in {"epoch", "phase"}
                        }
                    ),
                    step=step,
                )

            if MLFLOW_LOG_MODELS and checkpoint is not None and checkpoint.exists():
                mlflow.log_artifact(str(checkpoint), artifact_path="checkpoints")
        return True
    except Exception as exc:  # pragma: no cover - depende del backend de tracking
        warnings.warn(f"No se pudo registrar el experimento en MLflow: {exc}")
        return False


__all__ = ["configure_mlflow", "log_experiment_result"]
