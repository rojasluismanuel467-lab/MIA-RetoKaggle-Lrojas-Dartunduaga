#!/usr/bin/env python3
"""Ejemplo mínimo de uso del paquete drowsy_cnn — para el compañero.

Muestra el patrón: config → split → dataset → model → training → inference.
Todo importa del paquete, sin duplicación.

Ejecutar: `python tools/example_usage.py`
"""
from __future__ import annotations

import pandas as pd

from drowsy_cnn.augmentation import build_transforms
from drowsy_cnn.config import DATA_DIR, ExpConfig
from drowsy_cnn.dataset import DrowsyDataset
from drowsy_cnn.models import build_pretrained
from drowsy_cnn.splits import make_split
from drowsy_cnn.training import train_one_config


def main() -> None:
    df_full = pd.read_csv(DATA_DIR / "train.csv")
    df_tr, df_val = make_split(df_full, test_size=0.2)
    print(f"Train: {len(df_tr)}, Val: {len(df_val)}")

    cfg = ExpConfig(
        name="ejemplo_efficientnet_cabin",
        model_kind="efficientnet_b0",
        aug_level="cabin_realistic",   # incluye robustez a iluminación + oclusiones
        batch_size=32,
        freeze_epochs=5,
        finetune_epochs=15,
        label_smoothing=0.1,
        num_workers=4,                  # aprovecha CPUs locales
    )
    result = train_one_config(cfg, df_tr, df_val, verbose=True)
    print(f"\nMejor val_acc: {result['best']['val_acc']:.4f}")
    print(f"Mejor val_dice: {result['best']['val_dice']:.4f}")
    print(f"Checkpoint: {result['ckpt']}")


if __name__ == "__main__":
    main()
