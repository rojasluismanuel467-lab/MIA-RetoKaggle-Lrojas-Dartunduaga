"""Estrategias de split train/val — estratificado + group-shuffle temporal + kfold."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedKFold,
    train_test_split,
)

from .config import SEED


def make_split(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split estratificado por clase, single holdout."""
    df_tr, df_val = train_test_split(
        df,
        test_size=test_size,
        stratify=df["class"],
        random_state=seed,
    )
    return df_tr.reset_index(drop=True), df_val.reset_index(drop=True)


def make_kfold(
    df: pd.DataFrame, n_splits: int = 5, seed: int = SEED
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """StratifiedKFold — devuelve lista de (df_train, df_val) por fold."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for tr_idx, val_idx in skf.split(df, df["class"]):
        folds.append(
            (df.iloc[tr_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True))
        )
    return folds


def make_group_split(
    df: pd.DataFrame, n_bins: int = 10, test_size: float = 0.2, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group-shuffle por rangos temporales de frame_idx — más pesimista y honesto."""
    frame_idx = df["filename"].str.extract(r"MP4-(\d+)").astype(int).squeeze()
    frame_group = pd.cut(frame_idx, bins=n_bins, labels=False)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr_idx, val_idx = next(gss.split(df, groups=frame_group))
    return df.iloc[tr_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


__all__ = ["make_split", "make_kfold", "make_group_split"]
