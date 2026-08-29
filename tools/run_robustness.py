#!/usr/bin/env python3
"""Corre la matriz de robustez sobre el mejor modelo (p18_crop_classifier + mega stage1).

Reutiliza checkpoints del disco — no re-entrena nada. ~3 min end-to-end.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from drowsy_cnn.config import CKPT_DIR, DATA_DIR, DEVICE, IMG_DIR, IMG_H, IMG_W, SEED
from drowsy_cnn.dataset import DrowsyDataset, cxcywh_norm_to_xyxy
from drowsy_cnn.inference import load_stage1_models, stage1_predict_bboxes
from drowsy_cnn.models import CropClassifier
from drowsy_cnn.robustness import CONDITION_TRANSFORMS
from drowsy_cnn.augmentation import build_transforms_crop
from drowsy_cnn.splits import make_split

import albumentations as A
import cv2
from torch.utils.data import Dataset
from drowsy_cnn.config import CLS2IDX, IDX2CLS


class RobustnessDataset(Dataset):
    """Aplica condición sintética a un crop del bbox — mide degradación por condición."""

    def __init__(self, df: pd.DataFrame, img_dir: Path,
                 condition: A.Compose, final: A.Compose, margin: float = 0.15):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.condition = condition
        self.final = final
        self.margin = margin

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = cv2.cvtColor(cv2.imread(str(self.img_dir / row["filename"])), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        bw = row.xmax - row.xmin
        bh = row.ymax - row.ymin
        mx, my = bw * self.margin, bh * self.margin
        x1, y1 = int(max(0, row.xmin - mx)), int(max(0, row.ymin - my))
        x2, y2 = int(min(w, row.xmax + mx)), int(min(h, row.ymax + my))
        crop = img[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else img
        cond_out = self.condition(image=crop)
        out = self.final(image=cond_out["image"])
        return out["image"], torch.tensor(CLS2IDX[row["class"]], dtype=torch.long)


@torch.no_grad()
def evaluate_condition(model: torch.nn.Module, df: pd.DataFrame,
                       condition: A.Compose) -> float:
    """Retorna accuracy sobre df bajo la condición dada."""
    model.eval()
    final_tf = build_transforms_crop(224, "none")
    ds = RobustnessDataset(df, IMG_DIR, condition, final_tf)
    ld = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    correct = total = 0
    for img_bchw, label_b in ld:
        img_bchw = img_bchw.to(DEVICE)
        label_b = label_b.to(DEVICE)
        logits_1 = model(img_bchw)
        logits_2 = model(torch.flip(img_bchw, dims=[-1]))
        probs = F.softmax((logits_1 + logits_2) / 2, dim=1)
        correct += (probs.argmax(1) == label_b).sum().item()
        total += label_b.size(0)
    return correct / total


def main() -> None:
    # 1. Cargar splits (mismo seed = mismo val que en §18)
    df_full = pd.read_csv(DATA_DIR / "train.csv")
    _, df_val = make_split(df_full, test_size=0.2, seed=SEED)
    print(f"Val samples: {len(df_val)}")

    # 2. Cargar stage 1 (mega ensemble) para predecir bboxes val
    print("\nCargando stage 1 (mega ensemble)...")
    stage1_models = load_stage1_models([
        ("p5_resnet18", "resnet18"),
        ("p5_mobilenet_v3_small", "mobilenet_v3_small"),
        ("p5_efficientnet_b0", "efficientnet_b0"),
        ("p17_effnet_snapshot_snap0", "efficientnet_b0"),
        ("p17_effnet_snapshot_snap1", "efficientnet_b0"),
        ("p17_effnet_snapshot_snap2", "efficientnet_b0"),
    ])
    print(f"  {len(stage1_models)} modelos stage 1 cargados")

    print("Prediciendo bboxes val...")
    val_bboxes_xyxy = stage1_predict_bboxes(stage1_models, df_val, is_test=False)
    df_val_predicted = df_val.copy()
    df_val_predicted[["xmin", "ymin", "xmax", "ymax"]] = val_bboxes_xyxy.round().astype(int)

    # 3. Cargar crop classifier (§18)
    print("\nCargando crop classifier del §18...")
    crop_model = CropClassifier(dropout=0.3).to(DEVICE)
    crop_model.load_state_dict(torch.load(CKPT_DIR / "p18_crop_classifier.pt",
                                          map_location=DEVICE, weights_only=True))

    # 4. Evaluar cada condición
    print(f"\nEvaluando {len(CONDITION_TRANSFORMS)} condiciones sintéticas:")
    rows = []
    for cond_name, cond_transform in CONDITION_TRANSFORMS.items():
        acc = evaluate_condition(crop_model, df_val_predicted, cond_transform)
        rows.append({"condition": cond_name, "val_acc": acc, "n": len(df_val_predicted)})
        print(f"  {cond_name:28s} val_acc = {acc:.4f}")

    df_robust = pd.DataFrame(rows).set_index("condition").sort_values("val_acc", ascending=False)
    print(f"\n=== MATRIZ DE ROBUSTEZ ordenada ===")
    print(df_robust.round(4).to_string())

    from drowsy_cnn.config import OUT_DIR
    out_path = OUT_DIR / "robustness_matrix.csv"
    df_robust.to_csv(out_path)
    print(f"\n✓ Escrito a {out_path}")


if __name__ == "__main__":
    main()
