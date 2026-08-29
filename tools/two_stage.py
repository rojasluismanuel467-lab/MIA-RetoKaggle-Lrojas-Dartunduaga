#!/usr/bin/env python3
"""
Two-stage classifier:
  Stage 1 (bbox regression): usar el mega ensemble ya entrenado.
  Stage 2 (classification):  entrenar un NUEVO EfficientNet-B0 que ve SOLO el crop del bbox.

Idea: en la imagen 224×224, el driver ocupa ~10% del área. El clasificador batalla porque ve
mucho fondo ruidoso. Recortar y clasificar solo la región del driver debería mejorar acc
significativamente (Girshick 2014 R-CNN, base de todas las arquitecturas de detección).
"""
import os, sys, math, warnings, time
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'
IMG_DIR = DATA_DIR / 'images'
CKPT_DIR = ROOT / 'checkpoints'
OUT_DIR = ROOT / 'outputs'
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
IMG_W, IMG_H = 1920, 1080
CLASSES = ['awake', 'drowsy']
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}
IMAGENET_MEAN = [0.485, 0.456, 0.406]; IMAGENET_STD = [0.229, 0.224, 0.225]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")


# ==================== HELPERS COMPARTIDOS ====================
class MultitaskWrapper(nn.Module):
    def __init__(self, backbone, feat_dim, n_classes=2, dropout_head=0.3):
        super().__init__()
        self.backbone = backbone; self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_head)
        self.head_cls = nn.Linear(feat_dim, n_classes)
        self.head_bbox = nn.Sequential(nn.Linear(feat_dim, 4), nn.Sigmoid())
    def forward(self, x):
        f = self.backbone(x)
        if f.ndim == 4: f = self.pool(f).flatten(1)
        f = self.dropout(f)
        return self.head_cls(f), self.head_bbox(f)


def build_pretrained(name, dropout_head=0.3):
    if name == 'resnet18':
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        fd = m.fc.in_features; m.fc = nn.Identity()
    elif name == 'mobilenet_v3_small':
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        fd = m.classifier[0].in_features; m.classifier = nn.Identity()
    elif name == 'efficientnet_b0':
        m = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0, global_pool='')
        fd = m.num_features
    return MultitaskWrapper(m, feat_dim=fd, dropout_head=dropout_head)


def cxcywh_norm_to_xyxy(bb, w, h):
    cx, cy, bw, bh = bb.unbind(-1)
    return torch.stack([(cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h], dim=-1)


# ==================== STAGE 2: CROP CLASSIFIER ====================
class CropClassifier(nn.Module):
    """EfficientNet-B0 puro para clasificación (sin head bbox)."""
    def __init__(self, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True,
                                          num_classes=0, global_pool='')
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.backbone.num_features, 2)
    def forward(self, x):
        f = self.pool(self.backbone(x)).flatten(1)
        return self.head(self.dropout(f))


class CropDataset(Dataset):
    """
    Dataset que recorta la imagen al bbox (con margen) antes de pasarla al transform.
    - Si tiene labels (train/val): usa bbox GT + margen 15%.
    - Si es test: requiere que df tenga columnas xmin/ymin/xmax/ymax (predichas por stage 1).
    """
    def __init__(self, df, img_dir, transform, margin: float = 0.15, is_test: bool = False):
        self.df = df.reset_index(drop=True); self.img_dir = Path(img_dir)
        self.tf = transform; self.margin = margin; self.is_test = is_test

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.cvtColor(cv2.imread(str(self.img_dir / row['filename'])), cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        # Aplicar margen 15% al bbox
        bw = row.xmax - row.xmin; bh = row.ymax - row.ymin
        mx = bw * self.margin; my = bh * self.margin
        x1 = int(max(0, row.xmin - mx)); y1 = int(max(0, row.ymin - my))
        x2 = int(min(W, row.xmax + mx)); y2 = int(min(H, row.ymax + my))
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:  # bbox degenerado
            crop = img
        out = self.tf(image=crop)
        img_t = out['image']
        if self.is_test:
            return img_t, row['filename']
        label = torch.tensor(CLS2IDX[row['class']], dtype=torch.long)
        return img_t, label


def build_transform_crop(size=224, aug='basic'):
    ops = []
    if aug in ('basic', 'strong'):
        ops.append(A.HorizontalFlip(p=0.5))
        ops.append(A.RandomBrightnessContrast(0.2, 0.2, p=0.5))
    if aug == 'strong':
        ops.append(A.ShiftScaleRotate(0.05, 0.1, 10, p=0.5))
        ops.append(A.HueSaturationValue(10, 15, 10, p=0.5))
        ops.append(A.GaussNoise(var_limit=(10, 30), p=0.3))
        ops.append(A.MotionBlur(blur_limit=5, p=0.3))
    ops += [A.Resize(size, size), A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return A.Compose(ops)


# ==================== STAGE 1: mega ensemble para predecir bboxes de test ====================
CKPT_LIST = [
    ('p5_resnet18', 'resnet18'),
    ('p5_mobilenet_v3_small', 'mobilenet_v3_small'),
    ('p5_efficientnet_b0', 'efficientnet_b0'),
    ('p17_effnet_snapshot_snap0', 'efficientnet_b0'),
    ('p17_effnet_snapshot_snap1', 'efficientnet_b0'),
    ('p17_effnet_snapshot_snap2', 'efficientnet_b0'),
]

# Necesitamos el DrowsyDataset original para stage 1
class DrowsyDataset(Dataset):
    def __init__(self, df, img_dir, transform, is_test=False):
        self.df = df.reset_index(drop=True); self.img_dir = Path(img_dir)
        self.tf = transform; self.is_test = is_test
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.cvtColor(cv2.imread(str(self.img_dir / row['filename'])), cv2.COLOR_BGR2RGB)
        if self.is_test:
            bboxes = [[IMG_W*0.4, IMG_H*0.4, IMG_W*0.6, IMG_H*0.6]]; labs = ['awake']
        else:
            bboxes = [[row.xmin, row.ymin, row.xmax, row.ymax]]; labs = [row['class']]
        out = self.tf(image=img, bboxes=bboxes, class_labels=labs)
        img_t = out['image']
        if self.is_test: return img_t, row['filename']
        H, W = img_t.shape[-2:]
        cx = ((out['bboxes'][0][0] + out['bboxes'][0][2])/2)/W
        cy = ((out['bboxes'][0][1] + out['bboxes'][0][3])/2)/H
        bw = (out['bboxes'][0][2] - out['bboxes'][0][0])/W
        bh = (out['bboxes'][0][3] - out['bboxes'][0][1])/H
        return img_t, torch.tensor(CLS2IDX[out['class_labels'][0]]), torch.tensor([cx,cy,bw,bh], dtype=torch.float32)


def build_transforms_bbox(size=224):
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.3))


def load_stage1_models():
    ms = []
    for name, kind in CKPT_LIST:
        p = CKPT_DIR / f'{name}.pt'
        if not p.exists(): continue
        m = build_pretrained(kind).to(DEVICE)
        m.load_state_dict(torch.load(p, map_location=DEVICE))
        ms.append(m)
    return ms


@torch.no_grad()
def stage1_predict_bboxes(models_list, df, is_test=False):
    """Corre el mega ensemble para producir bboxes (xmin,ymin,xmax,ymax en pixels)."""
    accum_bb = 0.0
    for m in models_list:
        m.eval()
        tf = build_transforms_bbox(224)
        ds = DrowsyDataset(df, IMG_DIR, tf, is_test=is_test)
        ld = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
        bb_l = []
        for batch in ld:
            img = batch[0].to(DEVICE)
            _, b1 = m(img)
            _, b2 = m(torch.flip(img, dims=[-1]))
            b2d = b2.clone(); b2d[:, 0] = 1.0 - b2[:, 0]
            bb_l.append(((b1 + b2d)/2).cpu())
        accum_bb = accum_bb + torch.cat(bb_l).numpy()
    bb = accum_bb / len(models_list)
    xyxy = cxcywh_norm_to_xyxy(torch.tensor(bb), IMG_W, IMG_H)
    xyxy[:, 0::2].clamp_(0, IMG_W); xyxy[:, 1::2].clamp_(0, IMG_H)
    return xyxy.numpy()


# ==================== EJECUCIÓN ====================
def train_crop_classifier(df_tr_c, df_val_c, epochs=25, batch=32, lr=1e-3, aug='strong'):
    tr_tf = build_transform_crop(224, aug)
    val_tf = build_transform_crop(224, 'none')
    tr_ds = CropDataset(df_tr_c, IMG_DIR, tr_tf, is_test=False)
    val_ds = CropDataset(df_val_c, IMG_DIR, val_tf, is_test=False)
    tr_ld = DataLoader(tr_ds, batch_size=batch, shuffle=True, num_workers=0, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=0)

    model = CropClassifier(dropout=0.3).to(DEVICE)
    # Class weights
    n = len(df_tr_c); counts = df_tr_c['class'].value_counts().to_dict()
    cw = torch.tensor([n/(2*counts[c]) for c in CLASSES], dtype=torch.float32).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_acc = 0.0; best_state = None
    for ep in range(epochs):
        model.train()
        tr_correct = tr_total = 0; tr_loss = 0.0
        for img, lab in tr_ld:
            img, lab = img.to(DEVICE), lab.to(DEVICE)
            logits = model(img); loss = loss_fn(logits, lab)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_correct += (logits.argmax(1) == lab).sum().item()
            tr_total += lab.size(0); tr_loss += loss.item() * lab.size(0)
        sched.step()
        model.eval()
        val_correct = val_total = 0; val_loss = 0.0
        with torch.no_grad():
            for img, lab in val_ld:
                img, lab = img.to(DEVICE), lab.to(DEVICE)
                logits = model(img); loss = loss_fn(logits, lab)
                val_correct += (logits.argmax(1) == lab).sum().item()
                val_total += lab.size(0); val_loss += loss.item() * lab.size(0)
        tr_acc = tr_correct/tr_total; val_acc = val_correct/val_total
        print(f"  ep{ep+1:>2}/{epochs}  train loss={tr_loss/tr_total:.3f} acc={tr_acc:.3f}  |  val loss={val_loss/val_total:.3f} acc={val_acc:.3f}")
        if val_acc > best_acc:
            best_acc = val_acc; best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_acc


@torch.no_grad()
def crop_predict_probs(model, df, is_test=False):
    model.eval()
    tf = build_transform_crop(224, 'none')
    ds = CropDataset(df, IMG_DIR, tf, is_test=is_test)
    ld = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
    probs_all, labels = [], []
    for batch in ld:
        img = batch[0].to(DEVICE)
        # TTA horizontal flip
        l1 = model(img); l2 = model(torch.flip(img, dims=[-1]))
        p = F.softmax((l1+l2)/2, dim=1).cpu()
        probs_all.append(p)
        if not is_test: labels.append(batch[1])
    probs = torch.cat(probs_all).numpy()
    return probs if is_test else (probs, torch.cat(labels).numpy())


# ============== MAIN ==============
df_full = pd.read_csv(DATA_DIR / 'train.csv')
df_test = pd.read_csv(DATA_DIR / 'test.csv')
df_tr, df_val = train_test_split(df_full, test_size=0.2, stratify=df_full['class'], random_state=SEED)
df_tr = df_tr.reset_index(drop=True); df_val = df_val.reset_index(drop=True)
print(f"Train: {len(df_tr)}, Val: {len(df_val)}, Test: {len(df_test)}")

# STAGE 1: predecir bboxes de val y test con mega ensemble (para stage 2 en inferencia)
print("\n[STAGE 1] Cargando mega ensemble para predecir bboxes...")
stage1 = load_stage1_models()
print(f"  {len(stage1)} modelos cargados")

# Para el train de stage 2: usamos GT bboxes (no queremos propagar error del stage 1 al training).
# Para val y test: usamos bboxes predichos por stage 1.
print("[STAGE 1] Prediciendo bboxes de val (para stage 2 eval)...")
val_bb = stage1_predict_bboxes(stage1, df_val, is_test=False)
df_val_predicted = df_val.copy()
df_val_predicted[['xmin','ymin','xmax','ymax']] = val_bb.round().astype(int)

print("[STAGE 1] Prediciendo bboxes de test (para stage 2 inference)...")
test_bb = stage1_predict_bboxes(stage1, df_test, is_test=True)
df_test_predicted = df_test.copy()
df_test_predicted['class'] = 'awake'  # placeholder
df_test_predicted[['xmin','ymin','xmax','ymax']] = test_bb.round().astype(int)

# STAGE 2: entrenar clasificador sobre CROPS
print("\n[STAGE 2] Entrenando CropClassifier sobre bbox GT (train) y predichos (val)...")
t0 = time.time()
crop_model, best_val_acc = train_crop_classifier(df_tr, df_val_predicted, epochs=25, lr=1e-3, aug='strong')
print(f"\n[STAGE 2] Entrenado en {(time.time()-t0)/60:.1f} min. Mejor val_acc = {best_val_acc:.4f}")
torch.save(crop_model.state_dict(), CKPT_DIR / 'p18_crop_classifier.pt')

# EVAL FINAL: comparar stage 2 solo vs ensemble con mega v2 stage 1
print("\n[EVAL] Predicciones de stage 2 sobre val (crop con bbox predicho)...")
crop_val_probs, val_labels = crop_predict_probs(crop_model, df_val_predicted, is_test=False)
crop_val_acc = (crop_val_probs.argmax(1) == val_labels).mean()
print(f"  Stage 2 solo (crop TTA hflip) val_acc = {crop_val_acc:.4f}")

# Ensemble stage 2 con las probs del mega ensemble (stage 1)
print("\n[EVAL] Ensemble two-stage (stage2 + mega v2 class probs)...")
# Para las probs del mega, corremos su predicción sobre df_val
@torch.no_grad()
def mega_class_probs(models_list, df, is_test=False):
    accum = 0.0
    for m in models_list:
        m.eval()
        tf = build_transforms_bbox(224)
        ds = DrowsyDataset(df, IMG_DIR, tf, is_test=is_test)
        ld = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
        pl = []
        for batch in ld:
            img = batch[0].to(DEVICE)
            l1, _ = m(img); l2, _ = m(torch.flip(img, dims=[-1]))
            pl.append(F.softmax((l1+l2)/2, dim=1).cpu())
        accum = accum + torch.cat(pl).numpy()
    return accum / len(models_list)

mega_val_probs = mega_class_probs(stage1, df_val, is_test=False)
mega_val_acc = (mega_val_probs.argmax(1) == val_labels).mean()
print(f"  Stage 1 solo (mega ensemble class probs) val_acc = {mega_val_acc:.4f}")

# Ensemble two-stage: mezcla ponderada
for w in [0.3, 0.5, 0.7]:
    combined = w * crop_val_probs + (1-w) * mega_val_probs
    acc = (combined.argmax(1) == val_labels).mean()
    print(f"  Two-stage (w_crop={w:.1f}) val_acc = {acc:.4f}")

# BEST combo: pick w=0.5 for submission (or dynamic if evaluated)
best_w = 0.5
final_val_probs = best_w * crop_val_probs + (1-best_w) * mega_val_probs
final_val_acc = (final_val_probs.argmax(1) == val_labels).mean()

# SUBMISSION final
print("\n[SUBMISSION] Generando submission_two_stage.csv...")
crop_test_probs = crop_predict_probs(crop_model, df_test_predicted, is_test=True)
mega_test_probs = mega_class_probs(stage1, df_test, is_test=True)
final_test_probs = best_w * crop_test_probs + (1-best_w) * mega_test_probs
final_preds = final_test_probs.argmax(1)

sub = pd.DataFrame({
    'filename': df_test_predicted['filename'].tolist(),
    'class': [IDX2CLS[p] for p in final_preds],
    'xmin': df_test_predicted['xmin'].tolist(),
    'ymin': df_test_predicted['ymin'].tolist(),
    'xmax': df_test_predicted['xmax'].tolist(),
    'ymax': df_test_predicted['ymax'].tolist(),
})
sub.to_csv(OUT_DIR / 'submission_two_stage.csv', index=False)
print(f"✓ submission_two_stage.csv ({len(sub)} filas)")
print(f"Distribución: {sub['class'].value_counts().to_dict()}")

# BOTTOM LINE
print("\n" + "="*60)
print(f"BOTTOM LINE — mejor val_acc alcanzado en este script:")
print(f"  Stage 2 solo               = {crop_val_acc:.4f}")
print(f"  Stage 1 solo (mega class)  = {mega_val_acc:.4f}")
print(f"  Two-stage (w=0.5)          = {final_val_acc:.4f}")
print("="*60)
