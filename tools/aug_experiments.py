#!/usr/bin/env python3
"""
Experimentos de data augmentation avanzada sobre stage 2 (CropClassifier del §18).

Basado en investigación agente + fuentes:
  - DrowsyDetectNet 2024 (CLAHE + brightness estándar)
  - Albumentations docs oficiales
  - RandAugment (Cubuk NeurIPS 2020) — +3% en small datasets
  - CutMix ICCV 2019
  - FaceMixup arxiv:2405.20259

Estrategias:
  A_perceptual  = strong + CoarseDropout + ImageCompression + Downscale + CLAHE
  B_lighting    = A + RandomShadow + RandomSunFlare + RandomGamma + ToGray
  C_aggressive  = B + RandAugment (o MixUp — probamos ambos)

Baseline actual (val_acc=0.988) es 'strong' del §18.

Reusa el stage 1 (mega ensemble) para predecir bboxes val/test.
"""
import os, sys, math, warnings, time, json
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


# ==== Reuso de las clases del two_stage.py ====
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


class CropClassifier(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True,
                                          num_classes=0, global_pool='')
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.backbone.num_features, 2)
    def forward(self, x):
        return self.head(self.dropout(self.pool(self.backbone(x)).flatten(1)))


class CropDataset(Dataset):
    def __init__(self, df, img_dir, transform, margin=0.15, is_test=False):
        self.df = df.reset_index(drop=True); self.img_dir = Path(img_dir)
        self.tf = transform; self.margin = margin; self.is_test = is_test
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.cvtColor(cv2.imread(str(self.img_dir / row['filename'])), cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        bw = row.xmax - row.xmin; bh = row.ymax - row.ymin
        mx, my = bw * self.margin, bh * self.margin
        x1 = int(max(0, row.xmin - mx)); y1 = int(max(0, row.ymin - my))
        x2 = int(min(W, row.xmax + mx)); y2 = int(min(H, row.ymax + my))
        crop = img[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else img
        out = self.tf(image=crop)
        if self.is_test: return out['image'], row['filename']
        return out['image'], torch.tensor(CLS2IDX[row['class']], dtype=torch.long)


# ==================== ESTRATEGIAS DE AUGMENTATION ====================
def build_transform(strategy: str, size: int = 224) -> A.Compose:
    """
    strategy:
      'none'          — baseline (val/test)
      'strong'        — BASELINE ACTUAL del §18 (val_acc=0.988)
      'A_perceptual'  — strong + CoarseDropout + ImageCompression + Downscale + CLAHE
      'B_lighting'    — A + RandomShadow + RandomSunFlare + RandomGamma + ToGray
      'C_randaug'     — B + RandAugment-style (varias aleatorias)
    """
    ops = []

    if strategy != 'none':
        # Baseline strong (compartido por todas menos 'none')
        ops += [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
            A.GaussNoise(var_limit=(10, 30), p=0.3),
            A.MotionBlur(blur_limit=5, p=0.3),
        ]

    if strategy in ('A_perceptual', 'B_lighting', 'C_randaug'):
        # A: perceptual/quality robustness
        ops += [
            A.CoarseDropout(max_holes=8, max_height=24, max_width=24,
                            min_holes=1, min_height=8, min_width=8, p=0.4),
            A.ImageCompression(quality_lower=75, quality_upper=95, p=0.3),
            A.Downscale(scale_min=0.75, scale_max=0.95, p=0.2),
            A.CLAHE(clip_limit=3.0, p=0.2),
        ]

    if strategy in ('B_lighting', 'C_randaug'):
        # B: cabin lighting
        ops += [
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, p=0.3),
            A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), num_flare_circles_lower=1, num_flare_circles_upper=3, p=0.1),
            A.RandomGamma(gamma_limit=(80, 120), p=0.2),
            A.ToGray(p=0.05),
        ]

    if strategy == 'C_randaug':
        # C: agregar 1 transform aleatorio extra por imagen (RandAugment-lite)
        ops += [
            A.OneOf([
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0)),
                A.Emboss(alpha=(0.2, 0.5)),
                A.Posterize(num_bits=4),
                A.Solarize(threshold=128),
                A.Equalize(),
            ], p=0.3),
        ]

    # Resize + normalize + tensor (siempre al final)
    ops += [
        A.Resize(size, size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]
    return A.Compose(ops)


# ==================== STAGE 1: bbox prediction (reusar) ====================
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
        return img_t, torch.tensor(CLS2IDX[out['class_labels'][0]]), \
               torch.tensor([((out['bboxes'][0][0]+out['bboxes'][0][2])/2)/W,
                             ((out['bboxes'][0][1]+out['bboxes'][0][3])/2)/H,
                             (out['bboxes'][0][2]-out['bboxes'][0][0])/W,
                             (out['bboxes'][0][3]-out['bboxes'][0][1])/H], dtype=torch.float32)


def transforms_bbox(size=224):
    return A.Compose([A.Resize(size, size), A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()],
                     bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.3))


@torch.no_grad()
def stage1_predict_bboxes(model_list, df, is_test=False):
    accum = 0.0
    for m in model_list:
        m.eval()
        ds = DrowsyDataset(df, IMG_DIR, transforms_bbox(224), is_test=is_test)
        ld = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
        bl = []
        for batch in ld:
            img = batch[0].to(DEVICE)
            _, b1 = m(img); _, b2 = m(torch.flip(img, dims=[-1]))
            b2d = b2.clone(); b2d[:, 0] = 1.0 - b2[:, 0]
            bl.append(((b1+b2d)/2).cpu())
        accum = accum + torch.cat(bl).numpy()
    xyxy = cxcywh_norm_to_xyxy(torch.tensor(accum / len(model_list)), IMG_W, IMG_H)
    xyxy[:, 0::2].clamp_(0, IMG_W); xyxy[:, 1::2].clamp_(0, IMG_H)
    return xyxy.numpy()


# ==================== TRAINING ====================
def train_crop(df_tr, df_val, strategy_name, epochs=25, batch=32, lr=1e-3):
    """Entrena CropClassifier con la estrategia de aug indicada. Devuelve mejor val_acc."""
    tr_tf = build_transform(strategy_name)
    val_tf = build_transform('none')
    tr_ld = DataLoader(CropDataset(df_tr, IMG_DIR, tr_tf), batch_size=batch, shuffle=True, num_workers=0, drop_last=True)
    val_ld = DataLoader(CropDataset(df_val, IMG_DIR, val_tf), batch_size=batch, shuffle=False, num_workers=0)

    model = CropClassifier(dropout=0.3).to(DEVICE)
    n = len(df_tr); counts = df_tr['class'].value_counts().to_dict()
    cw = torch.tensor([n/(2*counts[c]) for c in CLASSES], dtype=torch.float32).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_acc, best_state, best_ep = 0.0, None, 0
    history = []
    for ep in range(epochs):
        model.train()
        for img, lab in tr_ld:
            img, lab = img.to(DEVICE), lab.to(DEVICE)
            loss = loss_fn(model(img), lab)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval(); correct = total = 0
        with torch.no_grad():
            for img, lab in val_ld:
                img, lab = img.to(DEVICE), lab.to(DEVICE)
                correct += (model(img).argmax(1) == lab).sum().item(); total += lab.size(0)
        vacc = correct/total
        history.append(vacc)
        if vacc > best_acc:
            best_acc, best_ep = vacc, ep+1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    return best_acc, best_ep, history, best_state


# ==================== MAIN ====================
df_full = pd.read_csv(DATA_DIR / 'train.csv')
df_test = pd.read_csv(DATA_DIR / 'test.csv')
df_tr, df_val = train_test_split(df_full, test_size=0.2, stratify=df_full['class'], random_state=SEED)
df_tr = df_tr.reset_index(drop=True); df_val = df_val.reset_index(drop=True)
print(f"Device: {DEVICE}  |  train {len(df_tr)}  val {len(df_val)}")

# Cargar stage 1 (mega) para producir bboxes de val (mismos que en §18)
STAGE1_LIST = [
    ('p5_resnet18', 'resnet18'), ('p5_mobilenet_v3_small', 'mobilenet_v3_small'),
    ('p5_efficientnet_b0', 'efficientnet_b0'),
    ('p17_effnet_snapshot_snap0', 'efficientnet_b0'),
    ('p17_effnet_snapshot_snap1', 'efficientnet_b0'),
    ('p17_effnet_snapshot_snap2', 'efficientnet_b0'),
]
print("\nCargando stage 1 para predecir bboxes val...")
stage1 = []
for name, kind in STAGE1_LIST:
    p = CKPT_DIR / f'{name}.pt'
    if not p.exists(): continue
    m = build_pretrained(kind).to(DEVICE)
    m.load_state_dict(torch.load(p, map_location=DEVICE))
    stage1.append(m)
val_bb = stage1_predict_bboxes(stage1, df_val, is_test=False)
df_val_pred = df_val.copy(); df_val_pred[['xmin','ymin','xmax','ymax']] = val_bb.round().astype(int)

# ---- CORRIDA DE 4 EXPERIMENTOS: strong (baseline §18), A, B, C ----
STRATEGIES = ['strong', 'A_perceptual', 'B_lighting', 'C_randaug']
results = {}
for strat in STRATEGIES:
    print(f"\n{'='*60}\nEntrenando CropClassifier con aug='{strat}'\n{'='*60}")
    t0 = time.time()
    best_acc, best_ep, hist, state = train_crop(df_tr, df_val_pred, strat, epochs=25)
    dt = time.time() - t0
    print(f"  best_val_acc = {best_acc:.4f}  (ep {best_ep})  |  duración {dt/60:.1f} min")
    results[strat] = {'best_acc': best_acc, 'best_ep': best_ep, 'history': hist}
    # Guardar checkpoint del mejor
    ckpt_path = CKPT_DIR / f'p19_crop_{strat}.pt'
    torch.save(state, ckpt_path)

# Comparativa final
print(f"\n{'='*60}\nCOMPARATIVA AUGMENTATION EXPERIMENTS\n{'='*60}")
print(f"{'estrategia':16s}  {'best_val_acc':>12s}  {'best_ep':>8s}")
for strat, r in sorted(results.items(), key=lambda x: -x[1]['best_acc']):
    print(f"  {strat:16s}  {r['best_acc']:>12.4f}  {r['best_ep']:>8d}")

# Guardar resultados
with open(OUT_DIR / 'aug_experiments.json', 'w') as f:
    json.dump({k: {kk: (vv if not isinstance(vv, list) else vv) for kk, vv in v.items()} for k, v in results.items()}, f, indent=2)
print(f"\n✓ Resultados guardados en outputs/aug_experiments.json")

# Si algún experimento supera el baseline (0.9881), generar submission
best_strat = max(results.items(), key=lambda x: x[1]['best_acc'])
best_val = best_strat[1]['best_acc']
if best_val >= 0.9881:
    print(f"\n✓ Ganador: {best_strat[0]} con val_acc={best_val:.4f}")
    # Cargar el mejor y generar submission
    ckpt = CKPT_DIR / f'p19_crop_{best_strat[0]}.pt'
    model_best = CropClassifier(dropout=0.3).to(DEVICE)
    model_best.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    # Predecir bboxes test
    test_bb = stage1_predict_bboxes(stage1, df_test, is_test=True)
    df_test_pred = df_test.copy(); df_test_pred['class'] = 'awake'
    df_test_pred[['xmin','ymin','xmax','ymax']] = test_bb.round().astype(int)
    # Predecir clases con TTA hflip
    model_best.eval()
    tf = build_transform('none')
    ds = CropDataset(df_test_pred, IMG_DIR, tf, is_test=True)
    ld = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
    probs_all, fnames = [], []
    with torch.no_grad():
        for img, fn in ld:
            img = img.to(DEVICE)
            l1 = model_best(img); l2 = model_best(torch.flip(img, dims=[-1]))
            probs_all.append(F.softmax((l1+l2)/2, dim=1).cpu())
            fnames.extend(fn)
    probs = torch.cat(probs_all).numpy()
    sub = pd.DataFrame({
        'filename': fnames,
        'class': [IDX2CLS[p] for p in probs.argmax(1)],
        'xmin': df_test_pred['xmin'].tolist(),
        'ymin': df_test_pred['ymin'].tolist(),
        'xmax': df_test_pred['xmax'].tolist(),
        'ymax': df_test_pred['ymax'].tolist(),
    })
    sub_path = OUT_DIR / f'submission_two_stage_{best_strat[0]}.csv'
    sub.to_csv(sub_path, index=False)
    print(f"✓ Submission escrita: {sub_path.name}")
else:
    print(f"\n(Ningún experimento superó baseline 0.9881. Mejor: {best_strat[0]} = {best_val:.4f})")
