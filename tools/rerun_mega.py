#!/usr/bin/env python3
"""Re-hace el mega ensemble multi-scale sin EMA (que estaba bugueado).
No re-entrena nada, solo carga checkpoints de disco.
"""
import os, sys, math, warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional

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

from drowsy_cnn.checkpoints import load_checkpoint
from drowsy_cnn.config import resolve_device

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'
IMG_DIR = DATA_DIR / 'images'
CKPT_DIR = ROOT / 'checkpoints'
OUT_DIR = ROOT / 'outputs'
SEED = 42
IMG_W, IMG_H = 1920, 1080
CLASSES = ['awake', 'drowsy']
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEVICE = resolve_device()


def build_transforms(size, use_imagenet_stats=True):
    mean = IMAGENET_MEAN if use_imagenet_stats else [0.4752, 0.4592, 0.4563]
    std = IMAGENET_STD if use_imagenet_stats else [0.2500, 0.2351, 0.2157]
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.3))


def xyxy_to_cxcywh_norm(bbox, img_w, img_h):
    x1, y1, x2, y2 = bbox
    return [((x1+x2)/2)/img_w, ((y1+y2)/2)/img_h, (x2-x1)/img_w, (y2-y1)/img_h]


def cxcywh_norm_to_xyxy(bb, w, h):
    cx, cy, bw, bh = bb.unbind(-1)
    return torch.stack([(cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h], dim=-1)


class DrowsyDataset(Dataset):
    def __init__(self, df, img_dir, transform, is_test=False):
        self.df = df.reset_index(drop=True); self.img_dir = Path(img_dir)
        self.transform = transform; self.is_test = is_test
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.cvtColor(cv2.imread(str(self.img_dir / row['filename'])), cv2.COLOR_BGR2RGB)
        if self.is_test:
            bboxes = [[IMG_W*0.4, IMG_H*0.4, IMG_W*0.6, IMG_H*0.6]]; labs = ['awake']
        else:
            bboxes = [[row.xmin, row.ymin, row.xmax, row.ymax]]; labs = [row['class']]
        out = self.transform(image=img, bboxes=bboxes, class_labels=labs)
        img_t = out['image']
        if self.is_test:
            return img_t, row['filename']
        aug_bbox = out['bboxes'][0] if out['bboxes'] else bboxes[0]
        H, W = img_t.shape[-2:]
        return img_t, torch.tensor(CLS2IDX[out['class_labels'][0]], dtype=torch.long), \
               torch.tensor(xyxy_to_cxcywh_norm(list(aug_bbox), img_w=W, img_h=H), dtype=torch.float32)


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


def build_pretrained(name, n_classes=2, dropout_head=0.3):
    if name == 'resnet18':
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        fd = m.fc.in_features; m.fc = nn.Identity()
    elif name == 'mobilenet_v3_small':
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        fd = m.classifier[0].in_features; m.classifier = nn.Identity()
    elif name == 'efficientnet_b0':
        m = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0, global_pool='')
        fd = m.num_features
    return MultitaskWrapper(m, feat_dim=fd, n_classes=n_classes, dropout_head=dropout_head)


def bbox_dice(pred_cxcywh, true_cxcywh, eps=1e-7):
    def to_xyxy(b):
        cx, cy, w, h = b.unbind(-1)
        return torch.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], dim=-1)
    p, t = to_xyxy(pred_cxcywh), to_xyxy(true_cxcywh)
    x1 = torch.max(p[..., 0], t[..., 0]); y1 = torch.max(p[..., 1], t[..., 1])
    x2 = torch.min(p[..., 2], t[..., 2]); y2 = torch.min(p[..., 3], t[..., 3])
    inter = (x2-x1).clamp(min=0) * (y2-y1).clamp(min=0)
    ap = (p[..., 2]-p[..., 0]) * (p[..., 3]-p[..., 1])
    at = (t[..., 2]-t[..., 0]) * (t[..., 3]-t[..., 1])
    iou = inter / (ap+at-inter+eps)
    return 2*iou / (1+iou)


# ==== MAIN ====
df_full = pd.read_csv(DATA_DIR / 'train.csv')
df_test = pd.read_csv(DATA_DIR / 'test.csv')
df_tr, df_val = train_test_split(df_full, test_size=0.2, stratify=df_full['class'], random_state=SEED)
df_tr = df_tr.reset_index(drop=True); df_val = df_val.reset_index(drop=True)
print(f"Device: {DEVICE}  |  val: {len(df_val)} samples")

# Cargar todos los checkpoints útiles (SIN EMA, SIN folds)
CKPT_LIST = [
    ('p5_resnet18',            'resnet18'),
    ('p5_mobilenet_v3_small',  'mobilenet_v3_small'),
    ('p5_efficientnet_b0',     'efficientnet_b0'),
    ('p17_effnet_snapshot_snap0', 'efficientnet_b0'),
    ('p17_effnet_snapshot_snap1', 'efficientnet_b0'),
    ('p17_effnet_snapshot_snap2', 'efficientnet_b0'),
]
models_list, names = [], []
for name, kind in CKPT_LIST:
    path = CKPT_DIR / f'{name}.pt'
    if not path.exists(): print(f"  ⚠ falta {path.name}, skip"); continue
    m = build_pretrained(kind).to(DEVICE)
    load_checkpoint(m, path, DEVICE)
    models_list.append(m); names.append(name)
print(f"Modelos cargados: {len(models_list)}")

# Evaluar cada uno solo
@torch.no_grad()
def eval_single(m, df, size=224):
    m.eval()
    tf = build_transforms(size, use_imagenet_stats=True)
    ld = DataLoader(DrowsyDataset(df, IMG_DIR, tf), batch_size=16, shuffle=False, num_workers=0)
    labs, preds, bbps, bbts = [], [], [], []
    for img, l, bt in ld:
        img = img.to(DEVICE)
        lg, bp = m(img)
        preds.append(lg.argmax(1).cpu()); labs.append(l)
        bbps.append(bp.cpu()); bbts.append(bt)
    p = torch.cat(preds); l = torch.cat(labs)
    bp = torch.cat(bbps); bt = torch.cat(bbts)
    return (p == l).float().mean().item(), bbox_dice(bp, bt).mean().item()

print("\n=== Métricas individuales sobre val ===")
metrics = []
for name, m in zip(names, models_list):
    a, d = eval_single(m, df_val)
    metrics.append((name, a, d))
    print(f"  {name:35s}  acc={a:.4f}  dice={d:.4f}")

# Weighted ensemble por (acc + dice) / 2
weights = [(a + d) / 2 for _, a, d in metrics]
print(f"\nPesos ensemble (mean acc+dice): {[f'{w:.3f}' for w in weights]}")

# Multi-scale TTA ensemble
@torch.no_grad()
def mega(model_list, weights, df, scales=(224, 288), is_test=False):
    accum_probs, accum_bb = 0.0, 0.0; total_w = 0.0
    fnames_out = None; labels_out = None; bbt_out = None
    for scale in scales:
        for w, m in zip(weights, model_list):
            m.eval()
            tf = build_transforms(scale, True)
            ds = DrowsyDataset(df, IMG_DIR, tf, is_test=is_test)
            ld = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
            probs_l, bb_l, fnames, labs, bts = [], [], [], [], []
            for batch in ld:
                if is_test:
                    img, fn = batch; fnames.extend(fn)
                else:
                    img, lab, bt = batch; labs.append(lab); bts.append(bt)
                img = img.to(DEVICE)
                l1, b1 = m(img)
                l2, b2 = m(torch.flip(img, dims=[-1]))
                b2d = b2.clone(); b2d[:, 0] = 1.0 - b2[:, 0]
                probs_l.append(F.softmax((l1+l2)/2, dim=1).cpu())
                bb_l.append(((b1+b2d)/2).cpu())
            probs_np = torch.cat(probs_l).numpy(); bb_np = torch.cat(bb_l).numpy()
            accum_probs = accum_probs + w * probs_np
            accum_bb = accum_bb + w * bb_np
            total_w += w
            if is_test and fnames_out is None: fnames_out = fnames
            if not is_test and labels_out is None:
                labels_out = torch.cat(labs).numpy(); bbt_out = torch.cat(bts).numpy()
    probs = accum_probs / total_w; bb = accum_bb / total_w
    if is_test:
        return {'filenames': fnames_out, 'probs': probs, 'preds': probs.argmax(1), 'bbox_norm': bb}
    return {'preds': probs.argmax(1), 'bbox_norm': bb, 'labels': labels_out, 'bbox_true': bbt_out}

res_val = mega(models_list, weights, df_val, scales=(224, 288))
acc = (res_val['preds'] == res_val['labels']).mean()
dice = bbox_dice(torch.tensor(res_val['bbox_norm']), torch.tensor(res_val['bbox_true'])).mean().item()
print(f"\n=== MEGA v2 (sin EMA, weighted por acc+dice, multi-scale 224+288) ===")
print(f"  val_acc  = {acc:.4f}")
print(f"  val_dice = {dice:.4f}")

# Submission
res_test = mega(models_list, weights, df_test, scales=(224, 288), is_test=True)
xyxy = cxcywh_norm_to_xyxy(torch.tensor(res_test['bbox_norm']), IMG_W, IMG_H)
xyxy[:, 0::2].clamp_(0, IMG_W); xyxy[:, 1::2].clamp_(0, IMG_H)
sub = pd.DataFrame({
    'filename': res_test['filenames'],
    'class': [IDX2CLS[p] for p in res_test['preds']],
    'xmin': xyxy[:,0].round().int().tolist(),
    'ymin': xyxy[:,1].round().int().tolist(),
    'xmax': xyxy[:,2].round().int().tolist(),
    'ymax': xyxy[:,3].round().int().tolist(),
})
sub.to_csv(OUT_DIR / 'submission_mega_v2.csv', index=False)
print(f"\n✓ submission_mega_v2.csv ({len(sub)} filas)")
print(f"Distribución: {sub['class'].value_counts().to_dict()}")
