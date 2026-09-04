#!/usr/bin/env python3
"""
FAST START: Flood Detection from Satellite Imagery
Single-file implementation - Run this to get started immediately

Prerequisites:
  pip install torch torchvision numpy rasterio albumentations opencv-python scipy tqdm matplotlib

Quick Start:
  1. Download Sen1Floods11 dataset from: https://github.com/cloudtostreet/Sen1Floods11
  2. Place images in: ./data/images/   and masks in: ./data/masks/
  3. Run: python flood_detector.py --mode train
  4. Predict: python flood_detector.py --mode predict --image path/to/sar.tif
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import rasterio
from albumentations import Compose, HorizontalFlip, VerticalFlip, RandomRotate90, Normalize
from albumentations.pytorch import ToTensorV2

# ===================== MODEL =====================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_ch=1, n_cls=1):
        super().__init__()
        self.inc = DoubleConv(n_ch, 64)
        self.d1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.d2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.d3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.d4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))
        self.u1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), DoubleConv(1024+512, 512))
        self.u2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), DoubleConv(512+256, 256))
        self.u3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), DoubleConv(256+128, 128))
        self.u4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), DoubleConv(128+64, 64))
        self.out = nn.Conv2d(64, n_cls, 1)

    def crop(self, x, target):
        dH, dW = x.size()[2] - target.size()[2], x.size()[3] - target.size()[3]
        return x[:, :, dH//2:dH//2+target.size()[2], dW//2:dW//2+target.size()[3]]

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.d1(x1); x3 = self.d2(x2); x4 = self.d3(x3); x5 = self.d4(x4)
        x = self.u1[1](torch.cat([self.crop(x4, self.u1[0](x5)), self.u1[0](x5)], 1))
        x = self.u2[1](torch.cat([self.crop(x3, self.u2[0](x)), self.u2[0](x)], 1))
        x = self.u3[1](torch.cat([self.crop(x2, self.u3[0](x)), self.u3[0](x)], 1))
        x = self.u4[1](torch.cat([self.crop(x1, self.u4[0](x)), self.u4[0](x)], 1))
        return self.out(x)

# ===================== DATASET =====================
class FloodDataset(Dataset):
    def __init__(self, img_dir, mask_dir, train=True):
        self.img_dir, self.mask_dir = img_dir, mask_dir
        self.files = [f for f in os.listdir(img_dir) if f.endswith(('.npy','.tif','.png'))]
        aug = [HorizontalFlip(p=0.5), VerticalFlip(p=0.5), RandomRotate90(p=0.5)] if train else []
        self.tf = Compose(aug + [Normalize(mean=[0.5], std=[0.5]), ToTensorV2()])

    def __len__(self): return len(self.files)

    def __getitem__(self, i):
        f = self.files[i]
        img = np.load(os.path.join(self.img_dir, f)) if f.endswith('.npy') else rasterio.open(os.path.join(self.img_dir, f)).read(1)
        msk = np.load(os.path.join(self.mask_dir, f.replace('.tif','.npy').replace('.png','.npy'))) if self.mask_dir else np.zeros_like(img)
        if img.ndim == 2: img = img[np.newaxis,...]
        t = self.tf(image=img.transpose(1,2,0), mask=msk)
        return t['image'], t['mask'].unsqueeze(0).float()

# ===================== LOSS & METRICS =====================
class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
    def forward(self, p, t):
        p = torch.sigmoid(p); p, t = p.view(-1), t.view(-1)
        dice = 1 - (2*(p*t).sum()+1)/(p.sum()+t.sum()+1)
        return 0.5*self.bce(p, t) + 0.5*dice

def iou(p, t, thr=0.5):
    p = (torch.sigmoid(p)>thr).float(); t = t.float()
    inter = (p*t).sum(); union = ((p+t)>0).float().sum()
    return (inter+1e-6)/(union+1e-6)

# ===================== TRAIN =====================
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")

    train_ds = FloodDataset(args.img_dir, args.mask_dir, train=True)
    val_ds = FloodDataset(args.val_img_dir, args.val_mask_dir, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = UNet(n_ch=args.channels, n_cls=1).to(device)
    criterion = DiceBCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    scaler = GradScaler()

    best_iou = 0
    for epoch in range(args.epochs):
        model.train(); train_loss = 0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            with autocast():
                out = model(imgs); loss = criterion(out, masks)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            train_loss += loss.item()

        model.eval(); val_iou = 0; val_loss = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                with autocast():
                    out = model(imgs)
                    val_loss += criterion(out, masks).item()
                    val_iou += iou(out, masks).item()

        avg_vl = val_loss/len(val_loader); avg_vi = val_iou/len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, Val Loss={avg_vl:.4f}, Val IoU={avg_vi:.4f}")
        scheduler.step(avg_vi)

        if avg_vi > best_iou:
            best_iou = avg_vi
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save({'model': model.state_dict(), 'iou': best_iou}, os.path.join(args.save_dir, 'best.pth'))
            print(f"  >>> Best model saved! IoU={best_iou:.4f}")

# ===================== PREDICT =====================
def predict(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_ch=args.channels, n_cls=1).to(device)
    ckpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(ckpt['model']); model.eval()
    print(f"Loaded model with IoU={ckpt.get('iou','N/A')}")

    with rasterio.open(args.image) as src:
        img = src.read().astype(np.float32); profile = src.profile
    if img.ndim == 2: img = img[np.newaxis,...]
    img = (img - 0.5) / 0.5

    # Sliding window prediction
    C, H, W = img.shape; ps = args.patch_size; stride = ps // 2
    prob = np.zeros((H, W), dtype=np.float32); wgt = np.zeros((H, W), dtype=np.float32)

    for y in tqdm(range(0, H-ps+1, stride), desc="Predicting"):
        for x in range(0, W-ps+1, stride):
            patch = img[:, y:y+ps, x:x+ps]
            if patch.shape[1:] != (ps, ps):
                patch = np.pad(patch, ((0,0),(0,ps-patch.shape[1]),(0,ps-patch.shape[2])))
            with torch.no_grad():
                p = torch.sigmoid(model(torch.from_numpy(patch).unsqueeze(0).to(device))).squeeze().cpu().numpy()
            prob[y:y+ps, x:x+ps] += p; wgt[y:y+ps, x:x+ps] += 1

    prob = np.divide(prob, wgt, out=np.zeros_like(prob), where=wgt!=0)
    mask = (prob > args.threshold).astype(np.uint8)

    # Save outputs
    profile.update(count=1, dtype=mask.dtype)
    with rasterio.open(args.output.replace('.tif','_mask.tif'), 'w', **profile) as dst:
        dst.write(mask, 1)
    profile.update(dtype=prob.dtype)
    with rasterio.open(args.output.replace('.tif','_prob.tif'), 'w', **profile) as dst:
        dst.write(prob, 1)
    print(f"Saved to {args.output}")

# ===================== MAIN =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast Flood Detection")
    parser.add_argument('--mode', choices=['train','predict'], required=True)
    parser.add_argument('--img_dir', default='./data/train/images')
    parser.add_argument('--mask_dir', default='./data/train/masks')
    parser.add_argument('--val_img_dir', default='./data/val/images')
    parser.add_argument('--val_mask_dir', default='./data/val/masks')
    parser.add_argument('--model_path', default='./checkpoints/best.pth')
    parser.add_argument('--image', default='')
    parser.add_argument('--output', default='./output/prediction.tif')
    parser.add_argument('--channels', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--save_dir', default='./checkpoints')
    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    else:
        predict(args)
