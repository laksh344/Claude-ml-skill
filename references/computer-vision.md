# Computer Vision Competitions

## Overview

Common types: image classification, object detection, instance/semantic segmentation,
satellite/aerial imagery, medical imaging (separate file), depth estimation, image generation.

Winning trend (2025–2026): **Transformers (ViT, Swin) have overtaken CNNs** for top solutions.
Best single model: ConvNeXt-Large or ViT-Large; best ensemble: mix transformer + CNN.

## Image Classification Pipeline

```python
import timm, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# 1. Dataset
class ImageDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
    
    def __len__(self): return len(self.df)
    
    def __getitem__(self, idx):
        img = Image.open(f"{self.img_dir}/{self.df.iloc[idx]['image_id']}.jpg").convert('RGB')
        label = self.df.iloc[idx]['label']
        if self.transform: img = self.transform(img)
        return img, label

# 2. Transforms
train_tfm = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_tfm = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. Model (timm)
model = timm.create_model('convnext_large', pretrained=True, num_classes=num_classes)
# Options: 'vit_large_patch16_224', 'swin_large_patch4_window7_224', 'efficientnet_b4'

# 4. Training loop with mixed precision
scaler = torch.cuda.amp.GradScaler()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

for epoch in range(epochs):
    model.train()
    for imgs, labels in train_loader:
        imgs, labels = imgs.cuda(), labels.cuda()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(imgs)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    scheduler.step()
```

## Object Detection (YOLO / DETR)

```python
# YOLOv8 (ultralytics) — fastest to get started
from ultralytics import YOLO

model = YOLO('yolov8l.pt')  # or yolov8x.pt for best accuracy
results = model.train(
    data='dataset.yaml',    # contains train/val paths + class names
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,
    augment=True,
    mosaic=1.0,
    mixup=0.1,
)

# DETR / RT-DETR for transformer-based detection
from transformers import AutoModelForObjectDetection, AutoImageProcessor
model = AutoModelForObjectDetection.from_pretrained("PekingU/rtdetr-r50")
```

## Segmentation

```python
# Semantic segmentation with segmentation-models-pytorch
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="resnet34",        # or "efficientnet-b4", "mit_b2"
    encoder_weights="imagenet",
    in_channels=3,
    classes=num_classes,
)

# Loss: combine Dice + BCE for segmentation
dice_loss = smp.losses.DiceLoss(mode='binary')
bce_loss = nn.BCEWithLogitsLoss()
loss = 0.5 * dice_loss(logits, masks) + 0.5 * bce_loss(logits, masks)
```

## Satellite / Aerial Imagery

```python
import rasterio
import numpy as np

# Multi-band satellite (RGB + NIR + SWIR etc.)
with rasterio.open("scene.tif") as src:
    img = src.read()  # (bands, H, W)
    transform = src.transform

# Compute NDVI (vegetation index)
red = img[3].astype(float)   # band index depends on sensor
nir = img[4].astype(float)
ndvi = (nir - red) / (nir + red + 1e-8)

# For large tiles: sliding window inference
def predict_tile(model, tile, patch_size=512, overlap=64):
    # Crop into overlapping patches, infer, stitch back
    ...
```

## Key Tips
- **Test Time Augmentation (TTA)**: average predictions over flips/rotations at inference
- **Label smoothing**: 0.1–0.2 smoothing helps with noisy labels
- **Progressive resizing**: train at 224 → fine-tune at 384 → infer at 512
- **Albumentations** for fast GPU-accelerated augmentation
- Satellite: check if bands are RGB-only or multi-spectral; normalize per-band
