# Audio / Bioacoustics Competitions (e.g., BirdCLEF+ 2026)

## Overview

BirdCLEF+ 2026 tasks: identify bird/animal species from audio recordings in the Pantanal.
- Metric: **padded cMAP** (class-wise mean average precision)
- Data: long soundscape recordings + short labeled clips per species
- Challenge: rare species, overlapping calls, background noise, domain shift

## Core Pipeline

```python
import librosa, numpy as np, torch
import torch.nn as nn
import timm

# 1. Load audio and create mel spectrogram
def audio_to_melspec(path, sr=32000, duration=5, n_mels=128, fmin=20, fmax=16000):
    y, _ = librosa.load(path, sr=sr, duration=duration)
    if len(y) < sr * duration:
        y = np.pad(y, (0, sr * duration - len(y)))  # pad short clips
    
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, fmin=fmin, fmax=fmax,
        n_fft=1024, hop_length=320
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)

# 2. CNN/ViT on spectrogram (treated as image)
class BirdModel(nn.Module):
    def __init__(self, num_classes, model_name='efficientnet_b0'):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, in_chans=1)
        n_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(n_features, num_classes)
    
    def forward(self, x):
        # x: (B, 1, n_mels, time)
        return self.backbone(x)
```

## Handling Long Soundscapes (Inference)

```python
def predict_soundscape(model, path, sr=32000, chunk_duration=5, stride=2.5):
    """Slide window over long recording, aggregate predictions per 5s chunk."""
    y, _ = librosa.load(path, sr=sr)
    total_duration = len(y) / sr
    
    predictions = {}
    t = 0
    while t + chunk_duration <= total_duration:
        chunk = y[int(t*sr):int((t+chunk_duration)*sr)]
        mel = audio_to_melspec_from_array(chunk, sr)
        mel_tensor = torch.tensor(mel).unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            logits = model(mel_tensor)
            probs = torch.sigmoid(logits).squeeze().numpy()
        
        # Store for each 5s window
        predictions[round(t + chunk_duration/2, 1)] = probs
        t += stride
    
    return predictions
```

## Data Augmentation (Critical for Audio)

```python
import albumentations as A
import torchaudio.transforms as T

# Spectrogram augmentations
def spec_augment(mel, freq_mask=15, time_mask=30):
    """SpecAugment: mask frequency bands and time steps."""
    mel = mel.copy()
    # Frequency masking
    f = np.random.randint(0, freq_mask)
    f0 = np.random.randint(0, mel.shape[0] - f)
    mel[f0:f0+f, :] = mel.mean()
    # Time masking
    t = np.random.randint(0, time_mask)
    t0 = np.random.randint(0, mel.shape[1] - t)
    mel[:, t0:t0+t] = mel.mean()
    return mel

# Mixup for multi-label audio
def mixup_audio(mel1, label1, mel2, label2, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    mixed_mel = lam * mel1 + (1 - lam) * mel2
    mixed_label = lam * label1 + (1 - lam) * label2
    return mixed_mel, mixed_label
```

## Handling Class Imbalance (Rare Species)

```python
# Focal loss for imbalanced multi-label classification
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()

# Oversampling rare species in DataLoader
from torch.utils.data import WeightedRandomSampler

species_counts = train_df['primary_label'].value_counts()
weights = train_df['primary_label'].map(lambda x: 1.0 / species_counts[x])
sampler = WeightedRandomSampler(weights, len(weights))
```

## Tips
- Pre-trained models on ImageNet work well on spectrograms (transfer learning)
- Use 5-second clips with 2.5s stride for test-time inference
- Secondary labels (background species) are noisy — weight them lower (0.3–0.5)
- Model ensemble: EfficientNet + ConvNeXt + ViT typically outperforms single model
- Normalize mel spectrogram per-clip (zero mean, unit variance)
