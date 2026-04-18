# Social Good / Geospatial Competitions (Zindi, DrivenData, NASA/NOAA)

## Overview

Platforms: Zindi (Africa-focused, ~100K users), DrivenData (NASA, NOAA, nonprofits),
AI for Good (UN), Grand Challenge (medical/satellite).

Common tasks: satellite image analysis, air quality estimation, crop mapping,
disaster damage detection, disease prediction, kelp/ecosystem segmentation,
wildlife conservation, climate modeling, infrastructure assessment.

## Satellite / Geospatial Imagery

```python
import rasterio
import numpy as np
from rasterio.plot import show
import geopandas as gpd
from shapely.geometry import box

# Read multi-band satellite image
with rasterio.open("scene.tif") as src:
    img = src.read()           # (bands, H, W)
    meta = src.meta
    bounds = src.bounds
    crs = src.crs
    print(f"Bands: {src.count}, CRS: {crs}, Shape: {src.width}x{src.height}")

# Compute spectral indices
def compute_ndvi(img, red_band=3, nir_band=4):
    red = img[red_band].astype(float)
    nir = img[nir_band].astype(float)
    return (nir - red) / (nir + red + 1e-8)

def compute_ndwi(img, green_band=2, nir_band=4):
    """Normalized Difference Water Index"""
    green = img[green_band].astype(float)
    nir = img[nir_band].astype(float)
    return (green - nir) / (green + nir + 1e-8)

# Stack indices as extra features
ndvi = compute_ndvi(img)
ndwi = compute_ndwi(img)
img_with_indices = np.concatenate([img, ndvi[np.newaxis], ndwi[np.newaxis]], axis=0)
```

## Semantic Segmentation for Satellite (Kelp Forest etc.)

```python
import segmentation_models_pytorch as smp
import torch

# U-Net with pretrained encoder
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=6,          # RGB + NIR + NDVI + NDWI
    classes=1,              # binary: kelp / no kelp
    activation=None,
)

# Dice + BCE loss (standard for segmentation)
criterion = smp.losses.DiceLoss(mode='binary') + \
            smp.losses.SoftBCEWithLogitsLoss()

# Metrics
tp, fp, fn, tn = smp.metrics.get_stats(
    outputs.sigmoid() > 0.5, masks.long(), mode='binary')
iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
```

## Tabular Zindi / DrivenData Tasks

```python
# Common pattern: CSV features + geolocation
# e.g., air quality prediction, crop yield, disease rates

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

df = pd.read_csv("train.csv")

# Handle categorical (country, district, crop_type)
for col in df.select_dtypes('object').columns:
    if col != 'target':
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Geospatial features from lat/lon
df['lat_bin'] = pd.cut(df['latitude'], bins=20, labels=False)
df['lon_bin'] = pd.cut(df['longitude'], bins=20, labels=False)
df['distance_to_equator'] = df['latitude'].abs()

# Merge external data (population density, elevation, climate)
# e.g., from WorldBank, OpenStreetMap, ERA5 climate reanalysis
```

## External Datasets Commonly Used

```python
# Free geospatial data sources:
# - Copernicus Sentinel-2 (10m multispectral satellite)
# - NASA SRTM (elevation)
# - ERA5 (climate reanalysis: temp, precip, wind)
# - WorldPop (population density)
# - OpenStreetMap (roads, buildings)

# Example: pull ERA5 climate features for given coordinates
import ee  # Google Earth Engine (requires authentication)

# Or use pre-extracted CSV files shared in competition notebooks
```

## Zindi-Specific Tips

- **Team size**: usually unlimited, but merge deadline is 1 week before close
- **Submission limits**: typically 5–10 per day
- **Data**: often smaller than Kaggle (thousands of rows, not millions) → overfit risk!
- **External data**: usually allowed unless explicitly forbidden — always check rules
- **Winning patterns**: tree models (LGBM) + careful feature engineering for tabular;
  SegFormer / UNet for satellite; DeBERTa / smaller LLMs for NLP tasks on limited data
- **Community**: African data scientists often have domain knowledge edge — read their notebooks!

## DrivenData Tips

- Winners must typically provide detailed write-ups (solution documentation required)
- NASA/NOAA competitions often have strict data licensing — check before using external
- Many comps use custom evaluation APIs (similar to Kaggle time-series API)
- Social good framing: explain your solution's real-world impact in write-up
