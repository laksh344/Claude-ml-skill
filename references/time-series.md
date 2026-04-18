# Time Series & Financial Forecasting Competitions

## Overview

Types: financial market prediction (Jane Street), energy, weather, demand forecasting.
Jane Street: real-time inference, ~9000 features, online learning, daily time-based CV.

## Core Time Series Pipeline

```python
import pandas as pd, numpy as np
from sklearn.model_selection import TimeSeriesSplit

df = pd.read_parquet("train.parquet")  # large competitions often use parquet

# CRITICAL: Always use time-based CV, never random split
tscv = TimeSeriesSplit(n_splits=5, gap=0)  # gap prevents leakage

for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
    # train / evaluate here
```

## Feature Engineering

```python
# Lag features
for lag in [1, 2, 3, 5, 10, 20]:
    df[f'lag_{lag}'] = df.groupby('symbol')['value'].shift(lag)

# Rolling statistics
for window in [5, 10, 20, 50]:
    df[f'rolling_mean_{window}'] = df.groupby('symbol')['value'].transform(
        lambda x: x.rolling(window, min_periods=1).mean())
    df[f'rolling_std_{window}'] = df.groupby('symbol')['value'].transform(
        lambda x: x.rolling(window, min_periods=1).std())

# Date/time features
df['hour'] = df['timestamp'].dt.hour
df['dayofweek'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

# Fourier features for seasonality
for period in [7, 30, 365]:
    df[f'sin_{period}'] = np.sin(2 * np.pi * df['day_of_year'] / period)
    df[f'cos_{period}'] = np.cos(2 * np.pi * df['day_of_year'] / period)
```

## Models for Time Series

```python
# LightGBM (strong baseline, handles missing values well)
import lightgbm as lgb

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 255,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'min_child_samples': 20,
}

# LSTM for sequential patterns
import torch, torch.nn as nn

class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # predict from last hidden state

# Temporal Fusion Transformer (state-of-art for multi-horizon)
# pip install pytorch-forecasting
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
```

## Jane Street Specifics

```python
# Online inference API pattern (Kaggle time-series API)
import kaggle_evaluation.jane_street_inference_server as server

class MyModel:
    def __init__(self):
        self.model = load_model()
        self.history = []  # for lag features
    
    def predict(self, test: pd.DataFrame, lags: pd.DataFrame | None) -> pd.DataFrame:
        # lags: previous rows with known responders
        features = self.engineer_features(test, lags)
        preds = self.model.predict(features)
        
        # Neutralize predictions (reduce market-beta exposure)
        preds = preds - preds.mean()
        
        test['responder_6'] = preds
        return test[['row_id', 'responder_6']]
    
    def engineer_features(self, test, lags):
        # Use lags for lag features
        ...

inference_server = server.InferenceServer(MyModel())
inference_server.serve()
```

## Anti-Leakage Checklist for Time Series
- [ ] All feature computation uses only past data (`.shift(1)` minimum)
- [ ] Validation set is always after training set in time
- [ ] No future information in lag/rolling features
- [ ] If using neural nets: batch norm replaced with layer norm (no future stats leakage)
- [ ] For online competitions: model must be stateless or properly track state between batches
