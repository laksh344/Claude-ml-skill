# Tabular / Structured Data Competitions

## Overview

Most common competition type on Kaggle. CSV/parquet data with numeric, categorical,
and sometimes text features. GBMs (LightGBM, XGBoost, CatBoost) dominate.

## Full Pipeline

```python
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
import optuna

# Load data
train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")
sub   = pd.read_csv("sample_submission.csv")

TARGET = 'target'
ID_COL = 'id'
FEATURES = [c for c in train.columns if c not in [TARGET, ID_COL]]

X = train[FEATURES]
y = train[TARGET]
X_test = test[FEATURES]
```

## Feature Engineering

```python
# Numeric interactions
df['ratio_a_b']   = df['col_a'] / (df['col_b'] + 1e-8)
df['product_a_b'] = df['col_a'] * df['col_b']
df['diff_a_b']    = df['col_a'] - df['col_b']

# Aggregate features (group statistics)
agg = df.groupby('category')['value'].agg(['mean','std','min','max','median'])
agg.columns = [f'cat_{s}' for s in agg.columns]
df = df.merge(agg, on='category', how='left')

# Encoding categoricals
# Low-cardinality: one-hot
df = pd.get_dummies(df, columns=['low_card_col'])

# High-cardinality: target encoding (with CV to avoid leakage)
from category_encoders import TargetEncoder
te = TargetEncoder(cols=['high_card_col'])
# Fit ONLY on train fold inside CV loop

# Frequency encoding
freq = df['cat_col'].value_counts()
df['cat_col_freq'] = df['cat_col'].map(freq)

# Handle missing values
df['col_missing_flag'] = df['col'].isnull().astype(int)
df['col'] = df['col'].fillna(df['col'].median())
```

## Cross-Validation Loop (Gold Standard)

```python
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
# For regression: KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

oof_preds  = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
feature_importance = pd.DataFrame()

lgb_params = {
    'objective':        'binary',
    'metric':           'auc',
    'learning_rate':    0.05,
    'num_leaves':       63,
    'max_depth':        -1,
    'min_child_samples':20,
    'subsample':        0.8,
    'subsample_freq':   1,
    'colsample_bytree': 0.8,
    'reg_alpha':        0.1,
    'reg_lambda':       1.0,
    'n_estimators':     2000,
    'verbosity':        -1,
    'random_state':     42,
}

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)]
    )

    oof_preds[val_idx]  = model.predict_proba(X_val)[:, 1]
    test_preds          += model.predict_proba(X_test)[:, 1] / N_FOLDS

    fi = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_,
                        'fold': fold})
    feature_importance = pd.concat([feature_importance, fi])

cv_score = roc_auc_score(y, oof_preds)
print(f"CV AUC: {cv_score:.5f}")
```

## Hyperparameter Tuning with Optuna

```python
def objective(trial):
    params = {
        'objective':        'binary',
        'metric':           'auc',
        'verbosity':        -1,
        'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves':       trial.suggest_int('num_leaves', 20, 300),
        'min_child_samples':trial.suggest_int('min_child_samples', 10, 100),
        'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha':        trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda':       trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'n_estimators':     1000,
        'early_stopping_rounds': 50,
    }
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for tr_idx, val_idx in skf.split(X, y):
        m = lgb.LGBMClassifier(**params)
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx],
              eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(-1)])
        pred = m.predict_proba(X.iloc[val_idx])[:, 1]
        scores.append(roc_auc_score(y.iloc[val_idx], pred))
    return np.mean(scores)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, show_progress_bar=True)
print("Best params:", study.best_params)
```

## Multi-Model Ensemble

```python
# 1. LightGBM  (already above)
# 2. XGBoost
xgb_params = {
    'objective':    'binary:logistic',
    'eval_metric':  'auc',
    'learning_rate':0.05,
    'max_depth':    6,
    'subsample':    0.8,
    'colsample_bytree':0.8,
    'n_estimators': 2000,
    'early_stopping_rounds': 100,
    'random_state': 42,
}
xgb_model = xgb.XGBClassifier(**xgb_params)

# 3. CatBoost (handles categoricals natively)
cat_model = CatBoostClassifier(
    iterations=2000, learning_rate=0.05,
    depth=6, l2_leaf_reg=3,
    eval_metric='AUC', random_seed=42, verbose=100,
    cat_features=cat_feature_indices
)

# 4. Blend
final = 0.4 * lgb_test + 0.3 * xgb_test + 0.3 * cat_test

# 5. Stacking (meta-learner on OOF predictions)
from sklearn.linear_model import LogisticRegression
meta_X = np.column_stack([lgb_oof, xgb_oof, cat_oof])
meta_model = LogisticRegression()
meta_model.fit(meta_X, y)
```

## Handling Imbalanced Data

```python
# Oversample minority class
from imblearn.over_sampling import SMOTE
X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)

# Or use scale_pos_weight in LightGBM/XGBoost
pos_weight = (y == 0).sum() / (y == 1).sum()
params['scale_pos_weight'] = pos_weight  # XGBoost
params['is_unbalance'] = True            # LightGBM alternative
```

## Pseudo-Labeling

```python
# Use high-confidence test predictions as additional training data
threshold_high = 0.9
threshold_low  = 0.1

pseudo_mask   = (test_preds > threshold_high) | (test_preds < threshold_low)
pseudo_labels = (test_preds[pseudo_mask] > 0.5).astype(int)

X_augmented = pd.concat([X, X_test[pseudo_mask]])
y_augmented = pd.concat([y, pd.Series(pseudo_labels)])
# Retrain with augmented data
```

## Tips
- Always check for **data leakage** — look for suspiciously high CV scores
- **Feature importance**: drop near-zero importance features to reduce noise
- **Target encoding** must be done inside CV folds to avoid leakage
- **Log-transform** skewed targets (e.g., `np.log1p(y)`) for regression
- Check if train/test **distributions differ** — use adversarial validation
