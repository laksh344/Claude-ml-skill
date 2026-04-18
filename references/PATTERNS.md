# kaggle-ml Competition Patterns

Reusable patterns for ML competitions. Reference these to avoid common mistakes.

---

## Pattern 1: Safe Data Loading with Validation

Always validate input before processing — never assume clean data in competitions.

```python
import pandas as pd
import numpy as np

def load_and_validate(path: str, target_col: str = None) -> pd.DataFrame:
    """Load competition data with automatic validation."""
    if not path:
        raise ValueError("Data path required")

    df = pd.read_csv(path) if path.endswith('.csv') else pd.read_parquet(path)

    # Report shape and quality immediately
    print(f"Shape: {df.shape}")
    print(f"Missing: {df.isnull().sum().sum()} total nulls")
    print(f"Duplicates: {df.duplicated().sum()}")

    if target_col and target_col in df.columns:
        print(f"Target balance:\n{df[target_col].value_counts(normalize=True)}")

    return df
```

## Pattern 2: Leak-Safe CV Pipeline

Fit ALL preprocessing INSIDE the CV loop — never on the full training set.

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import numpy as np

# ✅ CORRECT — scaler fitted per fold
def cv_with_preprocessing(X, y, model_fn, n_splits=5):
    skf  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oofs = np.zeros(len(X))

    for fold, (tr, val) in enumerate(skf.split(X, y)):
        scaler = StandardScaler()                  # new scaler per fold
        X_tr   = scaler.fit_transform(X[tr])       # fit ONLY on train fold
        X_val  = scaler.transform(X[val])          # transform val with train stats
        model  = model_fn()
        model.fit(X_tr, y[tr])
        oofs[val] = model.predict_proba(X_val)[:, 1]

    return oofs

# ❌ WRONG — leaks test statistics into training
# scaler = StandardScaler().fit(X_train)  # ← fitted on full train before CV split
```

## Pattern 3: Robust Error Handling in Pipelines

```python
import logging
logger = logging.getLogger(__name__)

def safe_predict(model, X, fallback=0.5):
    """Predict with fallback for production robustness."""
    try:
        preds = model.predict_proba(X)[:, 1]
        if np.isnan(preds).any() or np.isinf(preds).any():
            logger.warning("NaN/Inf in predictions — returning fallback")
            return np.full(len(X), fallback)
        return preds
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return np.full(len(X), fallback)
```

## Pattern 4: Experiment Tracking

Log every run — one missed result can cost you the competition.

```python
import wandb, time
from datetime import datetime

def track_experiment(params: dict, metrics: dict, model_name: str):
    """Standard experiment logging for competition runs."""
    wandb.init(
        project = "kaggle-competition",
        name    = f"{model_name}_{datetime.now().strftime('%m%d_%H%M')}",
        config  = params,
        reinit  = True
    )
    wandb.log(metrics)
    wandb.finish()
    print(f"[{model_name}] CV: {metrics.get('cv_score', '?'):.5f}")
```

## Pattern 5: Submission Safety Check

Always validate before submitting — format bugs waste submission quota.

```python
def validate_submission(sub: pd.DataFrame, sample_sub: pd.DataFrame) -> bool:
    """Catch format errors before wasting a submission."""
    errors = []

    if list(sub.columns) != list(sample_sub.columns):
        errors.append(f"Column mismatch: got {list(sub.columns)}")
    if len(sub) != len(sample_sub):
        errors.append(f"Row count mismatch: {len(sub)} vs {len(sample_sub)}")
    if sub.isnull().any().any():
        errors.append("NaN values in submission")
    if (sub.select_dtypes('number') < 0).any().any():
        errors.append("Negative predictions (check sigmoid/softmax)")

    for err in errors:
        print(f"❌ {err}")

    if not errors:
        print("✅ Submission valid — safe to upload")
        print(sub.describe())

    return len(errors) == 0
```

## Pattern 6: Config Loading

```python
import yaml

def load_config(config_path: str = "assets/config.yaml") -> dict:
    """Load and validate skill configuration."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Apply competition defaults from config
    comp = config.get('settings', {}).get('competition', {})
    return {
        'cv_folds':      comp.get('default_cv_folds', 5),
        'random_state':  comp.get('default_random_state', 42),
        'min_cot_ratio': comp.get('min_cot_ratio', 0.75),
        'answer_range':  comp.get('aimo_answer_range', [0, 99999]),
    }
```

---

## Anti-Patterns to Avoid

### ❌ Data Leakage (Most Common Competition Killer)

```python
# BAD — fitting on full dataset before split
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)
X_train, X_test = train_test_split(X_scaled)   # test already leaked into scaler!

# GOOD — fit inside CV loop or after split only
X_train, X_test = train_test_split(X, random_state=42)
scaler = StandardScaler().fit(X_train)          # fit ONLY on train
X_test_scaled = scaler.transform(X_test)
```

### ❌ Public LB Chasing

```python
# BAD — overfitting to public leaderboard
if public_lb_score > best_public:
    use_this_model()     # may catastrophically fail on private LB

# GOOD — trust cross-validation
if cv_score > best_cv:
    use_this_model()     # stable across train/private LB splits
```

### ❌ Swallowing Exceptions

```python
# BAD
try:
    train_model()
except:
    pass     # silent failure wastes hours

# GOOD
try:
    train_model()
except MemoryError:
    logger.error("OOM — reduce batch size or use gradient checkpointing")
    raise
except RuntimeError as e:
    logger.error(f"Training failed: {e}")
    raise
```

### ❌ Excess Capacity on Sparse Logic (2026 Research Finding)

```python
# BAD for ARC-AGI / sparse reasoning tasks
model = HugeTransformer(layers=48, d_model=4096, moe=True)
# → catastrophic overfitting on sparse logic; TRM (7M params) beats this

# GOOD — minimal model with deep refinement
model = TinyRecursiveModel(d_model=64, n_layers=2)  # refine recursively
```

### ❌ Random Shuffle on Time Series

```python
# BAD
X_train, X_test = train_test_split(X, shuffle=True)  # mixes future into past!

# GOOD
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tscv.split(X):
    # val is always chronologically after train
```

---

## Competition Type → Pattern Mapping

| Competition | Key Pattern to Use |
|-------------|-------------------|
| Tabular | Pattern 2 (CV) + Pattern 3 (safe predict) + Pattern 5 (submission) |
| Time Series | Pattern 2 with `TimeSeriesSplit` + Anti-pattern: no shuffle |
| ARC-AGI-3 | Pattern 3 (error handling) + avoid Anti-pattern: excess capacity |
| Nemotron / LLM | Pattern 4 (tracking) + config `min_cot_ratio ≥ 0.75` |
| AIMO Math | Pattern 5 (validate answers in [0, 99999]) + majority vote |
| All Competitions | Pattern 1 (validate) + Pattern 4 (track) + Pattern 5 (submission) |
