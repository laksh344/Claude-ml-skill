# Minimal Neural Network / ONNX Competitions (NeuroGolf, CompressARC-style)

## Overview & 2026 Research State

Key 2026 research insight (CompressARC, TRM):
- **Depth of recursive search >> breadth of parameter count** for abstract reasoning
- CompressARC: 76K params solved ARC tasks by framing intelligence as code-golf (MDL)
- TRM: 7M params, 2 layers achieved 45% on ARC-AGI-1, beating massive LLMs
- Adding excess capacity (SwiGLU MoE) to sparse logic tasks → **catastrophic overfitting**

For NeuroGolf and similar: **correct first, smallest second**.

## Minimum Description Length (MDL) — CompressARC Principle

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MDLSolver(nn.Module):
    """
    CompressARC-inspired: train ONLY on the single target puzzle.
    Overfit to compress the puzzle → the compression encodes the solution rule.
    Uses VAE-style loss: KL divergence + reconstruction (MDL objective).
    """
    def __init__(self, n_colors=10, d=32, max_h=30, max_w=30):
        super().__init__()
        self.n_colors = n_colors
        # Equivariant to valid ARC transformations (color permutation, rotation)
        self.encoder  = nn.Sequential(
            nn.Embedding(n_colors, d),
            nn.Linear(d, d), nn.GELU(),
        )
        self.mu_head     = nn.Linear(d, d)
        self.logvar_head = nn.Linear(d, d)
        self.decoder     = nn.Linear(d, n_colors)
    
    def forward(self, x):
        # x: (B, H*W) color indices
        h       = self.encoder(x)                           # (B, HW, d)
        mu      = self.mu_head(h)
        logvar  = self.logvar_head(h)
        # Reparameterization trick
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z   = mu + eps * std
        else:
            z = mu
        return self.decoder(z), mu, logvar
    
    def mdl_loss(self, logits, targets, mu, logvar, beta=0.1):
        """MDL loss: reconstruction + KL divergence (compress puzzle into weights)."""
        recon = F.cross_entropy(logits.reshape(-1, self.n_colors), targets.reshape(-1))
        kl    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + beta * kl


def compress_arc_solve(puzzle, n_steps=1000, lr=1e-3):
    """
    Train exclusively on this puzzle's input/output pairs.
    No pretraining dataset. No domain-specific language.
    Just gradient descent compressing the puzzle into weights.
    """
    model     = MDLSolver()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_inputs  = [torch.tensor(p['input'],  dtype=torch.long).flatten().unsqueeze(0)
                     for p in puzzle['train']]
    train_targets = [torch.tensor(p['output'], dtype=torch.long).flatten()
                     for p in puzzle['train']]
    
    model.train()
    for step in range(n_steps):
        total_loss = 0
        for inp, tgt in zip(train_inputs, train_targets):
            logits, mu, logvar = model(inp)
            loss = model.mdl_loss(logits, tgt, mu, logvar)
            total_loss += loss
        
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if total_loss.item() < 1e-4:
            break
    
    # Predict test input
    model.eval()
    test_inp  = torch.tensor(puzzle['test'][0]['input'], dtype=torch.long).flatten().unsqueeze(0)
    H = len(puzzle['test'][0]['input']); W = len(puzzle['test'][0]['input'][0])
    with torch.no_grad():
        logits, _, _ = model(test_inp)
        pred_flat = logits.squeeze(0).argmax(dim=-1)[:H*W].tolist()
    return [pred_flat[i*W:(i+1)*W] for i in range(H)]
```

## Task Complexity Routing (Save Parameters)

```python
import numpy as np

def analyze_task_complexity(task):
    """Route to smallest possible model for each task type."""
    train = task['train']
    
    # 1. Is it a pure color mapping? (no spatial reasoning needed)
    def is_color_mapping(pairs):
        for p in pairs:
            inp = np.array(p['input']).flatten()
            out = np.array(p['output']).flatten()
            if len(inp) != len(out):
                return False
            mapping = {}
            for i, o in zip(inp, out):
                if i in mapping and mapping[i] != o:
                    return False
                mapping[i] = o
        return True
    
    # 2. Same-size input/output?
    same_size = all(
        np.array(p['input']).shape == np.array(p['output']).shape
        for p in train
    )
    
    complexity = 'trivial' if is_color_mapping(train) else \
                 'spatial'  if same_size else 'resize'
    
    return {'complexity': complexity, 'same_size': same_size}

class ColorMappingModel(nn.Module):
    """~100 params. For pure color permutation tasks."""
    def __init__(self, n_colors=10):
        super().__init__()
        self.transform = nn.Linear(n_colors, n_colors)
    
    def forward(self, x_onehot):   # (B, HW, n_colors)
        return self.transform(x_onehot)

class MinimalConvModel(nn.Module):
    """~1K params. For same-size spatial tasks."""
    def __init__(self, n_colors=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_colors, 16, 3, padding=1), nn.GELU(),
            nn.Conv2d(16, n_colors, 1)
        )
    def forward(self, x):   # x: (B, n_colors, H, W) one-hot
        return self.net(x)

def select_model(task):
    info = analyze_task_complexity(task)
    if info['complexity'] == 'trivial': return ColorMappingModel()    # ~100 params
    if info['same_size']:               return MinimalConvModel()      # ~1K params
    return MDLSolver()                                                 # ~50K params
```

## ONNX Export Pipeline

```python
import torch, onnx, onnxruntime as ort, numpy as np

def export_to_onnx(model, puzzle, output_path):
    model.eval()
    
    # Create dummy input matching task's grid shape
    H = len(puzzle['test'][0]['input'])
    W = len(puzzle['test'][0]['input'][0])
    dummy = torch.zeros(1, H*W, dtype=torch.long)
    
    torch.onnx.export(
        model, dummy, output_path,
        opset_version = 17,
        input_names   = ['grid_flat'],
        output_names  = ['logits'],
        dynamic_axes  = {'grid_flat': {1: 'seq_len'}, 'logits': {1: 'seq_len'}}
    )
    
    # Validate
    onnx.checker.check_model(onnx.load(output_path))
    print(f"ONNX model valid: {output_path}")
    
    # Verify inference
    sess  = ort.InferenceSession(output_path)
    input_data = dummy.numpy()
    result     = sess.run(None, {'grid_flat': input_data})
    print(f"ONNX output shape: {result[0].shape}")
    return output_path

def get_model_size_kb(path):
    import os
    return os.path.getsize(path) / 1024
```

## Model Size Optimization

```python
# 1. Quantize to reduce size (INT8)
from onnxruntime.quantization import quantize_dynamic, QuantType
import subprocess

quantize_dynamic("model.onnx", "model_int8.onnx", weight_type=QuantType.QInt8)

# 2. Simplify graph
subprocess.run(["python", "-m", "onnxsim", "model.onnx", "model_simplified.onnx"])

# 3. Count parameters before export (minimize this!)
def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}  ({total/1e6:.3f}M)")
    return total

# 4. Avoid these (from TRM research — hurt sparse logic tasks):
# - SwiGLU MoE components (catastrophic overfitting on sparse data)
# - Large hidden dimensions (diminishing returns vs overfitting risk)
# - Deep attention stacks (use MLP-Mixer / linear layers when L ≤ D)
```

## Full Competition Pipeline

```python
import json, os

def solve_all_tasks(tasks_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    
    for task_file in os.listdir(tasks_dir):
        if not task_file.endswith('.json'):
            continue
        task_id = task_file[:-5]
        
        with open(os.path.join(tasks_dir, task_file)) as f:
            task = json.load(f)
        
        # 1. Route to smallest capable model
        model = select_model(task)
        
        # 2. Train on this puzzle only
        try:
            if isinstance(model, MDLSolver):
                prediction = compress_arc_solve(task)
            else:
                prediction = train_and_predict(model, task)
        except Exception as e:
            print(f"Task {task_id} failed: {e}")
            prediction = fallback_prediction(task)
        
        results[task_id] = prediction
        
        # 3. Export to ONNX
        onnx_path = os.path.join(output_dir, f"{task_id}.onnx")
        try:
            export_to_onnx(model, task, onnx_path)
        except Exception as e:
            print(f"ONNX export failed for {task_id}: {e}")
    
    return results
```

## Tips (From 2026 Research)
- CompressARC lesson: correctness from MDL compression + puzzle-specific training
- TRM lesson: 7M params + 16 recursive refinement steps > 70B autoregressive params
- Avoid MoE and excess capacity for sparse logic → use minimal linear/conv architectures
- For NeuroGolf scoring: correctness first, smallest model second — get all pairs right before shrinking
- ONNX dynamic axes: always use for variable grid sizes to avoid re-exporting per task
