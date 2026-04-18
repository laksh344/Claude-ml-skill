# ARC-AGI / Grid Reasoning (ARC Prize 2026 — ARC-AGI-3)

## Overview & 2026 Research State

ARC-AGI-3 (launched March 25, 2026) fundamentally shifts from static visual grids to
**interactive reasoning** in novel turn-based game environments with no instructions or rules.
Agents must use a continuous **explore → perceive → plan → act** loop.

Metric: **RHAE (Relative Human Action Efficiency)** — squared penalty for inefficiency.

### Frontier LLM Performance on ARC-AGI-3 (March 2026)
| Model | Score | Human Baseline |
|-------|-------|---------------|
| Gemini 3.1 Pro | 0.37% | 100% |
| GPT-5.4 | 0.26% | 100% |
| Claude Opus 4.6 | 0.25% | 100% |
| Simple RL + graph exploration | **12.58%** | 100% |

**Critical finding**: RL + systematic exploration beats every frontier LLM by 30–50×.
Scale and text pretraining do NOT transfer to causal induction + spatial reasoning.

## RHAE Scoring

```
RHAE = (human_actions / ai_actions)²

Example:
  Human takes 10 actions → AI takes 10 → RHAE = 1.0 (perfect)
  Human takes 10 actions → AI takes 20 → RHAE = 0.25 (squared penalty)
  Human takes 10 actions → AI takes 100 → RHAE = 0.01 (severe penalty)
```

RHAE forces: systematic exploration, memory compression, vision-based salience tracking.
Brute-force random sampling is heavily penalized.

## Approach 1: RL + Graph-Based State Exploration (Best for ARC-AGI-3)

```python
import gymnasium as gym
import numpy as np
from collections import defaultdict

class ARCEnv(gym.Env):
    """Wrapper for ARC-AGI-3 interactive game environment."""
    def reset(self):
        self.state = initial_state()
        self.action_history = []
        self.belief_map = {}         # hypothesis about game mechanics
        return self.observe()
    
    def step(self, action):
        prev_state = self.state.copy()
        self.state = apply_action(self.state, action)
        reward     = self.compute_reward(prev_state, self.state, action)
        done       = self.is_terminal(self.state)
        self.action_history.append(action)
        return self.observe(), reward, done, {}
    
    def observe(self):
        # Parse visual frame into structured representation
        return {
            'grid':    self.state['grid'],       # current visual state
            'history': self.action_history[-5:], # recent context
            'beliefs': self.belief_map,          # inferred mechanics
        }

class SystematicExplorer:
    """Graph-based state-space explorer — far outperforms LLMs on ARC-AGI-3."""
    
    def __init__(self):
        self.visited    = set()
        self.graph      = defaultdict(list)       # state → [next_states]
        self.hypotheses = []                      # inferred rules
    
    def plan_action(self, obs):
        state_key = self.encode_state(obs['grid'])
        
        # BFS/DFS over unexplored states
        if state_key not in self.visited:
            self.visited.add(state_key)
            self.update_beliefs(obs)
        
        # Hypothesis-driven action selection
        best_action = self.score_actions(obs)
        return best_action
    
    def update_beliefs(self, obs):
        """Infer game mechanics from observations."""
        # Compare previous state → action → new state
        # Build a causal model: action X in context Y → effect Z
        if len(obs['history']) >= 2:
            cause  = obs['history'][-2]
            effect = obs['history'][-1]
            self.hypotheses.append({'cause': cause, 'effect': effect})
    
    def encode_state(self, grid):
        return tuple(map(tuple, grid))

# PPO agent for ARC-AGI-3
from stable_baselines3 import PPO

model = PPO(
    "MultiInputPolicy",    # handles dict observations
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    verbose=1
)
model.learn(total_timesteps=1_000_000)
```

## Approach 2: Weight-Space Refinement (TRM-Inspired — Best for ARC-AGI-1/2)

TRM (7M params, 2 layers) achieved 45% on ARC-AGI-1 by recursively refining in weight space.
This beats massive LLMs that accumulate errors via autoregressive generation.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyRecursiveModel(nn.Module):
    """
    TRM-inspired: separate latent reasoning + answer states, updated recursively.
    For fixed-context tasks where L <= D: replace attention with linear layers (MLP-Mixer style).
    Avoids quadratic attention cost. Executes up to N=16 refinement steps.
    """
    def __init__(self, d_model=64, n_colors=10, max_size=30):
        super().__init__()
        seq_len  = max_size * max_size
        
        # Use linear layers instead of attention when context fits (avoids quadratic cost)
        self.reasoning_state = nn.Parameter(torch.randn(1, seq_len, d_model))
        self.answer_state    = nn.Parameter(torch.randn(1, seq_len, d_model))
        
        # MLP-Mixer inspired layers
        self.token_mix  = nn.Sequential(nn.Linear(seq_len, seq_len), nn.GELU())
        self.channel_mix= nn.Sequential(nn.Linear(d_model, d_model*4), nn.GELU(),
                                         nn.Linear(d_model*4, d_model))
        self.color_embed= nn.Embedding(n_colors + 1, d_model)
        self.output_proj= nn.Linear(d_model, n_colors)
    
    def refine_step(self, x, reasoning):
        """Single recursive refinement step with BPTT."""
        # Mix tokens across sequence
        x_mixed = self.token_mix(x.transpose(1, 2)).transpose(1, 2)
        # Update reasoning state
        reasoning = reasoning + self.channel_mix(x_mixed + reasoning)
        return x_mixed, reasoning
    
    def forward(self, grid, n_steps=16):
        B, H, W = grid.shape
        x         = self.color_embed(grid.view(B, -1))      # (B, HW, d)
        reasoning = self.reasoning_state.expand(B, -1, -1)
        
        # Recursive refinement — up to N steps via BPTT
        for step in range(n_steps):
            x, reasoning = self.refine_step(x, reasoning)
        
        return self.output_proj(reasoning)   # (B, HW, n_colors)


def weight_space_refinement(puzzle, n_steps=16, lr=1e-3):
    """
    Train a tiny model ONLY on this puzzle's training pairs.
    CompressARC insight: overfit to compress the single puzzle → generalize to test.
    """
    model     = TinyRecursiveModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for step in range(500):   # inner loop
        total_loss = 0
        for pair in puzzle['train']:
            inp = torch.tensor(pair['input'],  dtype=torch.long).unsqueeze(0)
            tgt = torch.tensor(pair['output'], dtype=torch.long).flatten()
            H, W = len(pair['output']), len(pair['output'][0])
            
            logits = model(inp, n_steps=16)
            loss   = F.cross_entropy(logits.squeeze(0), tgt)
            total_loss += loss
        
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Verify: does model perfectly predict all training pairs?
        if total_loss.item() < 1e-4:
            break
    
    # Predict test grid
    test_inp = torch.tensor(puzzle['test'][0]['input'], dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        logits   = model(test_inp)
        H_t      = len(puzzle['test'][0]['input'])
        W_t      = len(puzzle['test'][0]['input'][0])
        pred_flat= logits.squeeze(0).argmax(dim=-1)[:H_t*W_t].tolist()
        return [pred_flat[i*W_t:(i+1)*W_t] for i in range(H_t)]
```

## Approach 3: Evolutionary Program Synthesis (SOAR-Inspired)

SOAR achieved 52% on ARC-AGI public test — highest open-source score — by alternating
exploration (candidate program generation) and hindsight learning (fine-tune on own traces).

```python
def soar_solve(puzzle, llm, n_generations=10, population_size=20):
    """
    SOAR: Self-Improving Operators for Automated Program Refinement.
    No DSL, no human solutions — purely LLM + self-improvement.
    """
    population     = []
    success_traces = []
    fail_traces    = []
    
    for gen in range(n_generations):
        # Phase 1: Exploration — generate and evaluate candidate programs
        new_candidates = [llm.generate_program(puzzle) for _ in range(population_size)]
        
        for prog in new_candidates:
            score = evaluate_program(prog, puzzle['train'])
            trace = {'program': prog, 'puzzle': puzzle, 'score': score}
            
            if score == 1.0:
                success_traces.append(trace)
                population.append(prog)
            else:
                fail_traces.append(trace)
                # Mutation: ask LLM to fix the error
                mutated = llm.mutate_program(prog, puzzle['train'])
                population.append(mutated)
        
        # Phase 2: Hindsight learning — fine-tune on own successful + failed traces
        if len(success_traces) > 5:
            llm.fine_tune(success_traces + fail_traces)   # converts both into training pairs
            success_traces, fail_traces = [], []           # reset after fine-tuning
        
        # Check if any program solves all training pairs
        for prog in population:
            if evaluate_program(prog, puzzle['train']) == 1.0:
                return run_program(prog, puzzle['test'][0]['input'])
    
    return None

def evaluate_program(program, train_pairs):
    """Run program on all training pairs, return fraction correct."""
    correct = 0
    for pair in train_pairs:
        try:
            result = run_program(program, pair['input'])
            if result == pair['output']:
                correct += 1
        except Exception:
            pass
    return correct / len(train_pairs)
```

## Approach 4: LLM-Based CoT (Baseline — ARC-AGI-1/2 Only)

```python
import anthropic

client = anthropic.Anthropic()

def solve_arc_with_llm(task):
    def grid_to_str(grid):
        return '\n'.join(' '.join(map(str, row)) for row in grid)
    
    examples = ""
    for i, pair in enumerate(task['train']):
        examples += f"\nExample {i+1}:\nInput:\n{grid_to_str(pair['input'])}\nOutput:\n{grid_to_str(pair['output'])}\n"
    
    prompt = f"""Study the input→output pattern, then apply the same transformation.
{examples}
Test Input:
{grid_to_str(task['test'][0]['input'])}

Step 1: What is the transformation rule?
Step 2: Apply it cell by cell.
Output ONLY the grid as space-separated numbers, one row per line."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    return parse_grid(response.content[0].text)

def parse_grid(text):
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()
             and all(c.isdigit() or c == ' ' for c in l.strip())]
    return [[int(x) for x in line.split()] for line in lines]
```

## Competition Strategy by ARC Version

| Version | Best Approach | Why |
|---------|--------------|-----|
| ARC-AGI-1 | TRM weight-space refinement OR LLM + CoT | Static grid tasks; pattern matching works |
| ARC-AGI-2 | SOAR evolutionary synthesis | More complex; needs program search |
| ARC-AGI-3 | RL + systematic graph exploration | Interactive; LLMs score near 0% |

## Tips
- For ARC-AGI-3: implement memory compression + hypothesis testing + visual salience
- Avoid injecting MoE / excess capacity for sparse logic tasks → catastrophic overfitting (observed in TRM experiments)
- CompressARC lesson: sometimes 76K params + correct objective > 70B params + text pretraining
- SOAR lesson: solution diversity plateaus for unsolved problems → use mutation + relabeling to break out
