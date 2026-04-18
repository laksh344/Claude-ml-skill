# RL Agent / Bot Competitions (e.g., Orbit Wars)

## Overview

These competitions use a **skill-rating ladder** (Gaussian N(μ, σ²) Elo-like system).
- Submissions play episodes against similar-rated bots
- μ (mean) = estimated skill; σ (uncertainty) decreases over time
- Only wins/losses matter for rating — not score margin
- Only your **latest 2 submissions** are tracked for final ranking

## Strategy

### Core Principle
Build a **robust, general agent** — not one that exploits a single strategy.
The ladder has diverse opponents; you need to beat many playstyles.

### Agent Architecture Options

```python
# Option 1: Rule-based heuristic (fast baseline, interpretable)
def agent(obs, config):
    # Parse observation
    # Apply hand-crafted rules
    return action

# Option 2: MCTS (strong for turn-based/game-tree problems)
class MCTSAgent:
    def __init__(self, simulations=100):
        self.simulations = simulations
    
    def search(self, state):
        root = Node(state)
        for _ in range(self.simulations):
            node = self.select(root)
            reward = self.simulate(node)
            self.backprop(node, reward)
        return root.best_action()

# Option 3: Deep RL (PPO/DQN for complex state spaces)
import torch
import torch.nn as nn
from stable_baselines3 import PPO

class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),    nn.ReLU(),
            nn.Linear(256, act_dim)
        )
    def forward(self, x):
        return self.net(x)
```

### Self-Play Training Loop

```python
# For deep RL bots: train via self-play to improve progressively
from collections import deque
import random

class SelfPlayTrainer:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.opponent_pool = deque(maxlen=20)  # keep past versions
    
    def train_episode(self):
        obs = self.env.reset()
        done = False
        experiences = []
        
        # Pick opponent from pool (80% random past, 20% latest)
        opponent = (random.choice(self.opponent_pool) 
                    if self.opponent_pool and random.random() < 0.8 
                    else self.agent)
        
        while not done:
            action = self.agent.act(obs)
            next_obs, reward, done, info = self.env.step(action)
            experiences.append((obs, action, reward, next_obs, done))
            obs = next_obs
        
        self.agent.update(experiences)
        
        # Periodically save snapshot to opponent pool
        if episode % 100 == 0:
            self.opponent_pool.append(copy.deepcopy(self.agent))
```

### PPO with Custom Environment

```python
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

def make_env(env_id, seed):
    def _init():
        env = gym.make(env_id)
        env.reset(seed=seed)
        return env
    return _init

# Parallel environments for faster training
n_envs = 8
envs = SubprocVecEnv([make_env("YourEnv-v0", i) for i in range(n_envs)])

model = PPO(
    "MlpPolicy", envs,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1,
    tensorboard_log="./tb_logs/"
)
model.learn(total_timesteps=10_000_000)
model.save("ppo_agent")
```

### Submission Template (Kaggle Simulation API)

```python
import numpy as np

def agent(obs, config):
    """
    obs: observation from environment
    config: competition configuration dict
    Returns: action (int or list)
    """
    # Always handle edge cases first
    if obs is None:
        return default_action(config)
    
    # Your logic here
    action = compute_action(obs, config)
    return action

# Validate locally before submitting
# Use the competition's provided local simulator
```

### Tips for Skill-Rating Competitions
- Submit early; more episodes = more stable rating
- A new submission starts at μ₀=600; expect volatile early results
- Analyze agent logs when submissions are marked Error
- Diversify strategies: a bot that does one thing predictably is exploitable
- Test against multiple local opponents before submitting
