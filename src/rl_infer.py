"""Inference utilities for the trained RL drilling agent."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from rl_env import EntropyDrillingEnv, config_from_json
from rl_train import QNetwork


def load_policy(model_path: Path, action_dim: int, device: torch.device) -> QNetwork:
    policy = QNetwork(action_dim)
    state_dict = torch.load(model_path, map_location=device)
    policy.load_state_dict(state_dict)
    policy.to(device)
    policy.eval()
    return policy


def greedy_policy(policy: QNetwork, state: np.ndarray, mask: np.ndarray) -> int:
    device = next(policy.parameters()).device
    tensor = torch.from_numpy(state[None, None, :, :]).float().to(device)
    with torch.no_grad():
        q_values = policy(tensor)[0].detach().cpu().numpy()
    q_values[~mask] = -1e9
    return int(np.argmax(q_values))


def run_episode(entropy: np.ndarray, kernel_cfg: Dict[str, float], model_path: Path, max_steps: int, kernel_size: int, min_separation: int, seed: int = 0) -> Tuple[List[Tuple[int, int]], List[float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = load_policy(model_path, 50 * 50, device)
    env_config = config_from_json(kernel_cfg, max_steps, kernel_size, min_separation)
    env = EntropyDrillingEnv(entropy, env_config)
    env.seed(seed)
    state = env.reset()
    mask = env.available_actions()
    picks: List[Tuple[int, int]] = []
    reductions: List[float] = []
    done = False
    while not done:
        action = greedy_policy(policy, state, mask)
        next_state, reward, done, info = env.step(action)
        picks.append((int(info["x"]), int(info["y"])))
        reductions.append(float(reward))
        state = next_state
        mask = env.available_actions()
        if len(picks) >= max_steps:
            break
    return picks, reductions
