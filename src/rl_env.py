"""Entropy reduction reinforcement learning environment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from apply_influence import apply_kernel, gaussian_kernel


@dataclass
class EnvConfig:
    kernel_theta: float
    kernel_l_parallel: float
    kernel_l_perp: float
    kernel_alpha: float
    kernel_size: int = 15
    min_separation: int = 3
    max_steps: int = 5


class EntropyDrillingEnv:
    """Stateful environment tracking entropy reductions."""

    def __init__(self, entropy_volume: np.ndarray, config: EnvConfig):
        if entropy_volume.ndim != 3:
            raise ValueError("Entropy volume must be 3-D (Z, Y, X)")
        self.initial_entropy = entropy_volume.astype(np.float32)
        self.entropy = self.initial_entropy.copy()
        self.config = config
        self.kernel = gaussian_kernel(
            size=config.kernel_size,
            theta=config.kernel_theta,
            l_par=config.kernel_l_parallel,
            l_perp=config.kernel_l_perp,
            alpha=config.kernel_alpha,
        )
        self.visited: List[Tuple[int, int]] = []
        self.steps = 0
        self.rng = np.random.default_rng(0)
        self._update_available_actions()

    def seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def _update_available_actions(self) -> None:
        ny, nx = self.entropy.shape[1], self.entropy.shape[2]
        self.mask = np.ones((ny, nx), dtype=bool)
        for x, y in self.visited:
            self._mask_radius(x, y)

    def _mask_radius(self, x: int, y: int) -> None:
        r = self.config.min_separation
        ny, nx = self.mask.shape
        x_min = max(0, x - r)
        x_max = min(nx, x + r + 1)
        y_min = max(0, y - r)
        y_max = min(ny, y + r + 1)
        self.mask[y_min:y_max, x_min:x_max] = False

    def reset(self) -> np.ndarray:
        self.entropy = self.initial_entropy.copy()
        self.visited.clear()
        self.steps = 0
        self._update_available_actions()
        return self._current_state()

    def _current_state(self) -> np.ndarray:
        state = self.entropy.sum(axis=0)
        state = state.astype(np.float32)
        return state

    def action_space(self) -> int:
        ny, nx = self.mask.shape
        return ny * nx

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        ny, nx = self.mask.shape
        y = action // nx
        x = action % nx
        if not self.mask[y, x]:
            # Invalid action, heavy penalty
            return self._current_state(), -1.0, True, {"invalid": 1.0}

        updated, reduction = apply_kernel(self.entropy, self.kernel, x, y)
        reward = float((self.entropy - updated).sum())
        self.entropy = updated
        self.visited.append((x, y))
        self.steps += 1
        self._mask_radius(x, y)
        done = self.steps >= self.config.max_steps or not self.mask.any()
        info = {
            "reward": reward,
            "entropy_remaining": float(self.entropy.sum()),
            "x": float(x),
            "y": float(y),
        }
        return self._current_state(), reward, done, info

    def available_actions(self) -> np.ndarray:
        return self.mask.reshape(-1)

    def sample_valid_action(self) -> int:
        valid = np.where(self.available_actions())[0]
        if len(valid) == 0:
            raise RuntimeError("No valid actions remaining")
        return int(self.rng.choice(valid))

    def clone(self) -> "EntropyDrillingEnv":
        clone = EntropyDrillingEnv(self.entropy.copy(), self.config)
        clone.initial_entropy = self.initial_entropy.copy()
        clone.entropy = self.entropy.copy()
        clone.visited = self.visited.copy()
        clone.steps = self.steps
        clone.mask = self.mask.copy()
        clone.kernel = self.kernel.copy()
        clone.rng = np.random.default_rng()
        clone.rng.bit_generator.state = self.rng.bit_generator.state
        return clone


def config_from_json(payload: Dict[str, float], max_steps: int, kernel_size: int, min_separation: int) -> EnvConfig:
    return EnvConfig(
        kernel_theta=float(payload["theta_mean_rad"]),
        kernel_l_parallel=float(payload["length_parallel"]),
        kernel_l_perp=float(payload["length_perpendicular"]),
        kernel_alpha=float(payload["alpha"]),
        kernel_size=kernel_size,
        min_separation=min_separation,
        max_steps=max_steps,
    )
