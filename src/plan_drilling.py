"""Plan borehole drilling locations that minimise entropy."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from build_entropy_from_ensemble import compute_categorical_probabilities, shannon_entropy
from rl_env import EntropyDrillingEnv, config_from_json
from rl_infer import run_episode
from utils_io import load_block_from_g12


def load_entropy_volume(ensemble_dir: Path) -> np.ndarray:
    members: List[np.ndarray] = []
    for path in sorted(ensemble_dir.glob("**/case.g12")):
        members.append(load_block_from_g12(path))
    if not members:
        raise ValueError(f"No .g12 volumes found under {ensemble_dir}")
    stack = np.stack(members, axis=0)
    probabilities, _ = compute_categorical_probabilities(stack)
    entropy = shannon_entropy(probabilities)
    return entropy


def beam_search(entropy: np.ndarray, kernel_cfg: Dict[str, float], max_steps: int, kernel_size: int, min_separation: int, beam_width: int) -> Tuple[List[Tuple[int, int]], List[float]]:
    env_config = config_from_json(kernel_cfg, max_steps, kernel_size, min_separation)
    initial_env = EntropyDrillingEnv(entropy, env_config)
    initial_state = initial_env.reset()
    frontier = [
        {
            "env": initial_env,
            "picks": [],
            "reduction": [],
            "state": initial_state,
        }
    ]
    best_sequence = []
    best_reward = -np.inf

    for step in range(max_steps):
        new_frontier = []
        for node in frontier:
            env = node["env"]
            state = node["state"]
            mask = env.available_actions()
            valid_actions = np.where(mask)[0]
            if len(valid_actions) == 0:
                total_reward = sum(node["reduction"])
                if total_reward > best_reward:
                    best_reward = total_reward
                    best_sequence = node["picks"]
                continue
            scores = state.reshape(-1)
            ranked = valid_actions[np.argsort(scores[valid_actions])[::-1]]
            for action in ranked[:beam_width]:
                # Clone environment by creating a new instance with the same state
                child_env = env.clone()
                next_state, reward, done, info = child_env.step(int(action))
                picks = node["picks"] + [(int(info["x"]), int(info["y"]))]
                reduction = node["reduction"] + [float(reward)]
                new_frontier.append({
                    "env": child_env,
                    "picks": picks,
                    "reduction": reduction,
                    "state": next_state,
                })
                if done or len(picks) >= max_steps:
                    total_reward = sum(reduction)
                    if total_reward > best_reward:
                        best_reward = total_reward
                        best_sequence = picks
        new_frontier.sort(key=lambda n: sum(n["reduction"]), reverse=True)
        frontier = new_frontier[:beam_width]
        if not frontier:
            break

    if not best_sequence and frontier:
        best_node = max(frontier, key=lambda n: sum(n["reduction"]))
        best_sequence = best_node["picks"]
    best_reductions = []
    env = EntropyDrillingEnv(entropy, env_config)
    env.reset()
    ny, nx = env.mask.shape
    for x, y in best_sequence:
        action = y * nx + x
        _, reward, _, _ = env.step(action)
        best_reductions.append(reward)
    return best_sequence, best_reductions


def overlay(entropy: np.ndarray, picks: List[Tuple[int, int]], out_path: Path) -> None:
    image = entropy.sum(axis=0)
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap="inferno")
    if picks:
        xs, ys = zip(*picks)
        plt.scatter(xs, ys, c="cyan", s=40, edgecolor="black")
        for idx, (x, y) in enumerate(picks):
            plt.text(x + 0.5, y + 0.5, str(idx + 1), color="white", ha="left", va="bottom")
    plt.title("Planned drillholes")
    plt.colorbar(label="Entropy (slice sum)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_plan(picks: List[Tuple[int, int]], reductions: List[float], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "picks.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["order", "x", "y", "delta_entropy"])
        for idx, ((x, y), reward) in enumerate(zip(picks, reductions), start=1):
            writer.writerow([idx, x, y, reward])


def min_separation_ok(picks: List[Tuple[int, int]], min_separation: int) -> bool:
    if min_separation <= 0:
        return True
    for i in range(len(picks)):
        x1, y1 = picks[i]
        for j in range(i + 1, len(picks)):
            x2, y2 = picks[j]
            if abs(x1 - x2) <= min_separation and abs(y1 - y2) <= min_separation:
                return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble", type=Path, required=True, help="Directory containing ensemble case folders")
    parser.add_argument("--kernel-config", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=None, help="Trained RL model (model_final.pt)")
    parser.add_argument("--n", type=int, required=True, help="Number of drillholes to plan")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kernel-size", type=int, default=15)
    parser.add_argument("--min-separation", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.time()
    entropy = load_entropy_volume(args.ensemble)
    kernel_cfg = json.loads(Path(args.kernel_config).read_text())

    if args.model and Path(args.model).exists():
        picks, reductions = run_episode(
            entropy,
            kernel_cfg,
            Path(args.model),
            args.n,
            args.kernel_size,
            args.min_separation,
            seed=args.seed,
        )
    else:
        picks, reductions = beam_search(
            entropy,
            kernel_cfg,
            args.n,
            args.kernel_size,
            args.min_separation,
            args.beam_width,
        )

    overlay(entropy, picks, args.out / "picks_overlay.png")
    save_plan(picks, reductions, args.out)

    stats = {
        "total_reduction": float(sum(reductions)),
        "per_step": reductions,
        "num_picks": len(picks),
        "min_separation_satisfied": min_separation_ok(picks, args.min_separation),
        "runtime_seconds": float(time.time() - start),
    }
    (args.out / "plan_stats.json").write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
