"""Train a DQN agent to reduce entropy."""
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Deque, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from rl_env import EntropyDrillingEnv, EnvConfig, config_from_json


class QNetwork(nn.Module):
    def __init__(self, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        dummy = torch.zeros(1, 1, 50, 50)
        feat_dim = self.net(dummy).shape[1]
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.net(x)
        return self.head(features)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*(self.buffer[idx] for idx in indices))
        return (
            np.stack(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


def epsilon_by_episode(start: float, end: float, decay: int, episode: int) -> float:
    return end + (start - end) * np.exp(-1.0 * episode / decay)


def load_entropy_volumes(paths: List[Path]) -> List[np.ndarray]:
    volumes = []
    for path in paths:
        volumes.append(np.load(path))
    return volumes


def select_action(network: QNetwork, state: np.ndarray, mask: np.ndarray, epsilon: float, device: torch.device) -> int:
    if np.random.rand() < epsilon:
        valid = np.where(mask)[0]
        if len(valid) == 0:
            return 0
        return int(np.random.choice(valid))
    with torch.no_grad():
        tensor = torch.from_numpy(state[None, None, :, :]).float().to(device)
        q_values = network(tensor)[0].detach().cpu().numpy()
    q_values[~mask] = -1e9
    return int(np.argmax(q_values))


def train_episode(env: EntropyDrillingEnv, network: QNetwork, target: QNetwork, buffer: ReplayBuffer, optimizer, device, gamma: float, batch_size: int, epsilon: float):
    state = env.reset()
    mask = env.available_actions()
    total_reward = 0.0
    losses = []
    done = False
    while not done:
        action = select_action(network, state, mask, epsilon, device)
        next_state, reward, done, info = env.step(action)
        buffer.push(state, action, reward, next_state, done)
        state = next_state
        mask = env.available_actions()
        total_reward += reward

        if len(buffer) >= batch_size:
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)
            states_t = torch.from_numpy(states[:, None, :, :]).float().to(device)
            actions_t = torch.from_numpy(actions).long().to(device)
            rewards_t = torch.from_numpy(rewards).to(device)
            next_states_t = torch.from_numpy(next_states[:, None, :, :]).float().to(device)
            dones_t = torch.from_numpy(dones).to(device)

            q_values = network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q_values = target(next_states_t).max(1)[0]
                targets = rewards_t + gamma * (1.0 - dones_t) * next_q_values
            loss = nn.functional.mse_loss(q_values, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    return total_reward, float(np.mean(losses) if losses else 0.0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entropy", type=Path, nargs="+", required=True, help="Paths to entropy_volume.npy files for training")
    parser.add_argument("--kernel-config", type=Path, required=True, help="aniso_kernels.json")
    parser.add_argument("--save-dir", type=Path, default=Path("models"))
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--buffer-size", type=int, default=5000)
    parser.add_argument("--target-update", type=int, default=10)
    parser.add_argument("--epsilon-start", type=float, default=0.9)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--kernel-size", type=int, default=15)
    parser.add_argument("--min-separation", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    kernel_cfg = json.loads(args.kernel_config.read_text())
    env_config = config_from_json(kernel_cfg, args.max_steps, args.kernel_size, args.min_separation)

    entropy_volumes = load_entropy_volumes(args.entropy)

    action_dim = 50 * 50
    policy_net = QNetwork(action_dim)
    target_net = QNetwork(action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net.to(device)
    target_net.to(device)

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
    buffer = ReplayBuffer(args.buffer_size)

    rewards: List[float] = []
    losses: List[float] = []

    save_dir = args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    for episode in tqdm(range(args.epochs), desc="training"):
        entropy = entropy_volumes[episode % len(entropy_volumes)]
        env = EntropyDrillingEnv(entropy, env_config)
        env.seed(args.seed + episode)
        epsilon = epsilon_by_episode(args.epsilon_start, args.epsilon_end, args.epsilon_decay, episode)
        reward, loss = train_episode(env, policy_net, target_net, buffer, optimizer, device, args.gamma, args.batch_size, epsilon)
        rewards.append(reward)
        losses.append(loss)
        if episode % args.target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

    torch.save(policy_net.state_dict(), save_dir / "model_final.pt")
    history = {
        "rewards": rewards,
        "losses": losses,
    }
    (save_dir / "plan_rl_stats.json").write_text(json.dumps(history, indent=2))

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 4))
        plt.plot(rewards)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.tight_layout()
        plt.savefig(save_dir / "rl_training_curve.png", dpi=200)
        plt.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
