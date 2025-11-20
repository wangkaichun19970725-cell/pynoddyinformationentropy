"""Build Shannon entropy volumes from a PyNoddy ensemble."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils_io import load_block_from_g12, save_json, save_numpy


def compute_categorical_probabilities(volumes: np.ndarray) -> np.ndarray:
    """Compute categorical probabilities per voxel.

    Parameters
    ----------
    volumes: np.ndarray
        Array with shape (K, Z, Y, X) containing integer lithology codes for
        ``K`` ensemble members.
    """

    unique_codes = np.unique(volumes)
    prob_stack = []
    for code in unique_codes:
        mask = volumes == code
        prob = mask.sum(axis=0) / float(volumes.shape[0])
        prob_stack.append(prob)
    probabilities = np.stack(prob_stack, axis=0)
    return probabilities, unique_codes


def shannon_entropy(probabilities: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    probs = np.clip(probabilities, eps, 1.0)
    entropy = -np.sum(probs * np.log(probs), axis=0)
    return entropy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True, help="Directory containing ensemble members")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    return parser.parse_args()


def load_ensemble(case_dir: Path) -> np.ndarray:
    members: List[np.ndarray] = []
    subdirs = sorted([p for p in case_dir.iterdir() if p.is_dir()])
    if not subdirs:
        raise ValueError(f"No ensemble members found under {case_dir}")
    for sub in tqdm(subdirs, desc="loading ensemble"):
        g12 = sub / "case.g12"
        members.append(load_block_from_g12(g12))
    return np.stack(members, axis=0)


def export_quicklook(entropy: np.ndarray, out_dir: Path) -> None:
    projections = {
        "mip_x": entropy.max(axis=2),
        "mip_y": entropy.max(axis=1),
        "mip_z": entropy.max(axis=0),
    }
    for name, image in projections.items():
        plt.figure(figsize=(6, 6))
        plt.imshow(image, cmap="inferno")
        plt.colorbar(label="Entropy (nats)")
        plt.title(name)
        plt.tight_layout()
        plt.savefig(out_dir / f"{name}.png", dpi=200)
        plt.close()


def main() -> None:
    args = parse_args()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    ensemble = load_ensemble(args.case)
    probabilities, codes = compute_categorical_probabilities(ensemble)
    entropy = shannon_entropy(probabilities)

    stats = {
        "num_members": int(ensemble.shape[0]),
        "grid_shape": list(entropy.shape),
        "entropy_min": float(entropy.min()),
        "entropy_max": float(entropy.max()),
        "entropy_mean": float(entropy.mean()),
        "codes": [int(code) for code in codes],
    }

    save_numpy(out_dir / "entropy_volume.npy", entropy)
    save_json(out_dir / "entropy_volume_stats.json", stats)
    export_quicklook(entropy, out_dir)

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
