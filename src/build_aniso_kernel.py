"""Estimate anisotropic influence kernels from PyNoddy models."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils_io import load_block_from_g12, save_json


def slice_orientations(volume: np.ndarray) -> List[float]:
    orientations: List[float] = []
    for z in range(volume.shape[0]):
        slice_ = volume[z].astype(float)
        if np.all(slice_ == slice_[0, 0]):
            continue
        gy, gx = np.gradient(slice_)
        mean_gx = np.mean(gx)
        mean_gy = np.mean(gy)
        angle = math.atan2(mean_gy, mean_gx)
        orientations.append(angle)
    return orientations


def summarise_orientations(angles: List[float]) -> float:
    if not angles:
        return 0.0
    vec = np.array([np.cos(angles), np.sin(angles)])
    mean_vec = vec.mean(axis=1)
    return math.atan2(mean_vec[1], mean_vec[0])


def build_kernel_config(angles: List[float]) -> dict:
    if not angles:
        angles = [0.0]
    mean_angle = summarise_orientations(angles)
    std_angle = float(np.std(angles))
    config = {
        "theta_mean_rad": float(mean_angle),
        "theta_mean_deg": float(np.degrees(mean_angle)),
        "theta_std_deg": float(np.degrees(std_angle)),
        "length_parallel": 7.0,
        "length_perpendicular": 3.0,
        "alpha": 0.85,
        "slice_orientations_deg": [float(np.degrees(a)) for a in angles],
    }
    return config


def plot_rose(angles: List[float], out_path: Path) -> None:
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, projection="polar")
    bins = np.linspace(0, 2 * np.pi, 24)
    hist, _ = np.histogram([a % (2 * np.pi) for a in angles], bins=bins)
    widths = np.diff(bins)
    ax.bar(bins[:-1], hist, width=widths, bottom=0.0, color="teal", alpha=0.7)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title("Dominant orientation rose plot")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g12", type=Path, nargs="+", required=True, help="One or more .g12 volumes")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    all_angles: List[float] = []
    for g12_path in tqdm(args.g12, desc="analysing models"):
        volume = load_block_from_g12(g12_path)
        all_angles.extend(slice_orientations(volume))

    kernel_config = build_kernel_config(all_angles)
    save_json(out_dir / "aniso_kernels.json", kernel_config)
    plot_rose(all_angles, out_dir / "orientation_rose.png")

    print("Estimated mean strike angle (deg):", kernel_config["theta_mean_deg"])


if __name__ == "__main__":
    main()
