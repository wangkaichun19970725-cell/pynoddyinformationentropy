"""Approximate entropy reduction from a proposed drillhole."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np

from utils_io import save_json, save_numpy


def gaussian_kernel(size: int, theta: float, l_par: float, l_perp: float, alpha: float) -> np.ndarray:
    radius = size // 2
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    x_rot = xx * cos_t + yy * sin_t
    y_rot = -xx * sin_t + yy * cos_t
    kernel = np.exp(-0.5 * ((x_rot / l_par) ** 2 + (y_rot / l_perp) ** 2))
    kernel /= kernel.sum()
    kernel *= alpha
    return kernel


def apply_kernel(entropy: np.ndarray, kernel: np.ndarray, x: int, y: int) -> Tuple[np.ndarray, np.ndarray]:
    reduction = np.zeros_like(entropy)
    radius = kernel.shape[0] // 2
    for z in range(entropy.shape[0]):
        x_min = max(0, x - radius)
        x_max = min(entropy.shape[2], x + radius + 1)
        y_min = max(0, y - radius)
        y_max = min(entropy.shape[1], y + radius + 1)

        kx_min = radius - (x - x_min)
        kx_max = radius + (x_max - x)
        ky_min = radius - (y - y_min)
        ky_max = radius + (y_max - y)

        window = kernel[ky_min:ky_max, kx_min:kx_max]
        local_entropy = entropy[z, y_min:y_max, x_min:x_max]
        reduction[z, y_min:y_max, x_min:x_max] += local_entropy * window

    updated = np.clip(entropy - reduction, a_min=0.0, a_max=None)
    return updated, reduction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entropy", type=Path, required=True)
    parser.add_argument("--kernel", type=Path, required=True)
    parser.add_argument("--x", type=int, required=True)
    parser.add_argument("--y", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kernel-size", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entropy = np.load(args.entropy)
    kernel_cfg = json.loads(Path(args.kernel).read_text())

    kernel = gaussian_kernel(
        size=args.kernel_size,
        theta=float(kernel_cfg["theta_mean_rad"]),
        l_par=float(kernel_cfg["length_parallel"]),
        l_perp=float(kernel_cfg["length_perpendicular"]),
        alpha=float(kernel_cfg["alpha"]),
    )

    updated, reduction = apply_kernel(entropy, kernel, args.x, args.y)

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    save_numpy(out_dir / "entropy_volume_after.npy", updated)
    save_numpy(out_dir / "reduction_volume.npy", reduction)
    save_json(
        out_dir / "influence_stats.json",
        {
            "original_entropy_sum": float(entropy.sum()),
            "updated_entropy_sum": float(updated.sum()),
            "entropy_reduction": float((entropy - updated).sum()),
            "x": int(args.x),
            "y": int(args.y),
        },
    )


if __name__ == "__main__":
    main()
