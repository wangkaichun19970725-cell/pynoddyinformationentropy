#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
probe_pynoddy_fold_unconf.py

Standalone probe runner for Fold/Unconformity parameter syntax and behavior in pynoddy.
Does NOT import or read any generator script.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import pynoddy
    from pynoddy.history import NoddyHistory
    from pynoddy.output import NoddyOutput
except Exception:
    pynoddy = None
    NoddyHistory = None
    NoddyOutput = None


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def boundary_complexity_2d(a: np.ndarray) -> float:
    t = 0.0
    c = 0
    if a.shape[0] > 1:
        t += float(np.mean(a[1:, :] != a[:-1, :]))
        c += 1
    if a.shape[1] > 1:
        t += float(np.mean(a[:, 1:] != a[:, :-1]))
        c += 1
    return t / max(1, c)


def save_slices(block_xyz: np.ndarray, out_png: Path):
    if plt is None:
        return
    x, y, z = block_xyz.shape
    xm, ym, zm = x // 2, y // 2, z // 2
    xy = block_xyz[:, :, zm]
    xz = block_xyz[:, ym, :]
    yz = block_xyz[xm, :, :]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(xy.T, cmap="tab20", origin="lower", interpolation="nearest")
    axes[0].set_title(f"XY z={zm}")
    axes[1].imshow(xz.T, cmap="tab20", origin="lower", interpolation="nearest", aspect="auto")
    axes[1].set_title(f"XZ y={ym}")
    axes[2].imshow(yz.T, cmap="tab20", origin="lower", interpolation="nearest", aspect="auto")
    axes[2].set_title(f"YZ x={xm}")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def dump_history_excerpt(his_path: Path, out_txt: Path):
    txt = his_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    keep = []
    keys = ["Fold", "FAULT", "Fault", "UNCONFORM", "Unconform", "STRAT", "Layer", "Event"]
    for i, line in enumerate(txt):
        if any(k in line for k in keys):
            lo = max(0, i - 2)
            hi = min(len(txt), i + 3)
            keep.extend(txt[lo:hi])
            keep.append("---")
    if len(keep) == 0:
        keep = txt[:200]
    out_txt.write_text("\n".join(keep), encoding="utf-8")


def make_base_history(seed: int, nx: int, ny: int, nz: int, cube_size: float) -> NoddyHistory:
    rng = np.random.default_rng(seed)
    h = NoddyHistory()

    try:
        h.set_origin(0.0, 0.0, 0.0)
    except Exception:
        pass
    try:
        h.set_extent(float(nx) * cube_size, float(ny) * cube_size, float(nz) * cube_size)
    except Exception:
        pass
    try:
        h.set_cube_size(float(cube_size))
    except Exception:
        pass

    n_layers = 8
    thicknesses = rng.integers(120, 260, size=n_layers).astype(int).tolist()
    layer_names = [f"L{i+1}" for i in range(n_layers)]
    h.add_event(
        "stratigraphy",
        {
            "name": "STRAT_1",
            "num_layers": int(n_layers),
            "layer_names": layer_names,
            "layer_thickness": thicknesses,
            "layer_thicknesses": thicknesses,
        },
    )
    return h


def run_model(h: NoddyHistory, noddy_exe: str, out_prefix: Path) -> np.ndarray:
    his_path = out_prefix.with_suffix(".his")
    h.write_history(str(his_path))
    pynoddy.compute_model(str(his_path), str(out_prefix), noddy_path=str(noddy_exe))
    out = NoddyOutput(str(out_prefix))
    if not isinstance(out.block, np.ndarray):
        raise RuntimeError("NoddyOutput.block is not ndarray")
    return out.block


def fold_trials(seed: int, n_trials: int) -> List[Dict[str, float]]:
    rng = np.random.default_rng(seed)
    trends = np.linspace(10.0, 170.0, max(6, n_trials))
    plunges = np.linspace(5.0, 35.0, max(4, min(8, n_trials)))
    out = []
    for i in range(n_trials):
        tr = float(trends[i % len(trends)])
        pl = float(plunges[(i * 3) % len(plunges)])
        amp = float(rng.uniform(100.0, 520.0))
        wl = float(rng.uniform(900.0, 3000.0))
        px = float(rng.uniform(-1200.0, 1200.0))
        py = float(rng.uniform(-1200.0, 1200.0))
        pz = float(rng.uniform(-1800.0, -150.0))
        out.append({
            "axis_dir": tr,
            "plunge": pl,
            "amplitude": amp,
            "wavelength": wl,
            "pos_x": px,
            "pos_y": py,
            "pos_z": pz,
        })
    return out


def unconf_trials(seed: int, n_trials: int, cube_size: float, nz: int) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed + 991)
    out = []
    max_total = max(2 * cube_size, 0.22 * (nz * cube_size))
    for i in range(n_trials):
        n_uc = int(rng.integers(1, 4))
        total = float(rng.uniform(2 * cube_size, max_total))
        t = [max(cube_size, total / n_uc)] * n_uc
        t = [int(round(v)) for v in t]
        z_ref = float(rng.uniform(-0.65 * nz * cube_size, -0.30 * nz * cube_size))
        dip = float(rng.uniform(3.0, 14.0))
        dip_dir = float(rng.uniform(0.0, 360.0))
        px = float(rng.uniform(-900.0, 900.0))
        py = float(rng.uniform(-900.0, 900.0))
        out.append({
            "num_layers": n_uc,
            "thicknesses": t,
            "z_ref": z_ref,
            "dip": dip,
            "dip_dir": dip_dir,
            "pos_x": px,
            "pos_y": py,
        })
    return out


def unconformity_thickness_proxy(block_xyz: np.ndarray, ignore_label: int = 0) -> float:
    # Estimate how dominant top-package is in upper Z part of crop
    z = block_xyz.shape[2]
    k = max(1, z // 5)
    top = block_xyz[:, :, z - k :].reshape(-1)
    top = top[top != int(ignore_label)]
    if top.size == 0:
        return 0.0
    _, c = np.unique(top, return_counts=True)
    return float(np.max(c)) / float(max(1, top.size))


def analyze_fold_visibility(block_xyz: np.ndarray) -> Dict[str, float]:
    x, y, z = block_xyz.shape
    xm, ym, zm = x // 2, y // 2, z // 2
    xy = block_xyz[:, :, zm]
    xz = block_xyz[:, ym, :]
    yz = block_xyz[xm, :, :]
    cxy = boundary_complexity_2d(xy)
    cxz = boundary_complexity_2d(xz)
    cyz = boundary_complexity_2d(yz)
    return {
        "complex_xy": cxy,
        "complex_xz": cxz,
        "complex_yz": cyz,
        "yz_over_xz": float(cyz / max(cxz, 1e-8)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noddy_exe", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="probe_out")
    ap.add_argument("--nx", type=int, default=80)
    ap.add_argument("--ny", type=int, default=80)
    ap.add_argument("--nz", type=int, default=80)
    ap.add_argument("--cube_size", type=float, default=50.0)
    ap.add_argument("--n_trials", type=int, default=12)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    if pynoddy is None or NoddyHistory is None or NoddyOutput is None:
        raise RuntimeError("pynoddy is not available in this environment. Activate your pynoddy env first.")

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "fold")
    ensure_dir(out_dir / "unconf")

    fold_rows: List[Dict[str, Any]] = []
    unconf_rows: List[Dict[str, Any]] = []

    # Fold probes: one fold at a time
    for i, fpar in enumerate(fold_trials(args.seed, args.n_trials)):
        h = make_base_history(args.seed + i, args.nx, args.ny, args.nz, args.cube_size)
        h.add_event(
            "fold",
            {
                "name": f"FOLD_PROBE_{i+1}",
                "pos": [fpar["pos_x"], fpar["pos_y"], fpar["pos_z"]],
                "amplitude": fpar["amplitude"],
                "wavelength": fpar["wavelength"],
                "axis_dir": fpar["axis_dir"],
                "plunge": fpar["plunge"],
            },
        )

        pref = out_dir / "fold" / f"fold_{i:03d}"
        block = run_model(h, args.noddy_exe, pref)
        save_slices(block, out_dir / "fold" / f"fold_{i:03d}_slices.png")
        dump_history_excerpt(pref.with_suffix(".his"), out_dir / "fold" / f"fold_{i:03d}_history_excerpt.txt")

        vis = analyze_fold_visibility(block)
        row = {"trial": i, **fpar, **vis}
        fold_rows.append(row)

        print(f"[FOLD] trial={i} params={fpar} vis={vis}")

    # Unconformity probes: one fold + one unconformity
    for i, upar in enumerate(unconf_trials(args.seed, args.n_trials, args.cube_size, args.nz)):
        h = make_base_history(args.seed + 1000 + i, args.nx, args.ny, args.nz, args.cube_size)
        h.add_event(
            "fold",
            {
                "name": "FOLD_BASE",
                "pos": [0.0, 0.0, -800.0],
                "amplitude": 260.0,
                "wavelength": 1700.0,
                "axis_dir": 35.0,
                "plunge": 20.0,
            },
        )
        h.add_event(
            "unconformity",
            {
                "name": f"UNCONF_PROBE_{i+1}",
                "pos": [upar["pos_x"], upar["pos_y"], upar["z_ref"]],
                "dip_dir": upar["dip_dir"],
                "dip_direction": upar["dip_dir"],
                "dip": upar["dip"],
                "num_layers": int(upar["num_layers"]),
                "layer_names": [f"UC{k+1}" for k in range(int(upar["num_layers"]))],
                "layer_thickness": upar["thicknesses"],
                "layer_thicknesses": upar["thicknesses"],
            },
        )

        pref = out_dir / "unconf" / f"unconf_{i:03d}"
        block = run_model(h, args.noddy_exe, pref)
        save_slices(block, out_dir / "unconf" / f"unconf_{i:03d}_slices.png")
        dump_history_excerpt(pref.with_suffix(".his"), out_dir / "unconf" / f"unconf_{i:03d}_history_excerpt.txt")

        proxy = unconformity_thickness_proxy(block, ignore_label=0)
        vis = analyze_fold_visibility(block)
        row = {"trial": i, **upar, "unconf_top_proxy": proxy, **vis}
        unconf_rows.append(row)

        print(f"[UNCONF] trial={i} params={upar} unconf_top_proxy={proxy:.4f} vis={vis}")

    # Save CSV + JSON
    with open(out_dir / "fold_trials.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fold_rows[0].keys()))
        w.writeheader()
        for r in fold_rows:
            w.writerow(r)

    with open(out_dir / "unconf_trials.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(unconf_rows[0].keys()))
        w.writeheader()
        for r in unconf_rows:
            w.writerow(r)

    with open(out_dir / "probe_summary.json", "w", encoding="utf-8") as f:
        json.dump({"fold_trials": fold_rows, "unconf_trials": unconf_rows}, f, indent=2)

    # concise markdown findings
    yz_over_xz = np.array([r["yz_over_xz"] for r in fold_rows], dtype=np.float64)
    best_idx = int(np.argmax(yz_over_xz))
    worst_idx = int(np.argmin(yz_over_xz))
    thick = np.array([r["unconf_top_proxy"] for r in unconf_rows], dtype=np.float64)
    th_idx = int(np.argmax(thick))

    md = []
    md.append("# probe_results.md\n\n")
    md.append("## Fold parameter mapping (empirical)\n")
    md.append(f"- Trials: {len(fold_rows)}\n")
    md.append(f"- Best YZ visibility trial: {best_idx}, yz_over_xz={yz_over_xz[best_idx]:.4f}\n")
    md.append(f"  - Params: {fold_rows[best_idx]}\n")
    md.append(f"- Worst YZ visibility trial: {worst_idx}, yz_over_xz={yz_over_xz[worst_idx]:.4f}\n")
    md.append(f"  - Params: {fold_rows[worst_idx]}\n")
    md.append("- Practical read: higher yz_over_xz means YZ shows fold clearer relative to XZ.\n\n")

    md.append("## Unconformity parameter mapping (empirical)\n")
    md.append(f"- Thickest top-package proxy trial: {th_idx}, proxy={thick[th_idx]:.4f}\n")
    md.append(f"  - Params: {unconf_rows[th_idx]}\n")
    md.append("- Lower unconf_top_proxy generally indicates thinner/non-dominant upper package.\n\n")

    md.append("## Suggested policy ranges (starting point)\n")
    md.append("- Folds: prioritize axis_dir away from cardinal windows and mix near-X, near-Y, and oblique sets.\n")
    md.append("- Unconformity: keep total thickness <= ~20-25% crop height and place z_ref within central folded-depth band.\n")
    md.append("- Use the generated history excerpts to verify exact field names accepted by your local pynoddy build.\n")

    (out_dir / "probe_results.md").write_text("".join(md), encoding="utf-8")

    print(f"[DONE] outputs in: {out_dir}")


if __name__ == "__main__":
    main()
