#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
diag_geology_v1.py

Standalone diagnostics for generated geology volumes.
It does NOT read or import any generator source code.

Example usage:
  python diag_geology_v1.py --npy npy/models.npy --ignore_label 0 --n_models 50 --report_dir diag_report_v1 --save_png 1 --save_stats 1

For v10-style gate replay:
  python diag_geology_v1.py --npy npy/models.npy --ignore_label 0 --dominant_gate 0.40 --min_strat_in_crop 6 --report_dir diag_v10_gate
"""

from __future__ import annotations

import os
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def parse_indices(s: Optional[str]) -> Optional[List[int]]:
    if s is None or str(s).strip() == "":
        return None
    out = []
    for x in str(s).split(","):
        x = x.strip()
        if x:
            out.append(int(x))
    return out


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_volumes(path: str) -> np.ndarray:
    arr = np.load(path, mmap_mode="r")
    if arr.ndim == 3:
        arr = arr[None, ...]
    if arr.ndim != 4:
        raise ValueError(f"Expected [N,Z,Y,X] or [Z,Y,X], got shape={arr.shape}")
    return arr


def select_model_indices(n_total: int, n_models: int, model_indices: Optional[List[int]]) -> List[int]:
    if model_indices is not None:
        idx = [i for i in model_indices if 0 <= i < n_total]
        return idx
    if n_models is None or int(n_models) < 0:
        return list(range(n_total))
    return list(range(min(int(n_models), n_total)))


def boundary_map_2d(slc: np.ndarray) -> np.ndarray:
    b = np.zeros(slc.shape, dtype=np.float32)
    if slc.shape[0] > 1:
        d0 = (slc[1:, :] != slc[:-1, :]).astype(np.float32)
        b[1:, :] += d0
        b[:-1, :] += d0
    if slc.shape[1] > 1:
        d1 = (slc[:, 1:] != slc[:, :-1]).astype(np.float32)
        b[:, 1:] += d1
        b[:, :-1] += d1
    return b


def transition_rate_2d(slc: np.ndarray) -> float:
    t = 0.0
    c = 0
    if slc.shape[0] > 1:
        t += float(np.mean(slc[1:, :] != slc[:-1, :]))
        c += 1
    if slc.shape[1] > 1:
        t += float(np.mean(slc[:, 1:] != slc[:, :-1]))
        c += 1
    return t / max(1, c)


def boundary_depth_curve(slc: np.ndarray) -> np.ndarray:
    # slc is [Z, H], returns boundary depth per horizontal coordinate
    if slc.shape[0] < 2:
        return np.zeros((slc.shape[1],), dtype=np.float32)
    dz = (slc[1:, :] != slc[:-1, :]).astype(np.float32)
    z_idx = np.arange(dz.shape[0], dtype=np.float32)[:, None]
    w = dz.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = (z_idx * dz).sum(axis=0) / np.maximum(w, 1e-6)
    depth[w <= 0] = np.nan
    if np.all(np.isnan(depth)):
        return np.zeros_like(depth)
    m = np.nanmean(depth)
    depth = np.where(np.isnan(depth), m, depth)
    return depth.astype(np.float32)


def curvature_proxy_from_depth(depth: np.ndarray) -> float:
    if depth.size < 5:
        return 0.0
    d2 = np.diff(depth, n=2)
    return float(np.mean(np.abs(d2)))


def periodicity_proxy(depth: np.ndarray) -> float:
    if depth.size < 8:
        return 0.0
    x = depth - np.mean(depth)
    spec = np.abs(np.fft.rfft(x))
    if spec.size <= 2:
        return 0.0
    peak = float(np.max(spec[1:]))
    base = float(np.mean(spec[1:]) + 1e-8)
    return peak / base


def dominant_frac_and_counts(vol: np.ndarray, ignore_label: int) -> Tuple[float, int, np.ndarray, np.ndarray]:
    vals = vol.reshape(-1)
    vals = vals[vals != int(ignore_label)]
    total_vox = int(vol.size)
    if vals.size == 0:
        return 1.0, 0, np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    labels, counts = np.unique(vals, return_counts=True)
    dom = float(np.max(counts)) / float(total_vox)
    return dom, int(labels.size), labels.astype(np.int64), counts.astype(np.int64)


def strat_present_count(labels: np.ndarray, strat_min: Optional[int], strat_max: Optional[int]) -> int:
    if labels.size == 0:
        return 0
    if strat_min is None or strat_max is None:
        return int(labels.size)
    m = (labels >= int(strat_min)) & (labels <= int(strat_max))
    return int(np.sum(m))


def fault_proxy_scores(vol: np.ndarray) -> Dict[str, float]:
    # volume [Z,Y,X]
    z, y, x = vol.shape
    dx = (vol[:, :, 1:] != vol[:, :, :-1]).astype(np.float32)
    dy = (vol[:, 1:, :] != vol[:, :-1, :]).astype(np.float32)

    # near-vertical discontinuities projected to Z-X / Z-Y
    dx_zx = dx.mean(axis=1) if y > 1 else np.zeros((z, max(1, x - 1)), dtype=np.float32)
    dy_zy = dy.mean(axis=2) if x > 1 else np.zeros((z, max(1, y - 1)), dtype=np.float32)

    def line_concentration(m: np.ndarray) -> float:
        if m.size == 0:
            return 0.0
        col = np.max(np.mean(m, axis=0)) if m.shape[1] > 0 else 0.0
        row = np.max(np.mean(m, axis=1)) if m.shape[0] > 0 else 0.0
        return float(max(col, row))

    # crude best-plane concentration via normal scan on discontinuity points
    pts = np.argwhere(np.pad(dx, ((0, 0), (0, 0), (0, 1)), mode="constant") + np.pad(dy, ((0, 0), (0, 1), (0, 0)), mode="constant") > 0)
    best_plane = 0.0
    if pts.shape[0] > 0:
        if pts.shape[0] > 20000:
            sel = np.random.default_rng(123).choice(pts.shape[0], size=20000, replace=False)
            pts = pts[sel]
        pts = pts.astype(np.float32)
        pts = pts - np.mean(pts, axis=0, keepdims=True)
        azis = np.linspace(0.0, 2.0 * np.pi, 18, endpoint=False)
        elev = np.linspace(-0.7, 0.7, 7)
        for a in azis:
            for e in elev:
                n = np.array([np.cos(e), np.sin(a) * np.sin(e), np.cos(a) * np.sin(e)], dtype=np.float32)
                p = pts @ n
                hist, _ = np.histogram(p, bins=24)
                sc = float(np.max(hist)) / float(max(1, pts.shape[0]))
                if sc > best_plane:
                    best_plane = sc

    return {
        "fault_proj_xz": line_concentration(dx_zx),
        "fault_proj_yz": line_concentration(dy_zy),
        "fault_plane_score": float(best_plane),
    }


def unconformity_proxy(vol: np.ndarray, ignore_label: int) -> Dict[str, float]:
    z, y, x = vol.shape
    if z < 4:
        return {"unconf_z_ref_est": 0.0, "unconf_trunc_score": 0.0, "unconf_thickness_frac_est": 0.0}

    scores = []
    for z0 in range(1, z - 1):
        top = vol[:z0, :, :].reshape(-1)
        bot = vol[z0:, :, :].reshape(-1)
        top = top[top != int(ignore_label)]
        bot = bot[bot != int(ignore_label)]
        if top.size == 0 or bot.size == 0:
            scores.append(0.0)
            continue
        lt, ct = np.unique(top, return_counts=True)
        lb, cb = np.unique(bot, return_counts=True)
        all_l = np.union1d(lt, lb)
        pt = np.zeros(all_l.size, dtype=np.float64)
        pb = np.zeros(all_l.size, dtype=np.float64)
        idx_t = np.searchsorted(all_l, lt)
        idx_b = np.searchsorted(all_l, lb)
        pt[idx_t] = ct / max(1, np.sum(ct))
        pb[idx_b] = cb / max(1, np.sum(cb))
        tv = 0.5 * np.sum(np.abs(pt - pb))
        # lateral continuity of boundary near z0
        b = float(np.mean(vol[z0, :, :] != vol[z0 - 1, :, :]))
        scores.append(float(tv * b))

    z_ref = int(np.argmax(scores)) + 1
    trunc_score = float(scores[z_ref - 1]) if scores else 0.0

    # estimate top-package thickness by similarity to top-layer distribution
    top_dist = vol[: max(1, z_ref // 3), :, :].reshape(-1)
    top_dist = top_dist[top_dist != int(ignore_label)]
    thickness = 1
    if top_dist.size > 0:
        lt, ct = np.unique(top_dist, return_counts=True)
        pt = ct / max(1, np.sum(ct))
        for zz in range(1, z):
            cur = vol[:zz, :, :].reshape(-1)
            cur = cur[cur != int(ignore_label)]
            if cur.size == 0:
                continue
            lc, cc = np.unique(cur, return_counts=True)
            all_l = np.union1d(lt, lc)
            a = np.zeros(all_l.size, dtype=np.float64)
            b = np.zeros(all_l.size, dtype=np.float64)
            a[np.searchsorted(all_l, lt)] = pt
            b[np.searchsorted(all_l, lc)] = cc / max(1, np.sum(cc))
            tv = 0.5 * np.sum(np.abs(a - b))
            if tv < 0.18:
                thickness = zz

    return {
        "unconf_z_ref_est": float(z_ref),
        "unconf_trunc_score": trunc_score,
        "unconf_thickness_frac_est": float(thickness) / float(max(1, z)),
    }


def slice_metrics(vol: np.ndarray, z_slice: int, y_slice: int, x_slice: int) -> Dict[str, float]:
    # vol [Z,Y,X]
    xy = vol[z_slice, :, :]
    xz = vol[:, y_slice, :]
    yz = vol[:, :, x_slice]

    c_xy = transition_rate_2d(xy)
    c_xz = transition_rate_2d(xz)
    c_yz = transition_rate_2d(yz)

    d_xz = boundary_depth_curve(xz)
    d_yz = boundary_depth_curve(yz)
    curv_xz = curvature_proxy_from_depth(d_xz)
    curv_yz = curvature_proxy_from_depth(d_yz)
    per_xz = periodicity_proxy(d_xz)
    per_yz = periodicity_proxy(d_yz)

    vis_xz = c_xz + curv_xz
    vis_yz = c_yz + curv_yz
    vis_ratio = vis_yz / max(1e-8, vis_xz)

    # principal variation direction using 3D boundary energies
    ex = float(np.mean(vol[:, :, 1:] != vol[:, :, :-1])) if vol.shape[2] > 1 else 0.0
    ey = float(np.mean(vol[:, 1:, :] != vol[:, :-1, :])) if vol.shape[1] > 1 else 0.0

    return {
        "complex_xy": c_xy,
        "complex_xz": c_xz,
        "complex_yz": c_yz,
        "curv_xz": curv_xz,
        "curv_yz": curv_yz,
        "period_xz": per_xz,
        "period_yz": per_yz,
        "vis_xz": vis_xz,
        "vis_yz": vis_yz,
        "vis_ratio_yz_over_xz": vis_ratio,
        "variation_energy_x": ex,
        "variation_energy_y": ey,
    }


def choose_slices(shape: Tuple[int, int, int], mode: str, z_override: Optional[int], y_override: Optional[int], x_override: Optional[int]) -> Tuple[int, int, int]:
    z, y, x = shape
    if mode == "mid":
        z0 = z // 2
        y0 = y // 2
        x0 = x // 2
    else:
        z0 = z // 2
        y0 = y // 2
        x0 = x // 2
    if z_override is not None:
        z0 = int(np.clip(z_override, 0, max(0, z - 1)))
    if y_override is not None:
        y0 = int(np.clip(y_override, 0, max(0, y - 1)))
    if x_override is not None:
        x0 = int(np.clip(x_override, 0, max(0, x - 1)))
    return z0, y0, x0


def maybe_save_slices_png(vol: np.ndarray, idx: int, z0: int, y0: int, x0: int, out_dir: Path):
    if plt is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(vol[z0, :, :], cmap="tab20", interpolation="nearest")
    axes[0].set_title(f"M{idx} XY z={z0}")
    axes[1].imshow(vol[:, y0, :], cmap="tab20", interpolation="nearest", aspect="auto")
    axes[1].set_title(f"M{idx} XZ y={y0}")
    axes[2].imshow(vol[:, :, x0], cmap="tab20", interpolation="nearest", aspect="auto")
    axes[2].set_title(f"M{idx} YZ x={x0}")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / f"model_{idx:04d}_slices.png", dpi=140)
    plt.close(fig)


def to_builtin(v: Any) -> Any:
    if isinstance(v, (np.floating, np.float32, np.float64)):
        return float(v)
    if isinstance(v, (np.integer, np.int32, np.int64)):
        return int(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npy", type=str, required=True)
    p.add_argument("--ignore_label", type=int, default=0)
    p.add_argument("--n_models", type=int, default=-1)
    p.add_argument("--model_indices", type=str, default=None)
    p.add_argument("--report_dir", type=str, default="diag_report")
    p.add_argument("--slice_mode", type=str, default="mid", choices=["mid", "specified"])
    p.add_argument("--z_slice", type=int, default=None)
    p.add_argument("--y_slice", type=int, default=None)
    p.add_argument("--x_slice", type=int, default=None)
    p.add_argument("--save_png", type=int, default=1)
    p.add_argument("--save_stats", type=int, default=1)
    p.add_argument("--expected_events", type=str, default=None)
    p.add_argument("--dominant_gate", type=float, default=None)
    p.add_argument("--min_strat_in_crop", type=int, default=None)
    p.add_argument("--strat_label_min", type=int, default=None)
    p.add_argument("--strat_label_max", type=int, default=None)
    args = p.parse_args()

    out_dir = Path(args.report_dir)
    ensure_dir(out_dir)
    if int(args.save_png) == 1:
        ensure_dir(out_dir / "slices")

    vols = load_volumes(args.npy)
    indices = select_model_indices(vols.shape[0], int(args.n_models), parse_indices(args.model_indices))
    if len(indices) == 0:
        raise RuntimeError("No models selected.")

    expected_events = None
    if args.expected_events:
        expected_events = json.loads(args.expected_events)

    rows: List[Dict[str, Any]] = []

    for i in indices:
        vol = np.asarray(vols[i])
        z0, y0, x0 = choose_slices(vol.shape, args.slice_mode, args.z_slice, args.y_slice, args.x_slice)

        dom, n_unique, labels, counts = dominant_frac_and_counts(vol, args.ignore_label)
        n_strat = strat_present_count(labels, args.strat_label_min, args.strat_label_max)

        sl = slice_metrics(vol, z0, y0, x0)
        fp = fault_proxy_scores(vol)
        up = unconformity_proxy(vol, args.ignore_label)

        gate_dom_pass = None if args.dominant_gate is None else (dom <= float(args.dominant_gate))
        gate_strat_pass = None if args.min_strat_in_crop is None else (n_strat >= int(args.min_strat_in_crop))

        row = {
            "model_idx": int(i),
            "dominant_frac": dom,
            "n_unique": int(n_unique),
            "n_strat_present": int(n_strat),
            "gate_dom_pass": gate_dom_pass,
            "gate_strat_pass": gate_strat_pass,
        }
        row.update(sl)
        row.update(fp)
        row.update(up)
        rows.append(row)

        if int(args.save_png) == 1 and plt is not None:
            maybe_save_slices_png(vol, int(i), z0, y0, x0, out_dir / "slices")

    # acceptance grid
    dom_grid = [0.30, 0.35, 0.40, 0.45]
    strat_grid = [4, 6, 8]
    acc_grid = []
    for dg in dom_grid:
        for sg in strat_grid:
            ok = 0
            for r in rows:
                if (r["dominant_frac"] <= dg) and (r["n_strat_present"] >= sg):
                    ok += 1
            acc_grid.append({"dominant_gate": dg, "min_strat": sg, "accept_rate": ok / float(len(rows))})

    # choose representatives
    arr_vis = np.array([r["vis_ratio_yz_over_xz"] for r in rows], dtype=np.float64)
    arr_fault = np.array([r["fault_plane_score"] for r in rows], dtype=np.float64)
    arr_uncth = np.array([r["unconf_thickness_frac_est"] for r in rows], dtype=np.float64)

    rep_best_fold = rows[int(np.argmax(arr_vis))]["model_idx"]
    rep_worst_fold = rows[int(np.argmin(arr_vis))]["model_idx"]
    rep_best_fault = rows[int(np.argmax(arr_fault))]["model_idx"]
    rep_thick_unconf = rows[int(np.argmax(arr_uncth))]["model_idx"]

    # aggregate findings for A-D
    med = lambda k: float(np.median([r[k] for r in rows]))
    mean = lambda k: float(np.mean([r[k] for r in rows]))

    txt = []
    txt.append("# Geological Diagnostics Report\n")
    txt.append(f"Analyzed models: {len(rows)}\n")
    txt.append(f"Input file: {args.npy}\n")

    txt.append("## A) Why YZ slices almost never show folds\n")
    txt.append(f"- Median visibility ratio (YZ/XZ): {med('vis_ratio_yz_over_xz'):.4f}.\n")
    txt.append(f"- Mean variation energy along X: {mean('variation_energy_x'):.4f}, along Y: {mean('variation_energy_y'):.4f}.\n")
    if mean("variation_energy_x") > mean("variation_energy_y") * 1.15:
        txt.append("- Interpretation: deformation varies more strongly with X than Y, so fixed-X YZ slices tend to appear flatter/less folded.\n")
    else:
        txt.append("- Interpretation: YZ fold invisibility is not solely due to X-dominant variation; inspect selected representative slices.\n")

    txt.append("\n## B) Are folds/faults/unconformities applied as intended (event count vs visual count)\n")
    if expected_events is not None:
        txt.append(f"- Provided expected event counts: {expected_events}.\n")
    txt.append(f"- Fold proxy (median XZ curvature): {med('curv_xz'):.4f}; median YZ curvature: {med('curv_yz'):.4f}.\n")
    txt.append(f"- Fault proxy (median plane score): {med('fault_plane_score'):.4f}.\n")
    txt.append(f"- Unconformity proxy (median truncation score): {med('unconf_trunc_score'):.4f}.\n")
    txt.append("- Visual under-count can occur when structures are present but orientation makes them weak in a chosen slice family.\n")

    txt.append("\n## C) Unconformity thickness/orientation/selective placement\n")
    txt.append(f"- Estimated unconformity thickness fraction median: {med('unconf_thickness_frac_est'):.4f}.\n")
    txt.append(f"- Estimated unconformity truncation score median: {med('unconf_trunc_score'):.4f}.\n")
    if med("unconf_thickness_frac_est") > 0.30:
        txt.append("- Potential issue: unconformity appears too thick and may dominate part of the crop.\n")
    else:
        txt.append("- Thickness appears within a modest range for most models.\n")

    txt.append("\n## D) Why v10_* gate passing may be harder than v9_*\n")
    txt.append("- Acceptance-rate grid (dominant_gate x min_strat):\n")
    for g in acc_grid:
        txt.append(f"  - dg={g['dominant_gate']:.2f}, min_strat={g['min_strat']}: accept={g['accept_rate']:.3f}\n")

    # bottleneck estimate
    frac_dom40 = np.mean([r["dominant_frac"] <= 0.40 for r in rows])
    frac_strat6 = np.mean([r["n_strat_present"] >= 6 for r in rows])
    txt.append(f"- Standalone pass rates: dominant<=0.40 -> {frac_dom40:.3f}, strat>=6 -> {frac_strat6:.3f}.\n")
    if frac_dom40 < frac_strat6:
        txt.append("- Bottleneck indication: dominant-label constraint is tighter than strat-count in this dataset.\n")
    else:
        txt.append("- Bottleneck indication: strat-count requirement is tighter than dominant-label constraint in this dataset.\n")

    txt.append("\n## Representative models\n")
    txt.append(f"- Best YZ fold visibility ratio: model {rep_best_fold}.\n")
    txt.append(f"- Worst YZ fold visibility ratio: model {rep_worst_fold}.\n")
    txt.append(f"- Strongest fault-plane proxy: model {rep_best_fault}.\n")
    txt.append(f"- Thickest unconformity estimate: model {rep_thick_unconf}.\n")

    txt.append("\n## Actionable recommendations\n")
    txt.append("- If YZ folds are weak: bias fold-axis sampling to increase Y-direction variation and inspect both XZ/YZ during tuning.\n")
    txt.append("- If fault visibility is weak: increase dip-slip-like pitch probability and ensure fault positions cross crop center more often.\n")
    txt.append("- If unconformity dominates: reduce unconformity thickness cap or narrow dip range around moderate values.\n")
    txt.append("- If gate acceptance is low: relax dominant_gate slightly or reduce min_strat_in_crop after examining acceptance grid.\n")

    report_path = out_dir / "diagnostic_report.md"
    report_path.write_text("".join(txt), encoding="utf-8")

    if int(args.save_stats) == 1:
        json_path = out_dir / "per_model_stats.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([{k: to_builtin(v) for k, v in r.items()} for r in rows], f, indent=2)

        csv_path = out_dir / "per_model_stats.csv"
        keys = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: to_builtin(v) for k, v in r.items()})

        grid_path = out_dir / "gate_acceptance_grid.csv"
        with open(grid_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["dominant_gate", "min_strat", "accept_rate"])
            w.writeheader()
            for g in acc_grid:
                w.writerow(g)

    # save representative images again with explicit names
    if int(args.save_png) == 1 and plt is not None:
        rep_dir = out_dir / "representative"
        ensure_dir(rep_dir)
        reps = [rep_best_fold, rep_worst_fold, rep_best_fault, rep_thick_unconf]
        names = ["best_fold_vis", "worst_fold_vis", "best_fault_proxy", "thickest_unconf"]
        for ridx, nm in zip(reps, names):
            vol = np.asarray(vols[ridx])
            z0, y0, x0 = choose_slices(vol.shape, args.slice_mode, args.z_slice, args.y_slice, args.x_slice)
            if plt is None:
                break
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(vol[z0, :, :], cmap="tab20", interpolation="nearest")
            axes[0].set_title(f"{nm} XY")
            axes[1].imshow(vol[:, y0, :], cmap="tab20", interpolation="nearest", aspect="auto")
            axes[1].set_title(f"{nm} XZ")
            axes[2].imshow(vol[:, :, x0], cmap="tab20", interpolation="nearest", aspect="auto")
            axes[2].set_title(f"{nm} YZ")
            for ax in axes:
                ax.axis("off")
            fig.tight_layout()
            fig.savefig(rep_dir / f"{nm}_model_{int(ridx):04d}.png", dpi=140)
            plt.close(fig)

    print(f"[DONE] report: {report_path}")
    if int(args.save_stats) == 1:
        print(f"[DONE] stats: {out_dir / 'per_model_stats.json'}")
        print(f"[DONE] stats: {out_dir / 'per_model_stats.csv'}")
        print(f"[DONE] grid:  {out_dir / 'gate_acceptance_grid.csv'}")


if __name__ == "__main__":
    main()
