#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
_gen_strat11_softprior50_okonly_comparefriendly.py (UPDATED)

User-requested upgrades:
  1) Configurable cube cropping: crop n*n*n -> m*m*m (defaults behave like 200->50 if raw is 200).
     - New arg: --crop_from_n (n). 0/negative means "use full block size".
     - Existing arg: --target_n (m).
  2) Total wall-time reporting for the whole run (start->finish).
  3) Quality gate acceleration via Torch (GPU if available), instead of NumPy-only CPU metrics.
     - New args: --qgate_torch, --qgate_device, --cuda_device.
  4) Optional smart crop origin on full block to maximize label complexity.
     - New args: --smart_crop, --smart_stride, --smart_alpha.

Everything else is kept the same as the provided comparefriendly generator.
"""

from __future__ import annotations

import os
import time
import argparse
import shutil
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import multiprocessing as mp

import pynoddy
from pynoddy.history import NoddyHistory
from pynoddy.output import NoddyOutput

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None


def resolve_noddy_path(user_arg: Optional[str]) -> str:
    if user_arg:
        p = Path(user_arg)
        if p.exists():
            return str(p)
        raise FileNotFoundError(f"--noddy not found: {p}")

    env = os.environ.get("NODDY_EXE")
    if env:
        p = Path(env)
        if p.exists():
            return str(p)

    cand = Path(r"C:\\Noddy\\noddy.exe")
    if cand.exists():
        return str(cand)

    for name in ["noddy.exe", "noddy_win64.exe"]:
        cand2 = Path.cwd() / name
        if cand2.exists():
            return str(cand2)

    which = shutil.which("noddy.exe") or shutil.which("noddy_win64.exe")
    if which:
        return which

    raise FileNotFoundError(
        'Cannot find Noddy executable. Provide --noddy "C:\\Noddy\\noddy.exe" or set env var NODDY_EXE.'
    )


RAW_DIR = "raw_noddy"
HIS_DIR = os.path.join(RAW_DIR, "his")
OUT_DIR = os.path.join(RAW_DIR, "out")
TMP_BLOCK_DIR = os.path.join("npy", "_tmp_blocks_okonly")
DEFAULT_OUT_PATH = os.path.join("npy", "models_softprior50_okonly.npy")


def ensure_dirs():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(HIS_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs("npy", exist_ok=True)
    os.makedirs(TMP_BLOCK_DIR, exist_ok=True)


def safe_cleanup_prefix(prefix: str):
    try:
        parent = Path(prefix).parent
        base = Path(prefix).name
        for p in parent.glob(base + ".*"):
            try:
                p.unlink()
            except Exception:
                pass
    except Exception:
        pass


def crop_direct_xyz(block_xyz: np.ndarray, target_n: int, z_crop: str) -> np.ndarray:
    nx, ny, nz = block_xyz.shape
    if target_n > nx or target_n > ny or target_n > nz:
        raise ValueError(f"target_n={target_n} larger than block {block_xyz.shape}")

    sx = (nx - target_n) // 2
    sy = (ny - target_n) // 2

    if z_crop == "top":
        sz0 = 0
    elif z_crop == "center":
        sz0 = (nz - target_n) // 2
    elif z_crop == "bottom":
        sz0 = nz - target_n
    else:
        raise ValueError(f"Unknown z_crop={z_crop}")

    return block_xyz[sx : sx + target_n, sy : sy + target_n, sz0 : sz0 + target_n]


def _z0_for_crop(nz: int, target_n: int, z_crop: str) -> int:
    if z_crop == "top":
        return 0
    if z_crop == "center":
        return (nz - target_n) // 2
    if z_crop == "bottom":
        return nz - target_n
    raise ValueError(f"Unknown z_crop={z_crop}")


def smart_crop_direct_xyz(
    block_xyz: np.ndarray,
    target_n: int,
    z_crop: str,
    smart_stride: int,
    ignore_label: int,
    smart_alpha: float,
):
    nx, ny, nz = block_xyz.shape
    if target_n > nx or target_n > ny or target_n > nz:
        raise ValueError(f"target_n={target_n} larger than block {block_xyz.shape}")

    sx_max = nx - target_n
    sy_max = ny - target_n
    z0 = int(np.clip(_z0_for_crop(nz, target_n, z_crop), 0, nz - target_n))

    stride = max(1, int(smart_stride))
    x_starts = list(range(0, sx_max + 1, stride))
    y_starts = list(range(0, sy_max + 1, stride))
    if len(x_starts) == 0 or x_starts[-1] != sx_max:
        x_starts.append(sx_max)
    if len(y_starts) == 0 or y_starts[-1] != sy_max:
        y_starts.append(sy_max)

    best = None
    best_info = None
    total_vox = float(target_n * target_n * target_n)
    ig = int(ignore_label)
    alpha = float(smart_alpha)

    for x0 in x_starts:
        x1 = x0 + target_n
        for y0 in y_starts:
            y1 = y0 + target_n
            w = block_xyz[x0:x1, y0:y1, z0 : z0 + target_n]
            labels, counts = np.unique(w, return_counts=True)
            if labels.size == 0:
                unique_count = 0
                dominant_frac = 1.0
            else:
                if ig in labels:
                    mask = labels != ig
                    labels_nz = labels[mask]
                    counts_nz = counts[mask]
                else:
                    labels_nz = labels
                    counts_nz = counts

                unique_count = int(labels_nz.size)
                dominant_frac = (float(np.max(counts_nz)) / total_vox) if counts_nz.size > 0 else 1.0

            score = float(unique_count) - alpha * float(dominant_frac)
            cand = (score, unique_count, dominant_frac, int(x0), int(y0), int(z0))
            if best is None or cand[0] > best[0]:
                best = cand
                best_info = (int(x0), int(y0), int(z0), float(score), int(unique_count), float(dominant_frac), int(nx), int(ny), int(nz))

    if best is None:
        x0 = 0
        y0 = 0
        z0 = int(np.clip(z0, 0, nz - target_n))
        w = block_xyz[x0 : x0 + target_n, y0 : y0 + target_n, z0 : z0 + target_n]
        labels, counts = np.unique(w, return_counts=True)
        unique_count = int(labels.size)
        dominant_frac = (float(np.max(counts)) / total_vox) if counts.size > 0 else 1.0
        score = float(unique_count) - alpha * float(dominant_frac)
        best_info = (x0, y0, z0, score, unique_count, dominant_frac, int(nx), int(ny), int(nz))

    bx, by, bz, _, _, _, _, _, _ = best_info
    cropped = block_xyz[bx : bx + target_n, by : by + target_n, bz : bz + target_n]
    return cropped, best_info


def crop_n_to_m_xyz(block_xyz: np.ndarray, crop_from_n: int, crop_to_n: int, z_crop: str) -> np.ndarray:
    nx, ny, nz = block_xyz.shape
    full_n = min(int(nx), int(ny), int(nz))

    n = int(crop_from_n)
    m = int(crop_to_n)

    if n <= 0:
        n = full_n
    if n > full_n:
        raise ValueError(f"crop_from_n={n} larger than block min-dim={full_n} (shape={block_xyz.shape})")
    if m > n:
        raise ValueError(f"crop_to_n={m} cannot be larger than crop_from_n={n}")

    if n != full_n:
        block_xyz = crop_direct_xyz(block_xyz, n, z_crop)

    if m != n:
        block_xyz = crop_direct_xyz(block_xyz, m, z_crop)

    return block_xyz


def xyz_to_zyx(block_xyz: np.ndarray) -> np.ndarray:
    return np.transpose(block_xyz, (2, 1, 0))


def axis_metrics_numpy(block_zyx: np.ndarray, ignore_label: int = 0) -> Dict[str, float]:
    a = block_zyx
    uniq = np.unique(a)
    if ignore_label in uniq:
        uniq = uniq[uniq != ignore_label]
    unique_nonignore = int(uniq.size)

    dz = float(np.mean(a[1:, :, :] != a[:-1, :, :])) if a.shape[0] > 1 else 0.0
    dy = float(np.mean(a[:, 1:, :] != a[:, :-1, :])) if a.shape[1] > 1 else 0.0
    dx = float(np.mean(a[:, :, 1:] != a[:, :, :-1])) if a.shape[2] > 1 else 0.0

    bd = (dz + dy + dx) / 3.0
    mx = max(dz, dy, dx)
    mn = min(dz, dy, dx)
    axis_ratio = (mn / mx) if mx > 0 else 0.0

    return {
        "unique": float(unique_nonignore),
        "dz": dz,
        "dy": dy,
        "dx": dx,
        "bd": bd,
        "axis_ratio": axis_ratio,
    }


def _pick_torch_device(qgate_device: str, cuda_device: int):
    if torch is None:
        return None

    mode = str(qgate_device).lower()
    if mode == "cpu":
        return torch.device("cpu")

    if mode == "cuda":
        if torch.cuda.is_available():
            try:
                torch.cuda.set_device(int(cuda_device))
            except Exception:
                pass
            return torch.device(f"cuda:{int(cuda_device)}")
        return torch.device("cpu")

    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(int(cuda_device))
        except Exception:
            pass
        return torch.device(f"cuda:{int(cuda_device)}")
    return torch.device("cpu")


def axis_metrics_torch(block_zyx: np.ndarray, ignore_label: int, device, cuda_sync: int = 0) -> Dict[str, float]:
    a = torch.from_numpy(block_zyx)
    a = a.to(device=device, dtype=torch.int32, non_blocking=True)

    ig = int(ignore_label)
    nz_frac = (a != ig).to(torch.float32).mean()
    uniq = torch.unique(a)
    if uniq.numel() > 0:
        uniq = uniq[uniq != ig]
    unique_nonignore = float(uniq.numel())

    dz = (a[1:, :, :] != a[:-1, :, :]).to(torch.float32).mean() if a.shape[0] > 1 else torch.tensor(0.0, device=device)
    dy = (a[:, 1:, :] != a[:, :-1, :]).to(torch.float32).mean() if a.shape[1] > 1 else torch.tensor(0.0, device=device)
    dx = (a[:, :, 1:] != a[:, :, :-1]).to(torch.float32).mean() if a.shape[2] > 1 else torch.tensor(0.0, device=device)

    bd = (dz + dy + dx) / 3.0

    if int(cuda_sync) == 1 and device.type == "cuda":
        torch.cuda.synchronize(device)

    dzv = float(dz.item())
    dyv = float(dy.item())
    dxv = float(dx.item())
    bdv = float(bd.item())
    mx = max(dzv, dyv, dxv)
    mn = min(dzv, dyv, dxv)
    ar = (mn / mx) if mx > 0 else 0.0

    return {
        "unique": float(unique_nonignore),
        "dz": dzv,
        "dy": dyv,
        "dx": dxv,
        "bd": bdv,
        "axis_ratio": float(ar),
        "nz_frac": float(nz_frac.item()),
    }


def _ang_dist(a: float, b: float) -> float:
    d = (a - b) % 360.0
    if d > 180.0:
        d = 360.0 - d
    return float(d)


def sample_angle_avoid(rng: np.random.Generator, lo: float, hi: float, avoid: List[float], tol: float, max_tries: int = 50) -> float:
    a = float(rng.uniform(lo, hi))
    for _ in range(max_tries):
        ok = True
        for c in avoid:
            if _ang_dist(a, c) < tol:
                ok = False
                break
        if ok:
            return float(a)
        a = float(rng.uniform(lo, hi))
    return float(a)


def add_stratigraphy(h: NoddyHistory, rng: np.random.Generator, name: str = "STRAT_1", n_layers_fixed: Optional[int] = None, nmin: int = 6, nmax: int = 8, tmin: int = 140, tmax: int = 280) -> int:
    if n_layers_fixed is not None and int(n_layers_fixed) > 0:
        num_layers = int(n_layers_fixed)
    else:
        num_layers = int(rng.integers(nmin, nmax + 1))

    thicknesses = rng.integers(tmin, tmax, size=num_layers).astype(int).tolist()
    layer_names = [f"L{i+1}" for i in range(num_layers)]
    h.add_event("stratigraphy", {"name": name, "num_layers": int(num_layers), "layer_names": layer_names, "layer_thickness": thicknesses, "layer_thicknesses": thicknesses})
    return num_layers


def add_tilt(h: NoddyHistory, rng: np.random.Generator, name: str, rotation_range: Tuple[float, float] = (-18.0, 18.0), plunge_dir_tol: float = 12.0, plunge_range: Tuple[float, float] = (-12.0, 12.0), pos_xy: Tuple[float, float] = (-900.0, 900.0), pos_z: Tuple[float, float] = (-1500.0, -120.0)) -> None:
    rotation = float(rng.uniform(*rotation_range))
    plunge_direction = sample_angle_avoid(rng, 0.0, 360.0, avoid=[0.0, 90.0, 180.0, 270.0], tol=plunge_dir_tol)
    plunge = float(rng.uniform(*plunge_range))
    px = float(rng.uniform(*pos_xy))
    py = float(rng.uniform(*pos_xy))
    pz = float(rng.uniform(*pos_z))
    h.add_event("tilt", {"name": name, "rotation": rotation, "plunge_direction": plunge_direction, "plunge": plunge, "pos": [px, py, pz]})


def add_fold_soft(h: NoddyHistory, rng: np.random.Generator, name: str, amp_range: Tuple[float, float] = (120.0, 420.0), wl_range: Tuple[float, float] = (1200.0, 2800.0), z_range: Tuple[float, float] = (-1800.0, -220.0), axis_tol: float = 12.0, plunge_range: Tuple[float, float] = (6.0, 35.0)) -> None:
    amp = float(rng.uniform(*amp_range))
    wl = float(rng.uniform(*wl_range))
    px = float(rng.uniform(-1200.0, 1200.0))
    py = float(rng.uniform(-1200.0, 1200.0))
    pz = float(rng.uniform(*z_range))
    axis_dir = sample_angle_avoid(rng, 0.0, 180.0, avoid=[0.0, 90.0, 180.0], tol=axis_tol)
    plunge = float(rng.uniform(*plunge_range))
    h.add_event("fold", {"name": name, "pos": [px, py, pz], "amplitude": amp, "wavelength": wl, "axis_dir": axis_dir, "plunge": plunge})


def add_fault_soft(h: NoddyHistory, rng: np.random.Generator, name: str, slip_scale: float, dip_dir_tol: float = 12.0) -> None:
    dip_dir = sample_angle_avoid(rng, 0.0, 360.0, avoid=[0.0, 90.0, 180.0, 270.0], tol=dip_dir_tol)
    dip = float(rng.uniform(60.0, 82.0))
    pitch = float(rng.uniform(-45.0, 45.0))
    pz = float(rng.uniform(-1700.0, -220.0))
    slip = float(rng.uniform(180.0, 520.0)) * float(slip_scale)
    px = float(rng.uniform(-1200.0, 1200.0))
    py = float(rng.uniform(-1200.0, 1200.0))
    h.add_event("fault", {"name": name, "pos": [px, py, pz], "dip_dir": dip_dir, "dip": dip, "pitch": pitch, "slip": slip})


def _estimate_unconf_reference_z(h: NoddyHistory, rng: np.random.Generator) -> float:
    _, _, origin_z = h.get_origin()
    _, _, length_z = h.get_extent()
    top_z = float(origin_z)
    length_z = float(length_z)
    bottom_z = top_z - length_z

    # Place the unconformity near the middle of the deformed package so it
    # intersects folded/faulted stratigraphy instead of floating near the top.
    # Use a tight central band for slight randomness while staying robust.
    rel = float(rng.uniform(0.40, 0.60))
    pz = top_z - rel * length_z

    eps = 1e-6
    pz = float(np.clip(pz, bottom_z + eps, top_z - eps))
    return pz


def _estimate_unconf_thickness_cap(h: NoddyHistory) -> int:
    _, _, length_z = h.get_extent()
    length_z = float(length_z)

    # Keep UC thin: hard cap by both percentage and a voxel-like tiny marker.
    cap_pct = max(1, int(np.floor(0.05 * length_z)))

    cube_size = None
    try:
        cs = h.get_cube_size()
        if isinstance(cs, (int, float)):
            cube_size = float(cs)
        elif isinstance(cs, (tuple, list)) and len(cs) > 0:
            cube_size = float(cs[0])
    except Exception:
        cube_size = None

    if cube_size is not None and cube_size > 0:
        cap_vox = max(1, int(np.floor(2.0 * cube_size)))
    else:
        cap_vox = max(1, int(np.floor(0.02 * length_z)))

    return int(max(1, min(cap_pct, cap_vox)))


def add_unconformity_soft(h: NoddyHistory, rng: np.random.Generator, name: str, dip_dir_tol: float = 12.0, pz_range: Tuple[float, float] = (-280.0, -230.0), dip_range: Tuple[float, float] = (0.0, 6.0), n_layers_fixed: Optional[int] = None, n_layers_range: Tuple[int, int] = (1, 2), t_range: Tuple[int, int] = (1, 10), max_total_thickness: int = 10) -> None:
    _, _, origin_z = h.get_origin()
    _, _, length_z = h.get_extent()
    top_z = float(origin_z)
    length_z = float(length_z)
    bottom_z = top_z - length_z

    px = float(rng.uniform(-1000.0, 1000.0))
    py = float(rng.uniform(-1000.0, 1000.0))

    # Use a robust, model-state-aware reference level inside the deformed stack.
    pz = _estimate_unconf_reference_z(h, rng)

    dip_dir = sample_angle_avoid(rng, 0.0, 360.0, avoid=[0.0, 90.0, 180.0, 270.0], tol=dip_dir_tol)
    dip = float(rng.uniform(*dip_range))

    cap_float = float(_estimate_unconf_thickness_cap(h))
    cap_i = int(max(1, np.floor(cap_float)))

    tmin = int(max(1, min(t_range[0], t_range[1])))
    tmax = int(max(tmin, max(t_range[0], t_range[1])))
    per_layer_max = int(min(cap_i, tmax))

    if n_layers_fixed is not None and int(n_layers_fixed) > 0:
        num_layers = int(n_layers_fixed)
    else:
        a, b = int(n_layers_range[0]), int(n_layers_range[1])
        if b < a:
            a, b = b, a
        num_layers = int(rng.integers(a, b + 1))

    if per_layer_max <= 0:
        per_layer_max = 1
    max_layers_thin = 2
    if num_layers > max_layers_thin:
        num_layers = max_layers_thin

    base_target = int(max_total_thickness) if (max_total_thickness is not None and int(max_total_thickness) > 0) else (tmin * num_layers)
    target_total_i = int(max(tmin * num_layers, min(base_target, cap_i)))

    max_feasible = per_layer_max * num_layers
    if target_total_i > max_feasible:
        target_total_i = max_feasible

    thicknesses = [tmin] * num_layers
    remaining = target_total_i - tmin * num_layers
    guard = 0
    while remaining > 0 and guard < 200000:
        j = int(rng.integers(0, num_layers))
        headroom = per_layer_max - thicknesses[j]
        if headroom <= 0:
            guard += 1
            continue
        add = int(min(remaining, headroom))
        thicknesses[j] += add
        remaining -= add
        guard += 1

    thicknesses = [int(max(1, min(cap_i, t))) for t in thicknesses]

    layer_names = [f"UC{i+1}" for i in range(len(thicknesses))]

    opts = {
        "name": name,
        "pos": [px, py, pz],
        "dip_dir": dip_dir,
        "dip_direction": dip_dir,
        "dip": dip,
        "num_layers": int(len(thicknesses)),
        "layer_names": layer_names,
        "layer_thickness": thicknesses,
        "layer_thicknesses": thicknesses,
    }
    h.add_event("unconformity", opts)


def compute_effective_quality_thresholds(quality_strength: float, base: Dict[str, float]) -> Dict[str, float]:
    s = float(quality_strength)
    s = max(0.0, min(1.0, s))

    lenient = {"min_nonzero_frac": 0.0, "min_unique": 0.0, "min_bd": 0.0, "min_axis_diff": 0.0, "min_axis_ratio": 0.0}
    strict = {"min_nonzero_frac": 1.1, "min_unique": 1e12, "min_bd": 1.1, "min_axis_diff": 1.1, "min_axis_ratio": 1.1}

    def lerp(a: float, b: float, t: float) -> float:
        return float(a + t * (b - a))

    if s <= 0.5:
        t = s / 0.5 if 0.5 > 0 else 0.0
        out = {k: lerp(lenient[k], base[k], t) for k in base.keys()}
    else:
        t = (s - 0.5) / 0.5
        out = {k: lerp(base[k], strict[k], t) for k in base.keys()}

    out["min_unique"] = float(np.ceil(out["min_unique"]))
    return out


def worker_one(args_tuple):
    (
        i, base_seed, noddy_path,
        crop_from_n, target_n, direct_crop, z_crop,
        smart_crop, smart_stride, smart_alpha,
        max_retries, keep_attempt_files, keep_tmp, ignore_label,
        strat_layers, n_tilts, n_folds, n_fault_gentle, n_fault_violent,
        slip_scale_gentle, slip_scale_violent, unconf_layers, struct_mode, slip_scale_legacy,
        q_min_nonzero_frac, q_min_unique, q_min_bd, q_min_axis_diff, q_min_axis_ratio,
        qgate_torch, qgate_device, cuda_device, cuda_sync,
        tilt2_prob,
    ) = args_tuple

    pid = os.getpid()
    last_err = ""

    use_torch = bool(int(qgate_torch)) and (torch is not None)
    torch_device = None
    if use_torch:
        torch_device = _pick_torch_device(str(qgate_device), int(cuda_device))

    for attempt in range(int(max_retries) + 1):
        try:
            seed_i = int(base_seed) + int(i) * 1000003 + attempt * 9176
            rng = np.random.default_rng(seed_i)

            h = NoddyHistory()

            if int(strat_layers) > 0:
                add_stratigraphy(h, rng, n_layers_fixed=int(strat_layers))
            else:
                add_stratigraphy(h, rng)

            if int(n_tilts) >= 0:
                if int(n_tilts) >= 1:
                    add_tilt(h, rng, name="TILT_1")
                for k in range(2, int(n_tilts) + 1):
                    add_tilt(h, rng, name=f"TILT_{k}", rotation_range=(-10.0, 10.0), plunge_range=(-8.0, 8.0))
            else:
                add_tilt(h, rng, name="TILT_1")
                if rng.random() < float(tilt2_prob):
                    add_tilt(h, rng, name="TILT_2", rotation_range=(-10.0, 10.0), plunge_range=(-8.0, 8.0))

            explicit_faults = (int(n_fault_gentle) >= 0) or (int(n_fault_violent) >= 0)
            explicit_folds = int(n_folds) >= 0
            explicit_unconf = int(unconf_layers) >= 0
            should_add_unconf = False
            unconf_fixed_layers = None

            if explicit_folds:
                for k in range(int(n_folds)):
                    add_fold_soft(h, rng, name=f"FOLD_{k+1}")

            if explicit_faults:
                ng = int(n_fault_gentle) if int(n_fault_gentle) >= 0 else 0
                nv = int(n_fault_violent) if int(n_fault_violent) >= 0 else 0
                idx = 1
                for _ in range(ng):
                    add_fault_soft(h, rng, name=f"FAULT_{idx}", slip_scale=float(slip_scale_gentle))
                    idx += 1
                for _ in range(nv):
                    add_fault_soft(h, rng, name=f"FAULT_{idx}", slip_scale=float(slip_scale_violent))
                    idx += 1

            if explicit_unconf:
                if int(unconf_layers) > 0:
                    should_add_unconf = True
                    unconf_fixed_layers = int(unconf_layers)

            if (not explicit_folds) or (not explicit_faults) or (not explicit_unconf):
                structs = ["fault", "fold", "unconf"]
                if str(struct_mode) == "2of3":
                    chosen = rng.choice(structs, size=2, replace=False).tolist()
                elif str(struct_mode) == "fault_only":
                    chosen = ["fault"]
                elif str(struct_mode) == "fold_only":
                    chosen = ["fold"]
                elif str(struct_mode) == "unconf_only":
                    chosen = ["unconf"]
                elif str(struct_mode) == "none":
                    chosen = []
                else:
                    chosen = structs

                if (not explicit_folds) and ("fold" in chosen):
                    add_fold_soft(h, rng, name="FOLD_1")
                if (not explicit_faults) and ("fault" in chosen):
                    add_fault_soft(h, rng, name="FAULT_1", slip_scale=float(slip_scale_legacy))
                if (not explicit_unconf) and ("unconf" in chosen):
                    should_add_unconf = True

            if should_add_unconf:
                if unconf_fixed_layers is not None:
                    add_unconformity_soft(h, rng, name="UNCONF_1", n_layers_fixed=int(unconf_fixed_layers))
                else:
                    add_unconformity_soft(h, rng, name="UNCONF_1")

            his_name = os.path.join(HIS_DIR, f"mc_{i:06d}_p{pid}_a{attempt:02d}.his")
            out_prefix = os.path.join(OUT_DIR, f"mc_{i:06d}_p{pid}_a{attempt:02d}")
            h.write_history(his_name)
            pynoddy.compute_model(his_name, out_prefix, noddy_path=noddy_path)

            out = NoddyOutput(out_prefix)
            block_xyz = out.block
            if not isinstance(block_xyz, np.ndarray):
                raise RuntimeError("NoddyOutput.block is not ndarray")

            if int(direct_crop) != 1:
                raise RuntimeError("This generator expects --direct_crop=1")

            smart_info = None
            if int(smart_crop) == 1:
                try:
                    block_xyz, smart_info = smart_crop_direct_xyz(
                        block_xyz,
                        int(target_n),
                        str(z_crop),
                        int(smart_stride),
                        int(ignore_label),
                        float(smart_alpha),
                    )
                except Exception:
                    block_xyz = crop_n_to_m_xyz(block_xyz, int(crop_from_n), int(target_n), str(z_crop))
                    smart_info = None
            else:
                block_xyz = crop_n_to_m_xyz(block_xyz, int(crop_from_n), int(target_n), str(z_crop))

            block_zyx = xyz_to_zyx(block_xyz).astype(np.int16)

            if use_torch and (torch_device is not None):
                m = axis_metrics_torch(block_zyx, ignore_label=int(ignore_label), device=torch_device, cuda_sync=int(cuda_sync))
                nz_frac = float(m["nz_frac"])
                unique = int(m["unique"])
                bd = float(m["bd"])
                dz, dy, dx = float(m["dz"]), float(m["dy"]), float(m["dx"])
                ar = float(m["axis_ratio"])
            else:
                nz_frac = float(np.mean(block_zyx != int(ignore_label)))
                m = axis_metrics_numpy(block_zyx, ignore_label=int(ignore_label))
                unique = int(m["unique"])
                bd = float(m["bd"])
                dz, dy, dx = float(m["dz"]), float(m["dy"]), float(m["dx"])
                ar = float(m["axis_ratio"])

            if nz_frac < float(q_min_nonzero_frac):
                raise RuntimeError(f"quality_reject nz_frac={nz_frac:.6f} < {float(q_min_nonzero_frac):.6f}")
            if unique < int(q_min_unique):
                raise RuntimeError(f"quality_reject unique={unique} < {int(q_min_unique)} (dz,dy,dx)=({dz:.4f},{dy:.4f},{dx:.4f}) bd={bd:.4f} ar={ar:.4f}")
            if bd < float(q_min_bd):
                raise RuntimeError(f"quality_reject bd={bd:.4f} < {float(q_min_bd):.4f} unique={unique} (dz,dy,dx)=({dz:.4f},{dy:.4f},{dx:.4f}) ar={ar:.4f}")
            if min(dz, dy, dx) < float(q_min_axis_diff):
                raise RuntimeError(f"quality_reject min_axis_diff={min(dz,dy,dx):.4f} < {float(q_min_axis_diff):.4f} unique={unique} bd={bd:.4f} (dz,dy,dx)=({dz:.4f},{dy:.4f},{dx:.4f}) ar={ar:.4f}")
            if ar < float(q_min_axis_ratio):
                raise RuntimeError(f"quality_reject axis_ratio={ar:.4f} < {float(q_min_axis_ratio):.4f} unique={unique} bd={bd:.4f} (dz,dy,dx)=({dz:.4f},{dy:.4f},{dx:.4f})")

            if int(smart_crop) == 1 and smart_info is not None:
                x0, y0, z0, score, uq, dfrac, fx, fy, fz = smart_info
                print(f"[INFO] smart_crop origin=({x0},{y0},{z0}) score={score:.6f} unique={uq} dominant_frac={dfrac:.6f} full_dims=({fx},{fy},{fz})")

            tmp_path = os.path.join(TMP_BLOCK_DIR, f"blk_{i:06d}.npy")
            np.save(tmp_path, block_zyx)

            if not bool(int(keep_attempt_files)):
                safe_cleanup_prefix(out_prefix)
                try:
                    os.remove(his_name)
                except Exception:
                    pass

            return (i, tmp_path, True, f"ok attempt={attempt} nz_frac={nz_frac:.4f} unique={unique} bd={bd:.4f} ar={ar:.4f}")

        except Exception as e:
            last_err = f"attempt={attempt} err={repr(e)}"
            if not bool(int(keep_attempt_files)):
                try:
                    safe_cleanup_prefix(out_prefix)
                except Exception:
                    pass
                try:
                    if os.path.exists(his_name):
                        os.remove(his_name)
                except Exception:
                    pass
            continue

    if not bool(int(keep_attempt_files)):
        safe_cleanup_prefix(os.path.join(OUT_DIR, f"mc_{i:06d}_p{pid}_a*"))
    return (i, "", False, last_err)


def _fmt_hms(seconds: float) -> str:
    s = int(round(seconds))
    hh = s // 3600
    mm = (s % 3600) // 60
    ss = s % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def main():
    t0_total = time.time()

    p = argparse.ArgumentParser()
    p.add_argument("--n_models", type=int, default=1000, help="Target number of models AFTER quality filtering (N_ok).")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--noddy", type=str, default=None)
    p.add_argument("--out", type=str, default=DEFAULT_OUT_PATH)

    p.add_argument("--crop_from_n", type=int, default=0, help="First crop cube size n (n*n*n). 0/negative means use full block size.")
    p.add_argument("--target_n", type=int, default=50, help="Final crop cube size m (m*m*m).")
    p.add_argument("--direct_crop", type=int, default=1)
    p.add_argument("--z_crop", type=str, default="top", choices=["top", "center", "bottom"])

    p.add_argument("--smart_crop", type=int, default=0, help="1=enable smart crop origin search on full block, 0=legacy crop.")
    p.add_argument("--smart_stride", type=int, default=2, help="Smart-crop XY search stride in voxels.")
    p.add_argument("--smart_alpha", type=float, default=5.0, help="Smart-crop score alpha: unique_count - alpha * dominant_frac.")

    p.add_argument("--struct_mode", type=str, default="2of3", choices=["2of3", "all", "fault_only", "fold_only", "unconf_only", "none"], help="Legacy switch: which of {fault,fold,unconf} to include when explicit counts are not set.")
    p.add_argument("--slip_scale", type=float, default=1.0, help="Legacy fault slip scale (used when --n_fault_gentle/--n_fault_violent are not set).")

    p.add_argument("--strat_layers", type=int, default=-1, help="Fixed stratigraphy layer count. -1 keeps legacy random (6–8).")
    p.add_argument("--n_tilts", type=int, default=-1, help="Number of tilts. -1 keeps legacy (1 + optional 2nd).")
    p.add_argument("--n_folds", type=int, default=-1, help="Number of folds. -1 keeps legacy (0/1 depending on struct_mode).")

    p.add_argument("--n_fault_gentle", type=int, default=-1, help="Number of gentle faults. -1 keeps legacy fault logic.")
    p.add_argument("--n_fault_violent", type=int, default=-1, help="Number of violent faults. -1 keeps legacy fault logic.")
    p.add_argument("--slip_scale_gentle", type=float, default=0.6, help="Slip scale for gentle faults.")
    p.add_argument("--slip_scale_violent", type=float, default=1.2, help="Slip scale for violent faults.")

    p.add_argument("--unconf_layers", type=int, default=-1, help="Fixed number of unconformity layers. 0 disables unconf. -1 keeps legacy random (1–3) if unconf is used.")

    p.add_argument("--quality", type=float, default=0.5, help="Quality gate strength in [0,1]. 0=no filter, 0.5=baseline thresholds, 1=reject all.")
    p.add_argument("--max_candidates", type=int, default=0, help="Max total candidate ids to try (0 = no limit). Useful to prevent infinite loops when quality is too strict.")

    p.add_argument("--max_retries", type=int, default=12)
    p.add_argument("--fail_fast", type=int, default=0)
    p.add_argument("--keep_attempt_files", type=int, default=0)
    p.add_argument("--keep_tmp", type=int, default=0)

    p.add_argument("--ignore_label", type=int, default=0)

    p.add_argument("--min_nonzero_frac", type=float, default=0.05)
    p.add_argument("--min_unique", type=int, default=6)
    p.add_argument("--min_bd", type=float, default=0.02)
    p.add_argument("--min_axis_diff", type=float, default=0.008)
    p.add_argument("--min_axis_ratio", type=float, default=0.05)

    p.add_argument("--tilt2_prob", type=float, default=0.35)

    p.add_argument("--qgate_torch", type=int, default=1, help="1=use Torch for quality-gate metrics (GPU if available), 0=NumPy CPU metrics.")
    p.add_argument("--qgate_device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Torch device for quality-gate metrics: auto/cpu/cuda.")
    p.add_argument("--cuda_device", type=int, default=0, help="CUDA device index when qgate_device=auto/cuda and CUDA is available.")
    p.add_argument("--cuda_sync", type=int, default=0, help="1=call torch.cuda.synchronize() before reading metrics (more deterministic, slightly slower).")

    args = p.parse_args()

    ensure_dirs()
    noddy_path = resolve_noddy_path(args.noddy)

    target_ok = int(args.n_models)
    if target_ok <= 0:
        raise ValueError("--n_models must be > 0")

    if float(args.quality) >= 1.0:
        raise RuntimeError("--quality=1.0 is defined as reject-all; no ensemble can be produced.")

    crop_from_n = int(args.crop_from_n)
    target_n = int(args.target_n)
    if target_n <= 0:
        raise ValueError("--target_n must be > 0")
    if crop_from_n > 0 and target_n > crop_from_n:
        raise ValueError("--target_n (m) cannot be larger than --crop_from_n (n)")

    base_q = {
        "min_nonzero_frac": float(args.min_nonzero_frac),
        "min_unique": float(args.min_unique),
        "min_bd": float(args.min_bd),
        "min_axis_diff": float(args.min_axis_diff),
        "min_axis_ratio": float(args.min_axis_ratio),
    }
    eff_q = compute_effective_quality_thresholds(float(args.quality), base_q)

    torch_ok = (torch is not None)
    if int(args.qgate_torch) == 1 and not torch_ok:
        print("[WARN] --qgate_torch=1 but torch is not installed in this environment; falling back to NumPy CPU metrics.")

    print(f"[INFO] Output: {args.out}")
    print(f"[INFO] Crop: n->m = ({crop_from_n if crop_from_n>0 else 'auto'}) -> {target_n}, z_crop={args.z_crop}")
    print(f"[INFO] Smart crop: enabled={int(args.smart_crop)} stride={int(args.smart_stride)} alpha={float(args.smart_alpha):.3f}")
    print(f"[INFO] Target shape: (N_ok={target_ok}, {target_n}, {target_n}, {target_n}) int16, order=NZYX")
    print(f"[INFO] Noddy: {noddy_path}")
    print(f"[INFO] Workers={int(args.workers)}, seed={int(args.seed)}")
    print(f"[INFO] Geology controls: strat_layers={args.strat_layers} n_tilts={args.n_tilts} n_folds={args.n_folds} n_fault_gentle={args.n_fault_gentle} n_fault_violent={args.n_fault_violent} unconf_layers={args.unconf_layers}")
    print(f"[INFO] Legacy struct_mode={args.struct_mode} slip_scale={float(args.slip_scale):.3f}")
    print(f"[INFO] Quality gate strength={float(args.quality):.2f} => effective thresholds: min_nonzero_frac={eff_q['min_nonzero_frac']:.6f} min_unique={int(eff_q['min_unique'])} min_bd={eff_q['min_bd']:.6f} min_axis_diff={eff_q['min_axis_diff']:.6f} min_axis_ratio={eff_q['min_axis_ratio']:.6f}")
    if int(args.qgate_torch) == 1:
        print(f"[INFO] Quality gate backend: Torch ({'available' if torch_ok else 'NOT available'}) device_mode={args.qgate_device} cuda_device={int(args.cuda_device)} cuda_sync={int(args.cuda_sync)}")
    else:
        print("[INFO] Quality gate backend: NumPy (CPU)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    max_candidates = int(args.max_candidates)
    if max_candidates <= 0:
        max_candidates = 2**31 - 1

    ok_paths: List[str] = []
    ok_src_idx: List[int] = []
    failed_ids: List[int] = []

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=int(args.workers)) as pool:
        pending: Dict[int, mp.pool.ApplyResult] = {}

        next_id = 0
        submitted = 0
        done = 0

        def submit_one(i: int):
            nonlocal submitted
            work = (
                i, int(args.seed), noddy_path,
                int(args.crop_from_n), int(args.target_n), int(args.direct_crop), str(args.z_crop),
                int(args.smart_crop), int(args.smart_stride), float(args.smart_alpha),
                int(args.max_retries), int(args.keep_attempt_files), int(args.keep_tmp), int(args.ignore_label),
                int(args.strat_layers), int(args.n_tilts), int(args.n_folds), int(args.n_fault_gentle), int(args.n_fault_violent),
                float(args.slip_scale_gentle), float(args.slip_scale_violent), int(args.unconf_layers), str(args.struct_mode), float(args.slip_scale),
                float(eff_q["min_nonzero_frac"]), float(eff_q["min_unique"]), float(eff_q["min_bd"]), float(eff_q["min_axis_diff"]), float(eff_q["min_axis_ratio"]),
                int(args.qgate_torch), str(args.qgate_device), int(args.cuda_device), int(args.cuda_sync),
                float(args.tilt2_prob),
            )
            pending[i] = pool.apply_async(worker_one, (work,))
            submitted += 1

        while len(pending) < int(args.workers) and next_id < max_candidates:
            submit_one(next_id)
            next_id += 1

        last_report = time.time()
        while pending:
            progressed = False
            for i in list(pending.keys()):
                res = pending[i]
                if res.ready():
                    progressed = True
                    done += 1
                    try:
                        (idx, tmp_path, success, msg) = res.get()
                    except Exception as e:
                        success = False
                        idx = i
                        tmp_path = ""
                        msg = f"worker_crash err={repr(e)}"

                    del pending[i]

                    if success:
                        if len(ok_paths) < target_ok:
                            ok_paths.append(tmp_path)
                            ok_src_idx.append(int(idx))
                        else:
                            if not int(args.keep_tmp):
                                try:
                                    if tmp_path:
                                        os.remove(tmp_path)
                                except OSError:
                                    pass
                    else:
                        failed_ids.append(int(idx))
                        if int(args.fail_fast) == 1:
                            raise RuntimeError(f"Fail-fast: i={idx}: {msg}")
                        if len(failed_ids) <= 20:
                            print(f"[ERR] i={idx}: {msg}")

                    if len(ok_paths) < target_ok and next_id < max_candidates:
                        submit_one(next_id)
                        next_id += 1

            now = time.time()
            if (now - last_report) > 5.0:
                last_report = now
                print(f"[PROGRESS] done={done} submitted~={submitted} | ok={len(ok_paths)}/{target_ok} failed={len(failed_ids)} inflight={len(pending)} next_id={next_id}")

            if not progressed:
                time.sleep(0.05)

            if len(ok_paths) < target_ok and next_id >= max_candidates and not pending:
                break

    ok_n = len(ok_paths)
    if ok_n < target_ok:
        raise RuntimeError(f"Could not reach target_ok={target_ok}. Got ok={ok_n} after trying {min(max_candidates, next_id)} candidates. Try lowering --quality, increasing --max_candidates, or relaxing base thresholds.")

    out = np.lib.format.open_memmap(str(out_path), mode="w+", dtype=np.int16, shape=(target_ok, target_n, target_n, target_n))
    order = np.argsort(np.array(ok_src_idx[:target_ok], dtype=np.int64))
    ok_src_sorted = [ok_src_idx[int(j)] for j in order]
    ok_paths_sorted = [ok_paths[int(j)] for j in order]

    for j, tmp_path in enumerate(ok_paths_sorted):
        out[j] = np.load(tmp_path)
        if not int(args.keep_tmp):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    del out

    json_dir = Path("json") / "gen_softprior_okonly"
    json_dir.mkdir(parents=True, exist_ok=True)
    meta_path = json_dir / (out_path.stem + "_meta.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"target_ok={target_ok} ok={target_ok} failed={len(failed_ids)}\n")
        f.write(f"quality_strength={float(args.quality):.3f}\n")
        f.write(f"crop_from_n={int(args.crop_from_n)} target_n={int(args.target_n)} z_crop={str(args.z_crop)}\n")
        f.write(f"smart_crop={int(args.smart_crop)} smart_stride={int(args.smart_stride)} smart_alpha={float(args.smart_alpha):.6f}\n")
        f.write(f"qgate_torch={int(args.qgate_torch)} qgate_device={str(args.qgate_device)} cuda_device={int(args.cuda_device)} cuda_sync={int(args.cuda_sync)}\n")
        f.write("effective_thresholds=" + ",".join([f"{k}={eff_q[k]}" for k in ["min_nonzero_frac", "min_unique", "min_bd", "min_axis_diff", "min_axis_ratio"]]) + "\n")
        f.write("ok_src_idx_sorted=" + ",".join(map(str, ok_src_sorted)) + "\n")
        if failed_ids:
            failed_ids.sort()
            f.write("failed_ids=" + ",".join(map(str, failed_ids)) + "\n")

    total_s = time.time() - t0_total
    print(f"[DONE] saved ok-only: {args.out}")
    print(f"[DONE] meta: {meta_path}")
    print(f"[TIME] total_wall_s={total_s:.3f}  ({_fmt_hms(total_s)})")
    if target_ok > 0:
        print(f"[TIME] avg_wall_s_per_ok_model={total_s/float(target_ok):.6f}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
