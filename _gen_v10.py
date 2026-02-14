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


def _dominant_unique_numpy(values: np.ndarray, ignore_label: int, total_vox: int) -> Tuple[float, int]:
    labels, counts = np.unique(values, return_counts=True)
    if labels.size == 0:
        return 1.0, 0
    ig = int(ignore_label)
    if ig in labels:
        mask = labels != ig
        counts = counts[mask]
    if counts.size == 0:
        return 1.0, 0
    dominant_frac = float(np.max(counts)) / float(total_vox)
    unique_count = int(counts.size)
    return dominant_frac, unique_count


def _strat_count_numpy(values: np.ndarray, ignore_label: int) -> int:
    labels = np.unique(values)
    labels = labels[labels != int(ignore_label)]
    return int(labels.size)


def _boundary_volume_numpy(block_xyz: np.ndarray) -> np.ndarray:
    a = block_xyz
    b = np.zeros(a.shape, dtype=np.float32)

    dx = (a[1:, :, :] != a[:-1, :, :]).astype(np.float32)
    dy = (a[:, 1:, :] != a[:, :-1, :]).astype(np.float32)
    dz = (a[:, :, 1:] != a[:, :, :-1]).astype(np.float32)

    b[1:, :, :] += dx
    b[:-1, :, :] += dx
    b[:, 1:, :] += dy
    b[:, :-1, :] += dy
    b[:, :, 1:] += dz
    b[:, :, :-1] += dz
    return b


def smart_crop_direct_xyz(
    block_xyz: np.ndarray,
    target_n: int,
    z_crop: str,
    smart_stride: int,
    ignore_label: int,
    smart_alpha: float,
    smart_z_stride: int,
    smart_topk: int,
):
    nx, ny, nz = block_xyz.shape
    if target_n > nx or target_n > ny or target_n > nz:
        raise ValueError(f"target_n={target_n} larger than block {block_xyz.shape}")

    sx_max = nx - target_n
    sy_max = ny - target_n
    sz_max = nz - target_n

    sx = max(1, int(smart_stride))
    sy = max(1, int(smart_stride))
    sz = max(1, int(smart_z_stride))
    topk = max(1, int(smart_topk))

    total_vox = int(target_n * target_n * target_n)
    b_np = _boundary_volume_numpy(block_xyz)

    use_cuda = (torch is not None) and torch.cuda.is_available()
    coarse = []

    if use_cuda:
        import torch.nn.functional as F

        dev = torch.device(f"cuda:{torch.cuda.current_device()}")
        a_t = torch.from_numpy(block_xyz.astype(np.int64, copy=False)).to(dev)
        b_t = torch.from_numpy(b_np).to(dev)

        bt = b_t.unsqueeze(0).unsqueeze(0)
        k = torch.ones((1, 1, target_n, target_n, target_n), dtype=torch.float32, device=dev)
        scores = F.conv3d(bt, k, stride=(sx, sy, sz)).squeeze(0).squeeze(0)

        flat = scores.reshape(-1)
        k_take = min(topk, int(flat.numel()))
        vals, idx = torch.topk(flat, k=k_take, largest=True)
        idx = idx.to(torch.int64)

        sy_out = scores.shape[1]
        sz_out = scores.shape[2]
        plane = sy_out * sz_out
        for n in range(k_take):
            idv = int(idx[n].item())
            ix = idv // int(plane)
            rem = idv % int(plane)
            iy = rem // int(sz_out)
            iz = rem % int(sz_out)
            x0 = min(int(ix * sx), sx_max)
            y0 = min(int(iy * sy), sy_max)
            z0 = min(int(iz * sz), sz_max)
            coarse.append((x0, y0, z0, float(vals[n].item())))

        refine = set()
        for x0, y0, z0, _ in coarse:
            for xi in range(max(0, x0 - sx), min(sx_max, x0 + sx) + 1):
                for yi in range(max(0, y0 - sy), min(sy_max, y0 + sy) + 1):
                    for zi in range(max(0, z0 - sz), min(sz_max, z0 + sz) + 1):
                        refine.add((int(xi), int(yi), int(zi)))

        best = None
        ig = int(ignore_label)
        for x0, y0, z0 in refine:
            w = a_t[x0 : x0 + target_n, y0 : y0 + target_n, z0 : z0 + target_n].reshape(-1).to(torch.int64)
            if ig is not None:
                w = w[w != ig]
            if w.numel() == 0:
                dominant_frac = 1.0
                unique_count = 0
            else:
                mn = int(w.min().item())
                if mn < 0:
                    w = w - mn
                _, counts = torch.unique(w, return_counts=True)
                unique_count = int(counts.numel())
                dominant_frac = float(counts.max().item()) / float(total_vox)

            bsum = float(torch.sum(b_t[x0 : x0 + target_n, y0 : y0 + target_n, z0 : z0 + target_n]).item())
            key = dominant_frac
            if best is None or key < best[0]:
                best = (key, x0, y0, z0, dominant_frac, unique_count, bsum)

        if best is None:
            x0 = y0 = z0 = 0
            w_np = block_xyz[x0 : x0 + target_n, y0 : y0 + target_n, z0 : z0 + target_n]
            dominant_frac, unique_count = _dominant_unique_numpy(w_np.reshape(-1), int(ignore_label), total_vox)
            bsum = float(np.sum(b_np[x0 : x0 + target_n, y0 : y0 + target_n, z0 : z0 + target_n]))
        else:
            _, x0, y0, z0, dominant_frac, unique_count, bsum = best
    else:
        x_coarse = list(range(0, sx_max + 1, sx))
        y_coarse = list(range(0, sy_max + 1, sy))
        z_coarse = list(range(0, sz_max + 1, sz))
        if len(x_coarse) == 0 or x_coarse[-1] != sx_max:
            x_coarse.append(sx_max)
        if len(y_coarse) == 0 or y_coarse[-1] != sy_max:
            y_coarse.append(sy_max)
        if len(z_coarse) == 0 or z_coarse[-1] != sz_max:
            z_coarse.append(sz_max)

        coarse_scored = []
        for x0 in x_coarse:
            for y0 in y_coarse:
                for z0 in z_coarse:
                    bsum = float(np.sum(b_np[x0 : x0 + target_n, y0 : y0 + target_n, z0 : z0 + target_n]))
                    coarse_scored.append((bsum, int(x0), int(y0), int(z0)))
        coarse_scored.sort(key=lambda t: t[0], reverse=True)
        coarse = [(x, y, z, s) for s, x, y, z in coarse_scored[:topk]]

        refine = set()
        for x0, y0, z0, _ in coarse:
            for xi in range(max(0, x0 - sx), min(sx_max, x0 + sx) + 1):
                for yi in range(max(0, y0 - sy), min(sy_max, y0 + sy) + 1):
                    for zi in range(max(0, z0 - sz), min(sz_max, z0 + sz) + 1):
                        refine.add((int(xi), int(yi), int(zi)))

        best = None
        for x0, y0, z0 in refine:
            w_np = block_xyz[x0 : x0 + target_n, y0 : y0 + target_n, z0 : z0 + target_n]
            dominant_frac, unique_count = _dominant_unique_numpy(w_np.reshape(-1), int(ignore_label), total_vox)
            bsum = float(np.sum(b_np[x0 : x0 + target_n, y0 : y0 + target_n, z0 : z0 + target_n]))
            key = dominant_frac
            if best is None or key < best[0]:
                best = (key, x0, y0, z0, dominant_frac, unique_count, bsum)

        if best is None:
            x0 = y0 = z0 = 0
            w_np = block_xyz[x0 : x0 + target_n, y0 : y0 + target_n, z0 : z0 + target_n]
            dominant_frac, unique_count = _dominant_unique_numpy(w_np.reshape(-1), int(ignore_label), total_vox)
            bsum = float(np.sum(b_np[x0 : x0 + target_n, y0 : y0 + target_n, z0 : z0 + target_n]))
        else:
            _, x0, y0, z0, dominant_frac, unique_count, bsum = best

    cropped = block_xyz[x0 : x0 + target_n, y0 : y0 + target_n, z0 : z0 + target_n]
    best_info = (int(x0), int(y0), int(z0), float(bsum), int(unique_count), float(dominant_frac), int(nx), int(ny), int(nz))
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


def add_fold_soft(h: NoddyHistory, rng: np.random.Generator, name: str, amp_range: Tuple[float, float] = (120.0, 420.0), wl_range: Tuple[float, float] = (1200.0, 2800.0), z_range: Tuple[float, float] = (-1800.0, -220.0), axis_tol: float = 20.0, plunge_range: Tuple[float, float] = (6.0, 35.0)) -> Tuple[float, float, float]:
    amp = float(rng.uniform(*amp_range))
    wl = float(rng.uniform(*wl_range))
    px = float(rng.uniform(-1200.0, 1200.0))
    py = float(rng.uniform(-1200.0, 1200.0))
    pz = float(rng.uniform(*z_range))
    axis_dir = sample_angle_avoid(rng, 0.0, 180.0, avoid=[0.0, 90.0, 180.0], tol=axis_tol)
    plunge = float(rng.uniform(*plunge_range))
    h.add_event("fold", {"name": name, "pos": [px, py, pz], "amplitude": amp, "wavelength": wl, "axis_dir": axis_dir, "plunge": plunge})
    return (px, py, pz)


def add_fault_soft(h: NoddyHistory, rng: np.random.Generator, name: str, slip_range: Tuple[float, float], dip_dir_tol: float = 12.0) -> Tuple[float, float, float]:
    dip_dir = sample_angle_avoid(rng, 0.0, 360.0, avoid=[0.0, 90.0, 180.0, 270.0], tol=dip_dir_tol)
    dip = float(rng.uniform(52.0, 86.0))
    if rng.random() < 0.68:
        pitch = float(rng.choice([-1.0, 1.0]) * rng.uniform(58.0, 88.0))
    else:
        pitch = float(rng.uniform(-28.0, 28.0))
    pz = float(rng.uniform(-1650.0, -170.0))
    slip = float(rng.uniform(float(slip_range[0]), float(slip_range[1])))
    px = float(rng.uniform(-1350.0, 1350.0))
    py = float(rng.uniform(-1350.0, 1350.0))
    h.add_event("fault", {"name": name, "pos": [px, py, pz], "dip_dir": dip_dir, "dip": dip, "pitch": pitch, "slip": slip})
    return (px, py, pz)


def _estimate_unconf_reference_z(h: NoddyHistory, rng: np.random.Generator) -> float:
    _, _, origin_z = h.get_origin()
    _, _, length_z = h.get_extent()
    top_z = float(origin_z)
    length_z = float(length_z)
    bottom_z = top_z - length_z

    pz = bottom_z + float(rng.uniform(0.42, 0.62)) * length_z
    eps = 1e-6
    pz = float(np.clip(pz, bottom_z + eps, top_z - eps))
    return pz


def _estimate_unconf_thicknesses(h: NoddyHistory, target_n: int, n_layers_fixed: Optional[int] = None) -> List[int]:
    _, _, length_z = h.get_extent()
    length_z = float(length_z)

    cube_size = None
    try:
        cs = h.get_cube_size()
        if isinstance(cs, (int, float)):
            cube_size = float(cs)
        elif isinstance(cs, (tuple, list)) and len(cs) > 0:
            cube_size = float(cs[0])
    except Exception:
        cube_size = None

    if cube_size is None or cube_size <= 0:
        cube_size = 50.0

    marker = int(max(1, np.ceil(cube_size)))
    crop_h = float(max(1, int(target_n))) * cube_size
    cap_crop = int(max(2 * marker, np.floor(0.25 * crop_h)))
    cap_model = int(max(2 * marker, np.floor(0.25 * length_z)))
    total_thickness = int(max(2 * marker, min(cap_crop, cap_model)))

    n_layers = int(n_layers_fixed) if (n_layers_fixed is not None and int(n_layers_fixed) > 1) else 2
    per = int(max(marker, np.floor(total_thickness / float(n_layers))))
    thicknesses = [per] * n_layers
    thicknesses[-1] += int(max(0, total_thickness - int(np.sum(thicknesses))))
    return [int(max(marker, t)) for t in thicknesses]


def add_unconformity_soft(h: NoddyHistory, rng: np.random.Generator, name: str, dip_dir_tol: float = 12.0, pz_range: Tuple[float, float] = (-280.0, -230.0), dip_range: Tuple[float, float] = (0.0, 6.0), n_layers_fixed: Optional[int] = None, n_layers_range: Tuple[int, int] = (1, 2), t_range: Tuple[int, int] = (1, 10), max_total_thickness: int = 10, target_n: int = 50, focus_xy: Optional[Tuple[float, float]] = None, focus_pz: Optional[float] = None) -> None:
    if focus_xy is not None:
        ex, ey, _ = h.get_extent()
        hx = 0.18 * float(ex)
        hy = 0.18 * float(ey)
        px = float(np.clip(float(focus_xy[0]) + rng.uniform(-hx, hx), -1500.0, 1500.0))
        py = float(np.clip(float(focus_xy[1]) + rng.uniform(-hy, hy), -1500.0, 1500.0))
    else:
        px = float(rng.uniform(-900.0, 900.0))
        py = float(rng.uniform(-900.0, 900.0))

    if focus_pz is not None:
        _, _, origin_z = h.get_origin()
        _, _, length_z = h.get_extent()
        top_z = float(origin_z)
        bottom_z = top_z - float(length_z)
        pz = float(np.clip(float(focus_pz) + rng.uniform(-0.08 * float(length_z), 0.08 * float(length_z)), bottom_z + 1e-6, top_z - 1e-6))
    else:
        pz = _estimate_unconf_reference_z(h, rng)

    dip_dir = sample_angle_avoid(rng, 0.0, 360.0, avoid=[0.0, 90.0, 180.0, 270.0], tol=dip_dir_tol)
    dip = float(rng.uniform(4.0, 14.0))

    thicknesses = _estimate_unconf_thicknesses(h, int(target_n), n_layers_fixed=n_layers_fixed)
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
    out = {
        "min_nonzero_frac": float(base["min_nonzero_frac"]),
        "min_unique": float(2.0),
        "min_bd": float(base["min_bd"]),
        "min_axis_diff": float(base["min_axis_diff"]),
        "min_axis_ratio": float(base["min_axis_ratio"]),
        "max_dominant_frac": float(s),
    }
    return out


def worker_one(args_tuple):
    (
        i, base_seed, noddy_path,
        crop_from_n, target_n, direct_crop, z_crop,
        smart_crop, smart_stride, smart_z_stride, smart_topk, smart_alpha,
        max_retries, keep_attempt_files, keep_tmp, ignore_label,
        strat_layers, n_tilts, n_folds, n_fold_gentle, n_fold_violent, n_fault_gentle, n_fault_violent,
        fault_slip_gentle_min, fault_slip_gentle_max, fault_slip_violent_min, fault_slip_violent_max,
        fold_amp_gentle_min, fold_amp_gentle_max, fold_amp_violent_min, fold_amp_violent_max,
        fold_wl_gentle_min, fold_wl_gentle_max, fold_wl_violent_min, fold_wl_violent_max,
        slip_scale_gentle, slip_scale_violent, unconf_layers, struct_mode, slip_scale_legacy,
        q_min_nonzero_frac, q_max_dominant_frac, min_strat_in_crop,
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
                strat_n_layers = add_stratigraphy(h, rng, n_layers_fixed=int(strat_layers))
            else:
                strat_n_layers = add_stratigraphy(h, rng)

            struct_points: List[Tuple[float, float, float]] = []

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
            tiered_folds = ((int(n_fold_gentle) >= 0) or (int(n_fold_violent) >= 0)) and (int(n_folds) < 0)
            explicit_folds = (int(n_folds) >= 0) or tiered_folds
            explicit_unconf = int(unconf_layers) >= 0
            should_add_unconf = False
            unconf_fixed_layers = None

            if explicit_folds:
                if tiered_folds:
                    ngf = int(n_fold_gentle) if int(n_fold_gentle) >= 0 else 0
                    nvf = int(n_fold_violent) if int(n_fold_violent) >= 0 else 0
                    for k in range(ngf):
                        pt = add_fold_soft(
                            h, rng, name=f"FOLD_G_{k+1}",
                            amp_range=(float(fold_amp_gentle_min), float(fold_amp_gentle_max)),
                            wl_range=(float(fold_wl_gentle_min), float(fold_wl_gentle_max)),
                        )
                        struct_points.append(pt)
                    for k in range(nvf):
                        pt = add_fold_soft(
                            h, rng, name=f"FOLD_V_{k+1}",
                            amp_range=(float(fold_amp_violent_min), float(fold_amp_violent_max)),
                            wl_range=(float(fold_wl_violent_min), float(fold_wl_violent_max)),
                        )
                        struct_points.append(pt)
                else:
                    for k in range(int(n_folds)):
                        pt = add_fold_soft(h, rng, name=f"FOLD_{k+1}")
                        struct_points.append(pt)

            if explicit_faults:
                ng = int(n_fault_gentle) if int(n_fault_gentle) >= 0 else 0
                nv = int(n_fault_violent) if int(n_fault_violent) >= 0 else 0
                idx = 1
                for _ in range(ng):
                    pt = add_fault_soft(
                        h, rng, name=f"FAULT_{idx}",
                        slip_range=(float(fault_slip_gentle_min), float(fault_slip_gentle_max)),
                    )
                    struct_points.append(pt)
                    idx += 1
                for _ in range(nv):
                    pt = add_fault_soft(
                        h, rng, name=f"FAULT_{idx}",
                        slip_range=(float(fault_slip_violent_min), float(fault_slip_violent_max)),
                    )
                    struct_points.append(pt)
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
                    pt = add_fold_soft(h, rng, name="FOLD_1")
                    struct_points.append(pt)
                if (not explicit_faults) and ("fault" in chosen):
                    pt = add_fault_soft(h, rng, name="FAULT_1", slip_range=(180.0 * float(slip_scale_legacy), 520.0 * float(slip_scale_legacy)))
                    struct_points.append(pt)
                if (not explicit_unconf) and ("unconf" in chosen):
                    should_add_unconf = True

            if should_add_unconf:
                focus_xy = None
                focus_pz = None
                if len(struct_points) > 0:
                    sp = np.array(struct_points, dtype=np.float64)
                    focus_xy = (float(np.mean(sp[:, 0])), float(np.mean(sp[:, 1])))
                    focus_pz = float(np.mean(sp[:, 2]))
                if unconf_fixed_layers is not None:
                    add_unconformity_soft(h, rng, name="UNCONF_1", n_layers_fixed=int(unconf_fixed_layers), target_n=int(target_n), focus_xy=focus_xy, focus_pz=focus_pz)
                else:
                    add_unconformity_soft(h, rng, name="UNCONF_1", target_n=int(target_n), focus_xy=focus_xy, focus_pz=focus_pz)

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
                    if (torch is not None) and torch.cuda.is_available():
                        torch.cuda.set_device(int(cuda_device))
                    block_xyz, smart_info = smart_crop_direct_xyz(
                        block_xyz,
                        int(target_n),
                        str(z_crop),
                        int(smart_stride),
                        int(ignore_label),
                        float(smart_alpha),
                        int(smart_z_stride),
                        int(smart_topk),
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

            if use_torch and (torch_device is not None) and (torch_device.type == "cuda"):
                vals = torch.from_numpy(block_zyx).to(device=torch_device, dtype=torch.int64, non_blocking=True).reshape(-1)
                vals = vals[vals != int(ignore_label)]
                if vals.numel() == 0:
                    dominant_frac = 1.0
                    n_strat_present = 0
                else:
                    _, counts = torch.unique(vals, return_counts=True)
                    dominant_frac = float(counts.max().item()) / float(block_zyx.size)
                    strat_vals = vals[(vals >= 1) & (vals <= int(strat_n_layers))]
                    if strat_vals.numel() == 0:
                        n_strat_present = 0
                    else:
                        n_strat_present = int(torch.unique(strat_vals).numel())
            else:
                vals_np = block_zyx.reshape(-1)
                vals_np = vals_np[vals_np != int(ignore_label)]
                if vals_np.size == 0:
                    dominant_frac = 1.0
                    n_strat_present = 0
                else:
                    _, counts_np = np.unique(vals_np, return_counts=True)
                    dominant_frac = float(np.max(counts_np)) / float(block_zyx.size)
                    strat_vals_np = vals_np[(vals_np >= 1) & (vals_np <= int(strat_n_layers))]
                    n_strat_present = int(np.unique(strat_vals_np).size) if strat_vals_np.size > 0 else 0

            if dominant_frac > float(q_max_dominant_frac):
                raise RuntimeError(f"quality_reject dominant_frac={dominant_frac:.6f} > {float(q_max_dominant_frac):.6f}")
            if int(n_strat_present) < int(min_strat_in_crop):
                raise RuntimeError(f"quality_reject n_strat_present={int(n_strat_present)} < {int(min_strat_in_crop)}")

            if int(smart_crop) == 1 and smart_info is not None:
                x0, y0, z0, score, uq, dfrac, fx, fy, fz = smart_info
                print(f"[INFO] smart_crop full_dims=({fx},{fy},{fz}) origin=({x0},{y0},{z0}) dominant_frac={dfrac:.6f} unique={uq}")

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
    p.add_argument("--smart_z_stride", type=int, default=2, help="Smart-crop Z search stride in voxels.")
    p.add_argument("--smart_topk", type=int, default=64, help="Top-K coarse smart-crop candidates for local refinement.")
    p.add_argument("--smart_alpha", type=float, default=5.0, help="Smart-crop score alpha: unique_count - alpha * dominant_frac.")

    p.add_argument("--struct_mode", type=str, default="2of3", choices=["2of3", "all", "fault_only", "fold_only", "unconf_only", "none"], help="Legacy switch: which of {fault,fold,unconf} to include when explicit counts are not set.")
    p.add_argument("--slip_scale", type=float, default=1.0, help="Legacy fault slip scale (used when --n_fault_gentle/--n_fault_violent are not set).")

    p.add_argument("--strat_layers", type=int, default=-1, help="Fixed stratigraphy layer count. -1 keeps legacy random (6–8).")
    p.add_argument("--n_tilts", type=int, default=-1, help="Number of tilts. -1 keeps legacy (1 + optional 2nd).")
    p.add_argument("--n_folds", type=int, default=-1, help="Number of folds. -1 keeps legacy (0/1 depending on struct_mode).")
    p.add_argument("--n_fold_gentle", type=int, default=-1, help="Number of gentle folds. -1 keeps legacy fold logic.")
    p.add_argument("--n_fold_violent", type=int, default=-1, help="Number of violent folds. -1 keeps legacy fold logic.")

    p.add_argument("--n_fault_gentle", type=int, default=-1, help="Number of gentle faults. -1 keeps legacy fault logic.")
    p.add_argument("--n_fault_violent", type=int, default=-1, help="Number of violent faults. -1 keeps legacy fault logic.")
    p.add_argument("--slip_scale_gentle", type=float, default=0.6, help="Slip scale for gentle faults.")
    p.add_argument("--slip_scale_violent", type=float, default=1.2, help="Slip scale for violent faults.")

    p.add_argument("--fault_slip_gentle_min", type=float, default=80.0)
    p.add_argument("--fault_slip_gentle_max", type=float, default=180.0)
    p.add_argument("--fault_slip_violent_min", type=float, default=320.0)
    p.add_argument("--fault_slip_violent_max", type=float, default=700.0)

    p.add_argument("--fold_amp_gentle_min", type=float, default=60.0)
    p.add_argument("--fold_amp_gentle_max", type=float, default=160.0)
    p.add_argument("--fold_amp_violent_min", type=float, default=280.0)
    p.add_argument("--fold_amp_violent_max", type=float, default=600.0)
    p.add_argument("--fold_wl_gentle_min", type=float, default=1400.0)
    p.add_argument("--fold_wl_gentle_max", type=float, default=2400.0)
    p.add_argument("--fold_wl_violent_min", type=float, default=700.0)
    p.add_argument("--fold_wl_violent_max", type=float, default=1300.0)

    p.add_argument("--unconf_layers", type=int, default=-1, help="Fixed number of unconformity layers. 0 disables unconf. -1 keeps legacy random (1–3) if unconf is used.")

    p.add_argument("--quality", type=float, default=0.5, help="Quality gate dominant-fraction threshold in [0,1].")
    p.add_argument("--max_candidates", type=int, default=0, help="Max total candidate ids to try (0 = no limit). Useful to prevent infinite loops when quality is too strict.")

    p.add_argument("--max_retries", type=int, default=12)
    p.add_argument("--fail_fast", type=int, default=0)
    p.add_argument("--keep_attempt_files", type=int, default=0)
    p.add_argument("--keep_tmp", type=int, default=0)

    p.add_argument("--ignore_label", type=int, default=0)
    p.add_argument("--min_strat_in_crop", type=int, default=0,
                   help="Minimum distinct strat labels required in final crop (excluding ignore label).")

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

    crop_from_n = int(args.crop_from_n)
    target_n = int(args.target_n)
    if target_n <= 0:
        raise ValueError("--target_n must be > 0")
    if crop_from_n > 0 and target_n > crop_from_n:
        raise ValueError("--target_n (m) cannot be larger than --crop_from_n (n)")

    # gentle vs violent are strictly disjoint by construction
    if not (float(args.fault_slip_gentle_min) < float(args.fault_slip_gentle_max)):
        raise ValueError("Invalid gentle fault slip range: require min < max")
    if not (float(args.fault_slip_violent_min) < float(args.fault_slip_violent_max)):
        raise ValueError("Invalid violent fault slip range: require min < max")
    if not (float(args.fault_slip_gentle_max) < float(args.fault_slip_violent_min)):
        raise ValueError("Fault slip ranges must be strictly disjoint: gentle_max < violent_min")

    if not (float(args.fold_amp_gentle_min) < float(args.fold_amp_gentle_max)):
        raise ValueError("Invalid gentle fold amplitude range: require min < max")
    if not (float(args.fold_amp_violent_min) < float(args.fold_amp_violent_max)):
        raise ValueError("Invalid violent fold amplitude range: require min < max")
    if not (float(args.fold_amp_gentle_max) < float(args.fold_amp_violent_min)):
        raise ValueError("Fold amplitude ranges must be strictly disjoint: gentle_max < violent_min")

    if not (float(args.fold_wl_gentle_min) < float(args.fold_wl_gentle_max)):
        raise ValueError("Invalid gentle fold wavelength range: require min < max")
    if not (float(args.fold_wl_violent_min) < float(args.fold_wl_violent_max)):
        raise ValueError("Invalid violent fold wavelength range: require min < max")

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
    print(f"[INFO] Smart crop: enabled={int(args.smart_crop)} stride={int(args.smart_stride)} z_stride={int(args.smart_z_stride)} topk={int(args.smart_topk)} alpha={float(args.smart_alpha):.3f}")
    print(f"[INFO] Target shape: (N_ok={target_ok}, {target_n}, {target_n}, {target_n}) int16, order=NZYX")
    print(f"[INFO] Noddy: {noddy_path}")
    print(f"[INFO] Workers={int(args.workers)}, seed={int(args.seed)}")
    print(f"[INFO] Geology controls: strat_layers={args.strat_layers} n_tilts={args.n_tilts} n_folds={args.n_folds} n_fold_gentle={args.n_fold_gentle} n_fold_violent={args.n_fold_violent} n_fault_gentle={args.n_fault_gentle} n_fault_violent={args.n_fault_violent} unconf_layers={args.unconf_layers}")
    print(f"[INFO] Legacy struct_mode={args.struct_mode} slip_scale={float(args.slip_scale):.3f}")
    print(f"[INFO] Quality gate strength={float(args.quality):.2f} => dominant_frac_max={eff_q['max_dominant_frac']:.6f} min_nonzero_frac={eff_q['min_nonzero_frac']:.6f} min_unique={int(eff_q['min_unique'])} min_strat_in_crop={int(args.min_strat_in_crop)}")
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
                int(args.smart_crop), int(args.smart_stride), int(args.smart_z_stride), int(args.smart_topk), float(args.smart_alpha),
                int(args.max_retries), int(args.keep_attempt_files), int(args.keep_tmp), int(args.ignore_label),
                int(args.strat_layers), int(args.n_tilts), int(args.n_folds), int(args.n_fold_gentle), int(args.n_fold_violent), int(args.n_fault_gentle), int(args.n_fault_violent),
                float(args.fault_slip_gentle_min), float(args.fault_slip_gentle_max), float(args.fault_slip_violent_min), float(args.fault_slip_violent_max),
                float(args.fold_amp_gentle_min), float(args.fold_amp_gentle_max), float(args.fold_amp_violent_min), float(args.fold_amp_violent_max),
                float(args.fold_wl_gentle_min), float(args.fold_wl_gentle_max), float(args.fold_wl_violent_min), float(args.fold_wl_violent_max),
                float(args.slip_scale_gentle), float(args.slip_scale_violent), int(args.unconf_layers), str(args.struct_mode), float(args.slip_scale),
                float(eff_q["min_nonzero_frac"]), float(eff_q["max_dominant_frac"]), int(args.min_strat_in_crop),
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
        f.write(f"smart_crop={int(args.smart_crop)} smart_stride={int(args.smart_stride)} smart_z_stride={int(args.smart_z_stride)} smart_topk={int(args.smart_topk)} smart_alpha={float(args.smart_alpha):.6f}\n")
        f.write(f"qgate_torch={int(args.qgate_torch)} qgate_device={str(args.qgate_device)} cuda_device={int(args.cuda_device)} cuda_sync={int(args.cuda_sync)}\n")
        f.write("effective_thresholds=" + ",".join([f"{k}={eff_q[k]}" for k in ["min_nonzero_frac", "min_unique", "max_dominant_frac"]]) + "\n")
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
