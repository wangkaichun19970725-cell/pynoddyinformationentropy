"""Utility helpers for PyNoddy based workflows.

The functions in this module intentionally avoid importing the heavy
`pynoddy` module at import time. Instead, call :func:`ensure_pynoddy`
before interacting with PyNoddy-specific functionality. This keeps the
modules usable in environments where PyNoddy is unavailable (e.g. during
lightweight unit tests) while still providing informative error messages
when the dependency is missing.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore


@dataclass
class GridSpec:
    """Simple container describing a 3-D grid."""

    nx: int
    ny: int
    nz: int
    origin: np.ndarray
    spacing: np.ndarray

    @classmethod
    def from_metadata(cls, meta: Dict[str, Any]) -> "GridSpec":
        origin = np.asarray(meta.get("origin", [0.0, 0.0, 0.0]), dtype=float)
        spacing = np.asarray(meta.get("spacing", [1.0, 1.0, 1.0]), dtype=float)
        return cls(
            nx=int(meta.get("nx", 50)),
            ny=int(meta.get("ny", 50)),
            nz=int(meta.get("nz", 50)),
            origin=origin,
            spacing=spacing,
        )

    def index_to_world(self, indices: np.ndarray) -> np.ndarray:
        """Converts (N, 3) array of voxel indices to world coordinates."""

        return self.origin + indices * self.spacing


class PynoddyNotInstalledError(ImportError):
    """Raised when PyNoddy is required but not importable."""


def ensure_pynoddy(pynoddy_path: Optional[str] = None) -> None:
    """Ensure the PyNoddy package is importable.

    Parameters
    ----------
    pynoddy_path:
        Optional explicit path to the PyNoddy source tree. When provided it is
        appended to ``sys.path``.
    """

    path_candidate = pynoddy_path or os.environ.get("PYNODDY_PATH")
    if path_candidate:
        path = os.path.abspath(os.path.expanduser(path_candidate))
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        import pynoddy  # noqa: F401  # pragma: no cover - imported for side effects
    except ImportError as exc:  # pragma: no cover - requires local install
        raise PynoddyNotInstalledError(
            "PyNoddy could not be imported. Set PYNODDY_PATH or use --pynoddy-path "
            "to point to the local RWTH PyNoddy installation."
        ) from exc


def load_yaml_or_json(path: Path) -> Dict[str, Any]:
    """Load a configuration file that may be YAML or JSON."""

    text = path.read_text()
    if path.suffix.lower() in {".json"}:
        return json.loads(text)
    if path.suffix.lower() in {".yml", ".yaml"}:
        if yaml is None:  # pragma: no cover - optional dependency
            raise RuntimeError("pyyaml is required to parse YAML configuration files")
        return yaml.safe_load(text)
    # Heuristic: try JSON first, fall back to YAML if available
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        if yaml is None:  # pragma: no cover - optional dependency
            raise
        return yaml.safe_load(text)


def read_g00_classes(g00_path: Path) -> Dict[int, str]:
    """Parse a minimal subset of a .g00 file to obtain class names.

    The RWTH PyNoddy ``.g00`` format is human-readable. We parse lines that
    contain ``CODE`` and ``NAME`` pairs, returning a mapping from integer code to
    lithology name.
    """

    classes: Dict[int, str] = {}
    current_code: Optional[int] = None
    for line in g00_path.read_text().splitlines():
        line = line.strip()
        if line.upper().startswith("CODE"):
            try:
                current_code = int(line.split()[1])
            except Exception:
                current_code = None
        elif line.upper().startswith("NAME") and current_code is not None:
            name = " ".join(line.split()[1:])
            classes[current_code] = name
            current_code = None
    if not classes:
        raise ValueError(f"No lithology classes were found in {g00_path}")
    return classes


def load_block_from_g12(g12_path: Path) -> np.ndarray:
    """Load a 3-D lithology block using PyNoddy's output helpers."""

    ensure_pynoddy()
    from pynoddy.output import NoddyOutput  # type: ignore

    base = g12_path.with_suffix("")
    noddy_output = NoddyOutput(str(base))
    block = noddy_output.block
    if block is None:
        raise RuntimeError(f"Failed to load block data from {g12_path}")
    data = np.asarray(block, dtype=np.int32)
    if data.ndim != 3:
        raise ValueError(f"Unexpected block shape {data.shape}; expected 3-D")
    return data


def save_numpy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def list_case_directories(root: Path) -> List[Path]:
    """Return sorted case directories under the given root."""

    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir()])


def split_indices(num_items: int, train_ratio: float, val_ratio: float) -> Dict[str, List[int]]:
    """Create train/val/test splits for a dataset size."""

    indices = np.arange(num_items)
    train_end = int(num_items * train_ratio)
    val_end = train_end + int(num_items * val_ratio)
    split = {
        "train": indices[:train_end].tolist(),
        "val": indices[train_end:val_end].tolist(),
        "test": indices[val_end:].tolist(),
    }
    return split


def format_case_name(index: int) -> str:
    return f"model_{index:05d}"
