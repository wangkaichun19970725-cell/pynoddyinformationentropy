"""Monte Carlo dataset generation with RWTH PyNoddy."""
from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
from tqdm import tqdm

from utils_io import ensure_pynoddy, format_case_name, load_yaml_or_json, split_indices


@dataclass
class ParameterSpec:
    name: str
    minimum: float
    maximum: float
    distribution: str = "uniform"

    @classmethod
    def from_mapping(cls, name: str, mapping: Mapping[str, Any]) -> "ParameterSpec":
        if "min" not in mapping or "max" not in mapping:
            raise ValueError(f"Parameter '{name}' must define 'min' and 'max'")
        return cls(
            name=name,
            minimum=float(mapping["min"]),
            maximum=float(mapping["max"]),
            distribution=str(mapping.get("distribution", "uniform")).lower(),
        )

    def sample(self, rng: np.random.Generator) -> float:
        if self.distribution == "uniform":
            return float(rng.uniform(self.minimum, self.maximum))
        if self.distribution == "normal":
            mean = (self.minimum + self.maximum) / 2.0
            std = (self.maximum - self.minimum) / 6.0
            value = rng.normal(mean, std)
            return float(np.clip(value, self.minimum, self.maximum))
        raise ValueError(f"Unsupported distribution '{self.distribution}'")


def build_history_from_template(template: str, values: Mapping[str, Any]) -> str:
    return template.format(**values)


def write_history(history_text: str, path: Path) -> None:
    path.write_text(history_text)


def run_pynoddy(history_path: Path, output_basename: Path, grid_size: int = 50) -> None:
    ensure_pynoddy()
    from pynoddy import compute_model  # type: ignore
    from pynoddy.history import GeologicalHistory  # type: ignore

    history = GeologicalHistory()
    history.read_history(str(history_path))
    compute_model(history, str(output_basename), resolution=(grid_size, grid_size, grid_size))


def copy_outputs(simulation_base: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for suffix in (".g00", ".g12"):
        src = simulation_base.with_suffix(suffix)
        dst = destination / f"case{suffix}"
        dst.write_bytes(src.read_bytes())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=False, help="YAML/JSON parameter configuration")
    parser.add_argument("--history-template", type=Path, required=True, help="Base .his template with {placeholders}")
    parser.add_argument("--out", type=Path, default=Path("data"), help="Output dataset directory")
    parser.add_argument("--num-models", type=int, default=100, help="Number of base cases per split")
    parser.add_argument("--ensemble-size", type=int, default=5, help="Number of Monte Carlo realisations per base case")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pynoddy-path", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_pynoddy(args.pynoddy_path)
    rng = np.random.default_rng(args.seed)

    if args.config:
        config = load_yaml_or_json(args.config)
        param_specs = {
            name: ParameterSpec.from_mapping(name, spec)
            for name, spec in config.get("parameters", {}).items()
        }
    else:
        param_specs = {}

    template_text = args.history_template.read_text()
    splits = split_indices(args.num_models, args.train_ratio, args.val_ratio)

    for split_name, indices in splits.items():
        for index in tqdm(indices, desc=f"{split_name} generation"):
            case_name = format_case_name(index)
            case_dir = args.out / split_name / case_name
            case_dir.mkdir(parents=True, exist_ok=True)
            for ensemble_idx in range(args.ensemble_size):
                sampled_values = {
                    name: spec.sample(rng) for name, spec in param_specs.items()
                }
                history_text = build_history_from_template(template_text, sampled_values)
                real_dir = case_dir / f"realisation_{ensemble_idx:04d}"
                real_dir.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_path = Path(tmpdir)
                    history_path = tmp_path / "case.his"
                    write_history(history_text, history_path)
                    simulation_base = tmp_path / "case"
                    run_pynoddy(history_path, simulation_base)
                    copy_outputs(simulation_base, real_dir)
                log = {
                    "base_index": int(index),
                    "ensemble_index": int(ensemble_idx),
                    "split": split_name,
                    "parameters": sampled_values,
                }
                (real_dir / "generation_log.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
