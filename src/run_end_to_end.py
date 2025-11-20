"""One-click end-to-end demonstration pipeline."""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import List


def run(cmd: List[str], env: dict | None = None) -> None:
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-models", type=int, default=10)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--n-drillholes", type=int, default=5)
    parser.add_argument("--history-template", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=False)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pynoddy-path", type=str, default=None)
    parser.add_argument("--train-entropy-cases", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_dir = args.data_dir
    outputs_dir = args.outputs
    entropy_dir = outputs_dir / "entropy"
    aniso_dir = outputs_dir / "aniso"
    plans_dir = outputs_dir / "plans"

    # 1) Dataset generation
    base_env = os.environ.copy()
    if args.pynoddy_path:
        base_env["PYNODDY_PATH"] = args.pynoddy_path

    gen_cmd = [
        "python",
        "src/gen_dataset.py",
        "--history-template",
        str(args.history_template),
        "--out",
        str(data_dir),
        "--num-models",
        str(args.num_models),
        "--ensemble-size",
        str(args.ensemble_size),
        "--seed",
        str(args.seed),
    ]
    if args.config:
        gen_cmd += ["--config", str(args.config)]
    if args.pynoddy_path:
        gen_cmd += ["--pynoddy-path", args.pynoddy_path]
    run(gen_cmd, env=base_env)

    # 2) Build entropy for selected training cases
    entropy_dir.mkdir(parents=True, exist_ok=True)
    train_cases = sorted((data_dir / "train").glob("model_*"))[: args.train_entropy_cases]
    if not train_cases:
        raise RuntimeError("No training cases found. Ensure dataset generation succeeded.")
    entropy_paths: List[Path] = []
    for case in train_cases:
        out_case = entropy_dir / case.name
        run([
            "python",
            "src/build_entropy_from_ensemble.py",
            "--case",
            str(case),
            "--out",
            str(out_case),
        ], env=base_env)
        entropy_paths.append(out_case / "entropy_volume.npy")

    # 3) Estimate anisotropy from the first case
    first_case = train_cases[0]
    first_g12_list = sorted(first_case.glob("realisation_*/case.g12"))
    if not first_g12_list:
        raise RuntimeError("No g12 volumes found in first training case")
    first_g12 = first_g12_list[0]
    run([
        "python",
        "src/build_aniso_kernel.py",
        "--g12",
        str(first_g12),
        "--out",
        str(aniso_dir),
    ], env=base_env)

    # 4) Train RL agent
    rl_cmd = [
        "python",
        "src/rl_train.py",
        "--entropy",
    ] + [str(path) for path in entropy_paths]
    rl_cmd += [
        "--kernel-config",
        str(aniso_dir / "aniso_kernels.json"),
        "--save-dir",
        str(args.models),
        "--epochs",
        "200",
        "--max-steps",
        str(args.n_drillholes),
    ]
    run(rl_cmd, env=base_env)

    # 5) Plan drilling on first test case
    test_cases = sorted((data_dir / "test").glob("model_*"))
    if not test_cases:
        raise RuntimeError("No test cases found for planning")
    test_case = test_cases[0]
    run([
        "python",
        "src/plan_drilling.py",
        "--ensemble",
        str(test_case),
        "--kernel-config",
        str(aniso_dir / "aniso_kernels.json"),
        "--model",
        str(args.models / "model_final.pt"),
        "--n",
        str(args.n_drillholes),
        "--out",
        str(plans_dir),
    ], env=base_env)


if __name__ == "__main__":
    main()
