# PyNoddy Information Entropy Planner

This repository contains an end-to-end, fully local pipeline for planning borehole drilling locations that maximise the reduction of geological uncertainty in 3-D block models generated with the RWTH edition of [PyNoddy](https://www.rwth-aachen.de).

All computations are performed offline. The workflow samples geological event parameters, generates an ensemble of plausible block models at a resolution of **50×50×50**, quantifies uncertainty via Shannon entropy, and trains a reinforcement learning (RL) agent that selects ordered drill locations that maximise the expected entropy reduction.

## Repository layout

```
project_root/
  data/                      # generated locally via gen_dataset.py
  models/                    # RL checkpoints
  outputs/                   # intermediate outputs (entropy, anisotropy, plans)
  src/
    gen_dataset.py           # Monte Carlo sampling + PyNoddy execution
    build_entropy_from_ensemble.py
    build_aniso_kernel.py
    apply_influence.py
    rl_env.py
    rl_train.py
    rl_infer.py
    plan_drilling.py
    run_end_to_end.py
  run_end_to_end.bat         # Windows batch runner (ASCII only)
  run_rl_only.bat            # Windows batch runner (ASCII only)
  environment.yml            # Conda environment specification
  README.md                  # This file
```

## Requirements

1. A working installation of the RWTH PyNoddy package (either on the Python path or accessible via a path passed to the scripts).
2. Python 3.10 and the packages listed in `environment.yml`. Create the environment:

```bash
conda env create -f environment.yml
conda activate entropy50
# Install PyTorch CPU build (optional, required for RL)
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

## Quick start (Linux/macOS)

```bash
python src/run_end_to_end.py --num-models 50 --n-drillholes 5
```

This convenience script orchestrates dataset generation, entropy computation, anisotropy estimation, RL training (or loading), and final drill planning. Outputs are written under `outputs/`.

## Windows batch runners

Two ASCII-only batch files provide a one-click experience:

- `run_end_to_end.bat` – generates a small dataset, trains/loads the RL agent, and produces a drilling plan.
- `run_rl_only.bat` – runs RL training only.

Both scripts pause at the end to avoid immediate console closure.

## Script overview

### Dataset generation – `src/gen_dataset.py`

Samples geological event parameters from user-specified ranges (YAML/JSON) and calls PyNoddy locally to build Monte Carlo ensembles. The script fills a lightweight history template in memory, runs PyNoddy, and only persists the `.g00` metadata file and `.g12` lithology volume (50×50×50) for each Monte Carlo realisation.

### Entropy estimation – `src/build_entropy_from_ensemble.py`

Given an ensemble of `.g12` volumes and the associated `.g00` class definitions, this script builds categorical probability tensors and the Shannon entropy volume. Minimum intensity projections along X/Y/Z are exported as PNG quicklooks.

### Anisotropic kernel estimation – `src/build_aniso_kernel.py`

Analyzes one or more realisations to infer the dominant stratigraphic orientation. Produces directional kernels (`aniso_kernels.json`) for fast entropy updates, plus an orientation rose plot.

### Fast entropy updates – `src/apply_influence.py`

Approximates entropy reduction for hypothetical drillholes using anisotropic convolution along estimated strata. Supports minimum separation constraints and can be reused inside planning algorithms.

### Reinforcement learning – `src/rl_env.py`, `src/rl_train.py`, `src/rl_infer.py`

Implements an entropy-reduction environment and a CNN-based DQN agent. Training statistics and the final policy are stored under `models/`.

### Planning – `src/plan_drilling.py`

Computes entropy for a new case, runs the RL policy (with optional beam-search fallback), and outputs the ordered drill plan in CSV format. Overlays and JSON stats summarise per-step entropy reductions.

### End-to-end demo – `src/run_end_to_end.py`

Bundles all stages: dataset generation, entropy calculation, anisotropy estimation, RL training, and drilling plan inference. Parameters expose seeds, counts, and file locations to enable lightweight experiments or large-scale runs.

## Logging & reproducibility

All major scripts support `--seed` arguments and write lightweight JSON logs for reproducibility. Progress is streamed to the console via `tqdm` or standard print statements.

## Data outputs

The pipeline produces the following artefacts:

- `data/` – Monte Carlo realisations with `.g00/.g12` pairs.
- `outputs/entropy/` – entropy volumes (`entropy_volume.npy`), statistics, and quicklooks.
- `outputs/aniso/` – anisotropy kernels (`aniso_kernels.json`) and rose plot.
- `models/` – trained RL agent (`model_final.pt`) and training curves.
- `outputs/plans/` – drilling plans (`picks.csv`, `picks_overlay.png`, `plan_stats.json`).

## Notes

- Gravity and magnetic outputs are not used anywhere in the codebase.
- Grid resolution is hard-coded to 50×50×50 (configurable via flags when required) to match the acceptance criteria.
- All scripts work offline and can be scaled up via command-line parameters.
- If PyNoddy resides in a non-standard location, pass `--pynoddy-path` to the relevant scripts or set the `PYNODDY_PATH` environment variable.

## License

See [LICENSE](LICENSE).
