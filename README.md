# MP-PSRP RL

Notebook-first modular scaffold for the multi-product petrol station replenishment problem.

## Python

This project is prepared for Python `3.11`.

The working virtual environment created in this workspace is:

```powershell
.venv311
```

## Install

```powershell
.\.venv311\Scripts\python.exe -m pip install -r requirements.txt
.\.venv311\Scripts\python.exe -m ipykernel install --user --name mp-psrp-rl
```

The quickstart notebook bootstraps the repo root into `sys.path`, so it works from `notebooks/` without an editable install. If you want `src` importable globally from the virtual environment, you can optionally try:

```powershell
.\.venv311\Scripts\python.exe -m pip install --no-build-isolation -e .
```

## Notebook usage

```python
from src.train.notebook import load_config, build_components

cfg = load_config(["env=small", "algo=ppo"])
env, policy, algo, trainer = build_components(cfg)
history = trainer.train()
```

The main notebook is [notebooks/00_quickstart.ipynb](/d:/PythonProjects/PSRP_D3PO/notebooks/00_quickstart.ipynb). It now includes:

- one shared synthetic instance generated through `mppsrp_nir`
- full algorithm suite comparison for `PPO`, `PPO-Lagrangian`, and `D3PO`
- clear `untrained policy` vs `trained policy` comparisons
- day-by-day station fill-ratio dynamics starting from day `0`
- final trained-policy comparison against `CP-SAT`
- experiment presets for `smoke`, `benchmark`, and `full`

[notebooks/01_cpsat_comparison.ipynb](/d:/PythonProjects/PSRP_D3PO/notebooks/01_cpsat_comparison.ipynb) is kept only as a short legacy redirect to the main notebook.

## Experiment presets

- `experiment=smoke`: quick notebook sanity run
- `experiment=benchmark`: moderate comparison run for local evaluation
- `experiment=full`: longer training preset for fuller experiments and seed sweeps

Example:

```powershell
python scripts/train.py env=medium experiment=benchmark algo=ppo
python scripts/train.py env=medium experiment=full algo=d3po seed=19
```
