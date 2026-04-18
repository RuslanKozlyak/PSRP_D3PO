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

- instance layout visualization
- inventory heatmaps
- action-mask inspection
- baseline episode rollout tracing
- short training-run charts
- before/after episode comparison
