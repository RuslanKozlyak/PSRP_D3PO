from __future__ import annotations

import sys
from pathlib import Path

try:
    import hydra
    import numpy as np
    from omegaconf import DictConfig, OmegaConf
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Hydra and numpy are required. Activate the project venv and install requirements first."
    ) from exc

from src.env.instance import generate_random_instance


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    env_cfg = dict(cfg_dict["env"])
    instance = generate_random_instance(env_cfg, seed=int(cfg_dict.get("seed", env_cfg.get("seed", 0))))
    output_dir = Path("artifacts") / "instances"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{env_cfg['name']}.npz"
    np.savez_compressed(
        output_path,
        station_coords=instance.station_coords,
        station_capacity=instance.station_capacity,
        initial_inventory=instance.initial_inventory,
        daily_demand=instance.daily_demand,
        compatibility=instance.station_vehicle_compatibility,
    )
    print(output_path)


if __name__ == "__main__":
    sys.exit(main())
