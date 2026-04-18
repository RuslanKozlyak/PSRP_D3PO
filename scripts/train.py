from __future__ import annotations

import sys

try:
    import hydra
    from omegaconf import DictConfig
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Hydra is not installed. Activate the project venv and install requirements first."
    ) from exc

from src.train.notebook import build_components


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    _, _, _, trainer = build_components(cfg)
    trainer.train()


if __name__ == "__main__":
    sys.exit(main())
