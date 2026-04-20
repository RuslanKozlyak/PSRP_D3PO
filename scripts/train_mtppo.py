from __future__ import annotations

import argparse
from dataclasses import replace

from psrp_mtppo.config import EnvironmentConfig, ModelConfig, TrainingConfig
from psrp_mtppo.experiments import plot_training_history, run_joint_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MTPPO on IRP-VMI.")
    parser.add_argument("--retailers", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    env_config = EnvironmentConfig(num_retailers=args.retailers)
    train_config = replace(TrainingConfig(device=args.device), training_iterations=args.iterations)
    trainer, history = run_joint_experiment(env_config, ModelConfig(), train_config)
    print(history.tail(1).to_string(index=False))
    figure = plot_training_history(history)
    figure.savefig("training_history.png", dpi=150, bbox_inches="tight")
    print(trainer.evaluate())


if __name__ == "__main__":
    main()
