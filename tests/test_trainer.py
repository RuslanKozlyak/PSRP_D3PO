from __future__ import annotations

import math
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from psrp_mtppo.baselines import evaluate_baseline, run_baseline_episode
from psrp_mtppo.config import EnvironmentConfig, ModelConfig, TrainingConfig
from psrp_mtppo.experiments import run_joint_experiment
from psrp_mtppo.ga import GAConfig


SMALL_GA = GAConfig(population_size=10, generations=3, seed=1)


class TrainerTests(unittest.TestCase):
    def test_joint_training_smoke(self) -> None:
        trainer, history = run_joint_experiment(
            EnvironmentConfig(num_retailers=5, horizon_days=3, seed=17),
            ModelConfig(attention_heads=4, attention_layers=1),
            TrainingConfig(
                train_batch_size=6,
                minibatch_size=3,
                ppo_epochs=1,
                training_iterations=1,
                evaluation_episodes=1,
                num_envs=2,
                device="cpu",
                seed=17,
            ),
        )
        metrics = trainer.evaluate(episodes=1)
        self.assertEqual(len(history), 1)
        self.assertIn("eval_sum_cost", history.columns)
        self.assertTrue(math.isfinite(float(metrics["eval_sum_cost"])))

    def test_ga_baselines_smoke(self) -> None:
        config = EnvironmentConfig(num_retailers=5, horizon_days=3, seed=19)
        for name in ("greedy", "ga_inv", "ga_vrp", "ga_irp"):
            results = evaluate_baseline(config, baseline=name, episodes=1, ga_config=SMALL_GA)
            self.assertEqual(len(results), 1)
            self.assertGreaterEqual(float(results.iloc[0]["sum_cost"]), 0.0)

    def test_episode_trace_shapes(self) -> None:
        config = EnvironmentConfig(num_retailers=5, horizon_days=3, seed=23)
        trainer, _history = run_joint_experiment(
            config,
            ModelConfig(attention_heads=4, attention_layers=1),
            TrainingConfig(
                train_batch_size=6,
                minibatch_size=3,
                ppo_epochs=1,
                training_iterations=1,
                evaluation_episodes=1,
                num_envs=2,
                device="cpu",
                seed=23,
            ),
        )
        mtppo_summary, mtppo_trace = trainer.rollout_episode(seed=123)
        baseline_summary, baseline_trace = run_baseline_episode(
            config,
            baseline="greedy",
            seed=123,
        )

        self.assertEqual(len(mtppo_trace["daily"]), config.horizon_days)
        self.assertEqual(mtppo_trace["ending_inventory"].shape, (config.horizon_days, config.num_retailers))
        self.assertEqual(len(baseline_trace["daily"]), config.horizon_days)
        self.assertGreaterEqual(float(mtppo_summary["sum_cost"]), 0.0)
        self.assertGreaterEqual(float(baseline_summary["sum_cost"]), 0.0)


if __name__ == "__main__":
    unittest.main()
