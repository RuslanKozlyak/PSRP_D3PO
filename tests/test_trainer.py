from __future__ import annotations

import math
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from psrp_mtppo.baselines import evaluate_heuristic_baseline
from psrp_mtppo.baselines import run_heuristic_episode
from psrp_mtppo.config import EnvironmentConfig, ModelConfig, TrainingConfig
from psrp_mtppo.experiments import run_joint_experiment


class TrainerTests(unittest.TestCase):
    def test_joint_training_smoke(self) -> None:
        trainer, history = run_joint_experiment(
            EnvironmentConfig(num_retailers=5, horizon_days=4, seed=17),
            ModelConfig(attention_heads=4),
            TrainingConfig(
                train_batch_size=8,
                minibatch_size=4,
                ppo_epochs=1,
                training_iterations=1,
                evaluation_episodes=1,
                seed=17,
            ),
        )
        metrics = trainer.evaluate(episodes=1)
        self.assertEqual(len(history), 1)
        self.assertIn("eval_sum_cost", history.columns)
        self.assertTrue(math.isfinite(float(metrics["eval_sum_cost"])))
        self.assertGreater(float(metrics["eval_route_distance"]), 0.0)

    def test_heuristic_baseline_smoke(self) -> None:
        results = evaluate_heuristic_baseline(
            EnvironmentConfig(num_retailers=5, horizon_days=4, seed=19),
            episodes=1,
            route_solver="greedy",
        )
        self.assertEqual(len(results), 1)
        self.assertGreater(float(results.iloc[0]["sum_cost"]), 0.0)

    def test_episode_trace_shapes(self) -> None:
        config = EnvironmentConfig(num_retailers=5, horizon_days=4, seed=23)
        trainer, _history = run_joint_experiment(
            config,
            ModelConfig(attention_heads=4),
            TrainingConfig(
                train_batch_size=8,
                minibatch_size=4,
                ppo_epochs=1,
                training_iterations=1,
                evaluation_episodes=1,
                seed=23,
            ),
        )
        mtppo_summary, mtppo_trace = trainer.rollout_episode(seed=123)
        baseline_summary, baseline_trace = run_heuristic_episode(config, seed=123, route_solver="greedy")

        self.assertEqual(len(mtppo_trace["daily"]), config.horizon_days)
        self.assertEqual(mtppo_trace["ending_inventory"].shape, (config.horizon_days, config.num_retailers))
        self.assertEqual(len(baseline_trace["daily"]), config.horizon_days)
        self.assertGreater(float(mtppo_summary["sum_cost"]), 0.0)
        self.assertGreater(float(baseline_summary["sum_cost"]), 0.0)


if __name__ == "__main__":
    unittest.main()
