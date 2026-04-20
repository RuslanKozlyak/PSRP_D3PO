from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from psrp_mtppo.config import EnvironmentConfig
from psrp_mtppo.env import IRPVMIEnv, JointAction


class EnvironmentTests(unittest.TestCase):
    def test_env_step_respects_shapes_and_costs(self) -> None:
        env = IRPVMIEnv(EnvironmentConfig(num_retailers=6, horizon_days=3, seed=11))
        obs, _ = env.reset(seed=11)
        action = JointAction(
            replenishment=np.full(6, 10.0, dtype=np.float32),
            route=[0, 1, 2, 3, 0, 4, 5, 6, 0],
        )
        next_obs, reward, terminated, truncated, info = env.step(action)

        self.assertEqual(next_obs["inventory_state"].shape, (6, 15))
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertLess(reward, 0.0)
        self.assertGreaterEqual(info["fill_rate"], 0.0)
        self.assertLessEqual(info["fill_rate"], 1.0)
        self.assertGreaterEqual(info["route_distance"], 0.0)

    def test_env_auto_completes_incomplete_routes(self) -> None:
        env = IRPVMIEnv(
            EnvironmentConfig(
                num_retailers=4,
                horizon_days=2,
                seed=5,
                initial_retailer_inventory=20.0,
            )
        )
        obs, _ = env.reset(seed=5)
        action = JointAction(
            replenishment=np.array([25.0, 25.0, 25.0, 25.0], dtype=np.float32),
            route=[0, 1],
        )
        _next_obs, _reward, _terminated, _truncated, info = env.step(action)
        self.assertEqual(info["executed_route"][0], 0)
        self.assertEqual(info["executed_route"][-1], 0)
        self.assertGreaterEqual(len(info["executed_route"]), 3)


if __name__ == "__main__":
    unittest.main()
