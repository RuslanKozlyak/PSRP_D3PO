from __future__ import annotations

import sys
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from psrp_mtppo.config import EnvironmentConfig, ModelConfig
from psrp_mtppo.env import IRPVMIEnv
from psrp_mtppo.models.mtppo import MTPPOModel


class ModelTests(unittest.TestCase):
    def test_mtppo_action_shapes(self) -> None:
        env = IRPVMIEnv(EnvironmentConfig(num_retailers=5, horizon_days=4, seed=3))
        obs, _ = env.reset(seed=3)
        tensor_obs = {
            key: torch.as_tensor(value, dtype=torch.float32).unsqueeze(0)
            for key, value in obs.items()
        }
        model = MTPPOModel(EnvironmentConfig(num_retailers=5, horizon_days=4), ModelConfig(attention_heads=4))
        output = model.act(tensor_obs, greedy=False)

        self.assertEqual(tuple(output.replenishment.shape), (1, 5))
        self.assertGreaterEqual(output.route[0], 0)
        self.assertEqual(output.route[0], 0)
        self.assertEqual(output.route[-1], 0)
        self.assertTrue(torch.isfinite(output.inventory_log_prob).all())
        self.assertTrue(torch.isfinite(output.routing_log_prob).all())
        self.assertTrue(torch.isfinite(output.value).all())


if __name__ == "__main__":
    unittest.main()
