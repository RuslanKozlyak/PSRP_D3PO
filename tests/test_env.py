from __future__ import annotations

import ast
import unittest
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from src.env.masks import compute_action_masks
from src.env.mp_psrp_env import MPPSRPEnv

ROOT = Path(__file__).resolve().parents[1]


class EnvSourceTests(unittest.TestCase):
    def test_env_modules_compile(self) -> None:
        for path in (ROOT / "src" / "env").glob("*.py"):
            with self.subTest(path=path.name):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_env_configs_exist(self) -> None:
        for name in ("small.yaml", "medium.yaml", "real_case.yaml"):
            self.assertTrue((ROOT / "configs" / "env" / name).exists())

    def test_depot_to_depot_parks_vehicle_and_advances_day(self) -> None:
        cfg = OmegaConf.load(ROOT / "configs" / "env" / "small.yaml")
        env = MPPSRPEnv(cfg)
        env.reset()

        zero_quantity = np.zeros(
            (env.instance.n_compartments, env.instance.n_products),
            dtype=np.float32,
        )

        _, _, done, truncated, info = env.step(
            {"vehicle": 0, "node": 0, "quantity": zero_quantity}
        )
        self.assertFalse(done)
        self.assertFalse(truncated)
        self.assertTrue(env.state.vehicle_shift_done[0])
        self.assertEqual(int(info["day"]), 0)

        _, reward_vec, done, truncated, info = env.step(
            {"vehicle": 1, "node": 0, "quantity": zero_quantity}
        )
        self.assertFalse(done)
        self.assertFalse(truncated)
        self.assertTrue(bool(info["day_advanced"]))
        self.assertEqual(int(info["day"]), 1)
        self.assertLess(float(reward_vec[2]), 0.0)

    def test_zero_delivery_station_visit_still_consumes_visit_and_blocks_self_loop(self) -> None:
        cfg = OmegaConf.load(ROOT / "configs" / "env" / "small.yaml")
        cfg.quantity_mode = "beta"
        env = MPPSRPEnv(cfg)
        obs, _ = env.reset()

        vehicle_idx = int(np.flatnonzero(obs["mask_vehicle"])[0])
        station_node = int(next(node for node in np.flatnonzero(obs["mask_node"][vehicle_idx]) if node > 0))
        zero_quantity = np.zeros(
            (env.instance.n_compartments, env.instance.n_products),
            dtype=np.float32,
        )

        obs, _, done, truncated, info = env.step(
            {"vehicle": vehicle_idx, "node": station_node, "quantity": zero_quantity}
        )
        self.assertFalse(done)
        self.assertFalse(truncated)
        self.assertEqual(float(info["delivered_volume"]), 0.0)
        self.assertEqual(int(env.state.station_visits_today[station_node - 1]), 1)
        self.assertFalse(bool(obs["mask_node"][vehicle_idx, station_node]))

    def test_compartments_are_fixed_one_to_one_with_products(self) -> None:
        cfg = OmegaConf.load(ROOT / "configs" / "env" / "small.yaml")
        env = MPPSRPEnv(cfg)
        env.reset()

        self.assertEqual(env.instance.n_compartments, env.instance.n_products)
        expected_products = np.arange(env.instance.n_products, dtype=np.int64)
        for vehicle_idx in range(env.instance.n_vehicles):
            np.testing.assert_array_equal(env.state.compartment_product[vehicle_idx], expected_products)

        vehicle_idx = 0
        product_idx = 0
        station_idx = int(np.flatnonzero(env.instance.station_vehicle_compatibility[vehicle_idx])[0])
        node_idx = station_idx + 1

        env.state.station_inventory[station_idx, product_idx] = (
            env.instance.station_capacity[station_idx, product_idx] - 1.0
        )
        env.instance.daily_demand[env.state.day, station_idx, product_idx] = max(
            float(env.instance.daily_demand[env.state.day, station_idx, product_idx]),
            5.0,
        )
        env.masks = compute_action_masks(env.state, env.instance)

        upper = env.masks.quantity_upper_bound[vehicle_idx, node_idx, :, product_idx]
        self.assertLessEqual(float(upper.sum()), 1.00001)
        self.assertGreaterEqual(float(upper[product_idx]), 0.0)
        nonmatching = np.delete(upper, product_idx)
        self.assertTrue(np.allclose(nonmatching, 0.0))
