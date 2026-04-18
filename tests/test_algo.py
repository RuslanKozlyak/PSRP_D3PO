from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

import torch

from src.algo.base import BaseAlgorithm
from src.train.notebook import build_components, load_config


ROOT = Path(__file__).resolve().parents[1]


class AlgoSourceTests(unittest.TestCase):
    def test_algo_modules_compile(self) -> None:
        for path in (ROOT / "src" / "algo").glob("*.py"):
            with self.subTest(path=path.name):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_train_modules_compile(self) -> None:
        for path in (ROOT / "src" / "train").glob("*.py"):
            with self.subTest(path=path.name):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_scripts_compile(self) -> None:
        for path in (ROOT / "scripts").glob("*.py"):
            with self.subTest(path=path.name):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_notebook_json_valid(self) -> None:
        for notebook_name in ("00_quickstart.ipynb", "01_cpsat_comparison.ipynb"):
            notebook_path = ROOT / "notebooks" / notebook_name
            with self.subTest(notebook=notebook_name):
                with notebook_path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                self.assertEqual(payload["nbformat"], 4)

    def test_notebook_has_no_saved_error_outputs(self) -> None:
        for notebook_name in ("00_quickstart.ipynb", "01_cpsat_comparison.ipynb"):
            notebook_path = ROOT / "notebooks" / notebook_name
            with self.subTest(notebook=notebook_name):
                with notebook_path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                error_outputs = [
                    output
                    for cell in payload["cells"]
                    if cell.get("cell_type") == "code"
                    for output in cell.get("outputs", [])
                    if output.get("output_type") == "error"
                ]
                self.assertEqual(error_outputs, [])

    def test_gitignore_exists(self) -> None:
        self.assertTrue((ROOT / ".gitignore").exists())

    def test_d3po_auto_objectives_follow_reward_mode(self) -> None:
        with_holding_cfg = load_config(["env=small", "algo=d3po", "experiment=smoke", "env.reward_mode=with_holding"])
        env, policy, _, _ = build_components(with_holding_cfg)
        self.assertEqual(tuple(env.reward_fn.components), ("distance", "holding", "safety"))
        self.assertEqual(policy.n_objectives, 3)

        without_holding_cfg = load_config(
            ["env=small", "algo=d3po", "experiment=smoke", "env.reward_mode=without_holding"]
        )
        env, policy, _, _ = build_components(without_holding_cfg)
        self.assertEqual(tuple(env.reward_fn.components), ("distance", "safety"))
        self.assertEqual(policy.n_objectives, 2)

    def test_compute_gae_uses_bootstrap_value_for_unfinished_rollout(self) -> None:
        rewards = torch.zeros(1, 1)
        values = torch.zeros(1, 1)
        dones = torch.zeros(1, 1)
        bootstrap_value = torch.ones(1, 1)

        advantages = BaseAlgorithm.compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            gamma=0.99,
            gae_lambda=1.0,
            bootstrap_value=bootstrap_value,
        )

        self.assertAlmostEqual(float(advantages.item()), 0.99, places=6)
