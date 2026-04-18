from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class EnvSourceTests(unittest.TestCase):
    def test_env_modules_compile(self) -> None:
        for path in (ROOT / "src" / "env").glob("*.py"):
            with self.subTest(path=path.name):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_env_configs_exist(self) -> None:
        for name in ("small.yaml", "medium.yaml", "real_case.yaml"):
            self.assertTrue((ROOT / "configs" / "env" / name).exists())
