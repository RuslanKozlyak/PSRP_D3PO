from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PolicySourceTests(unittest.TestCase):
    def test_policy_modules_compile(self) -> None:
        for path in (ROOT / "src" / "policy").glob("*.py"):
            with self.subTest(path=path.name):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_notebook_exists(self) -> None:
        self.assertTrue((ROOT / "notebooks" / "00_quickstart.ipynb").exists())
