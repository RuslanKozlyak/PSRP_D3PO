from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path


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
