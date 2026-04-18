from __future__ import annotations

import ast
import unittest
from pathlib import Path

import numpy as np

from src.benchmarks.cpsat import build_euclidean_weight_matrix, cpsat_data_model_to_instance


ROOT = Path(__file__).resolve().parents[1]


class BenchmarkSourceTests(unittest.TestCase):
    def test_benchmark_modules_compile(self) -> None:
        for path in (ROOT / "src" / "benchmarks").glob("*.py"):
            with self.subTest(path=path.name):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_cpsat_data_model_to_instance_shapes(self) -> None:
        distance_matrix = build_euclidean_weight_matrix(
            np.array(
                [
                    [0.5, 0.5],
                    [0.2, 0.2],
                    [0.8, 0.2],
                    [0.5, 0.8],
                ],
                dtype=np.float64,
            ),
            distance_scale=100.0,
        )
        data_model = {
            "distance_matrix": distance_matrix.tolist(),
            "travel_time_matrix": distance_matrix.tolist(),
            "station_data": [
                [1, 0, 5, 95, 50, 10, 10],
                [1, 1, 5, 95, 50, 8, 8],
                [2, 0, 5, 95, 50, 10, 10],
                [2, 1, 5, 95, 50, 8, 8],
                [3, 0, 5, 95, 50, 10, 10],
                [3, 1, 5, 95, 50, 8, 8],
            ],
            "k_vehicles": 2,
            "vehicle_compartments": [[40, 40], [30, 30]],
            "vehicle_time_windows": [[0, 32400], [0, 32400]],
            "restriction_matrix": [[1, 1, 1, 1], [1, 1, 1, 1]],
            "service_times": [900, 600, 600, 600],
            "max_trips_per_day": 2,
        }
        instance = cpsat_data_model_to_instance(data_model, max_steps=128)
        self.assertEqual(instance.n_stations, 3)
        self.assertEqual(instance.n_products, 2)
        self.assertEqual(instance.n_vehicles, 2)
        self.assertEqual(instance.n_compartments, 2)
        self.assertEqual(instance.daily_demand.shape, (2, 3, 2))
        self.assertEqual(instance.station_vehicle_compatibility.shape, (2, 3))

    def test_cpsat_data_model_rejects_compartment_product_mismatch(self) -> None:
        distance_matrix = build_euclidean_weight_matrix(
            np.array(
                [
                    [0.5, 0.5],
                    [0.2, 0.2],
                    [0.8, 0.2],
                ],
                dtype=np.float64,
            ),
            distance_scale=100.0,
        )
        data_model = {
            "distance_matrix": distance_matrix.tolist(),
            "travel_time_matrix": distance_matrix.tolist(),
            "station_data": [
                [1, 0, 5, 95, 50, 10, 10],
                [1, 1, 5, 95, 50, 8, 8],
                [2, 0, 5, 95, 50, 10, 10],
                [2, 1, 5, 95, 50, 8, 8],
            ],
            "k_vehicles": 1,
            "vehicle_compartments": [[40, 40, 40]],
            "vehicle_time_windows": [[0, 32400]],
            "restriction_matrix": [[1, 1, 1]],
            "service_times": [900, 600, 600],
        }
        with self.assertRaises(ValueError):
            cpsat_data_model_to_instance(data_model, max_steps=64)
