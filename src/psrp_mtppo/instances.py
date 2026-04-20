from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import EnvironmentConfig


@dataclass(slots=True)
class ProblemInstance:
    retailer_coords: np.ndarray
    depot_coord: np.ndarray
    demands: np.ndarray
    retailer_capacity: np.ndarray
    initial_retailer_inventory: np.ndarray
    supplier_initial_inventory: float
    distance_matrix: np.ndarray


def euclidean_distance_matrix(depot_coord: np.ndarray, retailer_coords: np.ndarray) -> np.ndarray:
    nodes = np.concatenate([depot_coord[None, :], retailer_coords], axis=0)
    deltas = nodes[:, None, :] - nodes[None, :, :]
    return np.linalg.norm(deltas, axis=-1).astype(np.float32)


def generate_instance(config: EnvironmentConfig, rng: np.random.Generator) -> ProblemInstance:
    retailer_coords = rng.uniform(
        config.coord_low,
        config.coord_high,
        size=(config.num_retailers, 2),
    ).astype(np.float32)
    depot_coord = rng.uniform(config.coord_low, config.coord_high, size=(2,)).astype(np.float32)
    demands = rng.integers(
        config.demand_low,
        config.demand_high + 1,
        size=(config.horizon_days, config.num_retailers),
        endpoint=False,
        dtype=np.int32,
    ).astype(np.float32)
    retailer_capacity = np.full(config.num_retailers, config.retailer_capacity, dtype=np.float32)
    initial_inventory = np.full(config.num_retailers, config.initial_retailer_inventory, dtype=np.float32)

    return ProblemInstance(
        retailer_coords=retailer_coords,
        depot_coord=depot_coord,
        demands=demands,
        retailer_capacity=retailer_capacity,
        initial_retailer_inventory=initial_inventory,
        supplier_initial_inventory=config.supplier_initial_inventory,
        distance_matrix=euclidean_distance_matrix(depot_coord, retailer_coords),
    )
