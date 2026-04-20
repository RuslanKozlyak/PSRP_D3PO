from __future__ import annotations

from math import ceil

import numpy as np
import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from .config import EnvironmentConfig
from .env import IRPVMIEnv, JointAction


def order_up_to_replenishment(obs: dict[str, np.ndarray], config: EnvironmentConfig) -> np.ndarray:
    inventory = obs["inventory_state"][:, 0]
    last_demand = obs["inventory_state"][:, -1]
    target = np.minimum(config.retailer_capacity, last_demand * 2.0 + config.vehicle_capacity * 0.25)
    replenishment = np.clip(target - inventory, 0.0, config.vehicle_capacity)
    return replenishment.astype(np.float32)


def greedy_route(replenishment: np.ndarray, distance_matrix: np.ndarray, vehicle_capacity: float) -> list[int]:
    remaining = replenishment.copy()
    route = [0]
    current_node = 0
    current_load = vehicle_capacity

    while np.any(remaining > 1e-6):
        feasible = np.where((remaining > 1e-6) & (remaining <= current_load + 1e-6))[0]
        if len(feasible) == 0:
            if route[-1] != 0:
                route.append(0)
            current_node = 0
            current_load = vehicle_capacity
            continue

        next_idx = min(
            feasible,
            key=lambda idx: distance_matrix[current_node, idx + 1],
        )
        route.append(int(next_idx + 1))
        current_load -= float(remaining[next_idx])
        remaining[next_idx] = 0.0
        current_node = int(next_idx + 1)

    if route[-1] != 0:
        route.append(0)
    return route


def ortools_route(replenishment: np.ndarray, distance_matrix: np.ndarray, vehicle_capacity: float) -> list[int]:
    active = np.where(replenishment > 1e-6)[0]
    if len(active) == 0:
        return [0]

    demands = [0] + [int(round(replenishment[idx])) for idx in active]
    submatrix_indices = [0] + [idx + 1 for idx in active]
    submatrix = distance_matrix[np.ix_(submatrix_indices, submatrix_indices)]
    num_vehicles = max(1, ceil(sum(demands) / vehicle_capacity))
    manager = pywrapcp.RoutingIndexManager(len(submatrix), num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(round(submatrix[from_node, to_node] * 1000))

    transit_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_index)

    def demand_callback(from_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        return demands[from_node]

    demand_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_index,
        0,
        [int(vehicle_capacity)] * num_vehicles,
        True,
        "Capacity",
    )

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    solution = routing.SolveWithParameters(search_parameters)
    if solution is None:
        return greedy_route(replenishment, distance_matrix, vehicle_capacity)

    route = [0]
    for vehicle_id in range(num_vehicles):
        index = routing.Start(vehicle_id)
        if vehicle_id > 0 and route[-1] != 0:
            route.append(0)
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node != 0:
                route.append(submatrix_indices[node])
            index = solution.Value(routing.NextVar(index))
        if route[-1] != 0:
            route.append(0)
    return route


def evaluate_heuristic_baseline(
    config: EnvironmentConfig,
    episodes: int = 4,
    route_solver: str = "ortools",
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    solver = ortools_route if route_solver == "ortools" else greedy_route

    for episode_idx in range(episodes):
        env = IRPVMIEnv(config)
        obs, _ = env.reset(seed=100 + episode_idx)
        terminated = False
        metrics = {
            "reward": 0.0,
            "inventory_cost": 0.0,
            "route_distance": 0.0,
            "route_cost": 0.0,
            "fill_rate_sum": 0.0,
            "days": 0.0,
        }
        while not terminated:
            replenishment = order_up_to_replenishment(obs, config)
            route = solver(replenishment, obs["distance_matrix"], config.vehicle_capacity)
            obs, reward, terminated, _truncated, info = env.step(JointAction(replenishment=replenishment, route=route))
            metrics["reward"] += reward
            metrics["inventory_cost"] += float(info["inventory_cost"])
            metrics["route_distance"] += float(info["route_distance"])
            metrics["route_cost"] += float(info["route_cost"])
            metrics["fill_rate_sum"] += float(info["fill_rate"])
            metrics["days"] += 1.0

        rows.append(
            {
                "reward": metrics["reward"],
                "inventory_cost": metrics["inventory_cost"],
                "route_distance": metrics["route_distance"],
                "route_cost": metrics["route_cost"],
                "fill_rate": metrics["fill_rate_sum"] / max(metrics["days"], 1.0),
                "sum_cost": metrics["inventory_cost"] + metrics["route_cost"],
            }
        )

    return pd.DataFrame(rows)


def run_heuristic_episode(
    config: EnvironmentConfig,
    seed: int = 123,
    route_solver: str = "ortools",
) -> tuple[dict[str, float], dict[str, pd.DataFrame]]:
    env = IRPVMIEnv(config)
    obs, _ = env.reset(seed=seed)
    solver = ortools_route if route_solver == "ortools" else greedy_route
    terminated = False

    daily_rows: list[dict[str, float]] = []
    inventory_before_rows: list[np.ndarray] = []
    replenishment_rows: list[np.ndarray] = []
    demand_rows: list[np.ndarray] = []
    ending_inventory_rows: list[np.ndarray] = []
    lost_sales_rows: list[np.ndarray] = []
    summary = {
        "reward": 0.0,
        "inventory_cost": 0.0,
        "route_cost": 0.0,
        "route_distance": 0.0,
        "fill_rate_sum": 0.0,
        "days": 0.0,
    }

    while not terminated:
        replenishment = order_up_to_replenishment(obs, config)
        route = solver(replenishment, obs["distance_matrix"], config.vehicle_capacity)
        obs, reward, terminated, _truncated, info = env.step(JointAction(replenishment=replenishment, route=route))

        inventory_before_rows.append(np.asarray(info["inventory_before_replenishment"], dtype=np.float32))
        replenishment_rows.append(np.asarray(info["replenishment"], dtype=np.float32))
        demand_rows.append(np.asarray(info["demand_vector"], dtype=np.float32))
        ending_inventory_rows.append(np.asarray(info["ending_inventory"], dtype=np.float32))
        lost_sales_rows.append(np.asarray(info["lost_sales_vector"], dtype=np.float32))
        daily_rows.append(
            {
                "day": float(info["day"]),
                "reward": float(reward),
                "inventory_cost": float(info["inventory_cost"]),
                "holding_cost": float(info["holding_cost"]),
                "sales_loss_cost": float(info["sales_loss_cost"]),
                "route_cost": float(info["route_cost"]),
                "route_distance": float(info["route_distance"]),
                "fill_rate": float(info["fill_rate"]),
                "total_replenishment": float(np.sum(info["replenishment"])),
                "total_demand": float(info["demand"]),
                "fulfilled": float(info["fulfilled"]),
                "lost_sales": float(np.sum(info["lost_sales_vector"])),
                "avg_inventory_before": float(np.mean(info["inventory_before_replenishment"])),
                "avg_inventory_after": float(np.mean(info["ending_inventory"])),
                "min_inventory_after": float(np.min(info["ending_inventory"])),
                "max_inventory_after": float(np.max(info["ending_inventory"])),
                "active_retailers": float(info["active_retailers"]),
                "invalid_moves": float(info["invalid_moves"]),
                "route_stops": float(info["route_stops"]),
            }
        )

        summary["reward"] += float(reward)
        summary["inventory_cost"] += float(info["inventory_cost"])
        summary["route_cost"] += float(info["route_cost"])
        summary["route_distance"] += float(info["route_distance"])
        summary["fill_rate_sum"] += float(info["fill_rate"])
        summary["days"] += 1.0

    retailer_columns = [f"retailer_{idx + 1}" for idx in range(config.num_retailers)]
    day_index = pd.Index(range(len(daily_rows)), name="day")
    trace_tables = {
        "daily": pd.DataFrame(daily_rows),
        "inventory_before": pd.DataFrame(inventory_before_rows, columns=retailer_columns, index=day_index),
        "replenishment": pd.DataFrame(replenishment_rows, columns=retailer_columns, index=day_index),
        "demand": pd.DataFrame(demand_rows, columns=retailer_columns, index=day_index),
        "ending_inventory": pd.DataFrame(ending_inventory_rows, columns=retailer_columns, index=day_index),
        "lost_sales": pd.DataFrame(lost_sales_rows, columns=retailer_columns, index=day_index),
    }
    episode_summary = {
        "reward": summary["reward"],
        "inventory_cost": summary["inventory_cost"],
        "route_cost": summary["route_cost"],
        "route_distance": summary["route_distance"],
        "fill_rate": summary["fill_rate_sum"] / max(summary["days"], 1.0),
        "sum_cost": summary["inventory_cost"] + summary["route_cost"],
        "days": summary["days"],
    }
    return episode_summary, trace_tables
