"""Baseline policies for IRP-VMI.

Four baselines are provided, three of them genetic algorithms matching
Lu et al. 2025 (Table 3 hyperparameters):

* ``greedy`` — fast order-up-to heuristic + nearest-neighbour route.
* ``ga_inv`` — GA(INV) replenishment + nearest-neighbour route (GA(INV)/A3C-style split).
* ``ga_vrp`` — order-up-to replenishment + GA(VRP) routing.
* ``ga_irp`` — GA(IRP) joint replenishment and routing (the paper's baseline).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .config import EnvironmentConfig
from .env import IRPVMIEnv, JointAction
from .ga import GAConfig, build_forecast, ga_inventory_action, ga_irp_action, ga_route


BaselineName = str


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


Policy = Callable[[dict[str, np.ndarray], EnvironmentConfig], tuple[np.ndarray, list[int]]]


def _greedy_policy(obs: dict[str, np.ndarray], config: EnvironmentConfig) -> tuple[np.ndarray, list[int]]:
    replenishment = order_up_to_replenishment(obs, config)
    route = greedy_route(replenishment, obs["distance_matrix"], config.vehicle_capacity)
    return replenishment, route


def _ga_inv_policy(
    obs: dict[str, np.ndarray],
    config: EnvironmentConfig,
    ga_config: GAConfig,
) -> tuple[np.ndarray, list[int]]:
    inventory = obs["inventory_state"][:, 0].astype(np.float32)
    forecast = build_forecast(obs)
    replenishment = ga_inventory_action(inventory, forecast, config, ga_config)
    route = greedy_route(replenishment, obs["distance_matrix"], config.vehicle_capacity)
    return replenishment, route


def _ga_vrp_policy(
    obs: dict[str, np.ndarray],
    config: EnvironmentConfig,
    ga_config: GAConfig,
) -> tuple[np.ndarray, list[int]]:
    replenishment = order_up_to_replenishment(obs, config)
    route, _distance = ga_route(
        replenishment,
        obs["distance_matrix"],
        config.vehicle_capacity,
        ga_config,
    )
    return replenishment, route


def _ga_irp_policy(
    obs: dict[str, np.ndarray],
    config: EnvironmentConfig,
    ga_config: GAConfig,
) -> tuple[np.ndarray, list[int]]:
    inventory = obs["inventory_state"][:, 0].astype(np.float32)
    forecast = build_forecast(obs)
    replenishment, route = ga_irp_action(
        inventory,
        forecast,
        obs["distance_matrix"],
        config,
        ga_config,
    )
    return replenishment, route


def build_policy(name: BaselineName, ga_config: GAConfig | None = None) -> Policy:
    ga_config = ga_config or GAConfig()
    name = name.lower()
    if name == "greedy":
        return _greedy_policy
    if name == "ga_inv":
        return lambda obs, config: _ga_inv_policy(obs, config, replace(ga_config))
    if name == "ga_vrp":
        return lambda obs, config: _ga_vrp_policy(obs, config, replace(ga_config))
    if name == "ga_irp":
        return lambda obs, config: _ga_irp_policy(obs, config, replace(ga_config))
    raise ValueError(f"Unknown baseline: {name}")


def _run_episode(
    config: EnvironmentConfig,
    policy: Policy,
    seed: int,
    collect_trace: bool = False,
) -> tuple[dict[str, float], list[dict[str, float]] | None, dict[str, list[np.ndarray]] | None]:
    env = IRPVMIEnv(config)
    obs, _ = env.reset(seed=seed)
    terminated = False

    summary = {
        "reward": 0.0,
        "inventory_cost": 0.0,
        "route_distance": 0.0,
        "route_cost": 0.0,
        "fill_rate_sum": 0.0,
        "days": 0.0,
    }
    daily_rows: list[dict[str, float]] = [] if collect_trace else None
    retailer_rows: dict[str, list[np.ndarray]] = (
        {
            "inventory_before": [],
            "replenishment": [],
            "demand": [],
            "ending_inventory": [],
            "lost_sales": [],
        }
        if collect_trace
        else None
    )

    while not terminated:
        replenishment, route = policy(obs, config)
        obs, reward, terminated, _truncated, info = env.step(
            JointAction(replenishment=replenishment, route=route)
        )

        summary["reward"] += float(reward)
        summary["inventory_cost"] += float(info["inventory_cost"])
        summary["route_distance"] += float(info["route_distance"])
        summary["route_cost"] += float(info["route_cost"])
        summary["fill_rate_sum"] += float(info["fill_rate"])
        summary["days"] += 1.0

        if collect_trace:
            retailer_rows["inventory_before"].append(np.asarray(info["inventory_before_replenishment"], dtype=np.float32))
            retailer_rows["replenishment"].append(np.asarray(info["replenishment"], dtype=np.float32))
            retailer_rows["demand"].append(np.asarray(info["demand_vector"], dtype=np.float32))
            retailer_rows["ending_inventory"].append(np.asarray(info["ending_inventory"], dtype=np.float32))
            retailer_rows["lost_sales"].append(np.asarray(info["lost_sales_vector"], dtype=np.float32))
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

    episode_summary = {
        "reward": summary["reward"],
        "inventory_cost": summary["inventory_cost"],
        "route_cost": summary["route_cost"],
        "route_distance": summary["route_distance"],
        "fill_rate": summary["fill_rate_sum"] / max(summary["days"], 1.0),
        "sum_cost": summary["inventory_cost"] + summary["route_cost"],
        "days": summary["days"],
    }
    return episode_summary, daily_rows, retailer_rows


def evaluate_baseline(
    config: EnvironmentConfig,
    baseline: BaselineName,
    episodes: int = 4,
    seeds: Iterable[int] | None = None,
    ga_config: GAConfig | None = None,
) -> pd.DataFrame:
    policy = build_policy(baseline, ga_config)
    seed_list = list(seeds) if seeds is not None else [100 + idx for idx in range(episodes)]
    rows: list[dict[str, float]] = []
    for seed in seed_list:
        summary, _trace, _retailers = _run_episode(config, policy, seed=seed)
        summary["baseline"] = baseline
        summary["seed"] = float(seed)
        rows.append(summary)
    return pd.DataFrame(rows)


def run_baseline_episode(
    config: EnvironmentConfig,
    baseline: BaselineName,
    seed: int = 123,
    ga_config: GAConfig | None = None,
) -> tuple[dict[str, float], dict[str, pd.DataFrame]]:
    policy = build_policy(baseline, ga_config)
    summary, daily_rows, retailer_rows = _run_episode(config, policy, seed=seed, collect_trace=True)

    retailer_columns = [f"retailer_{idx + 1}" for idx in range(config.num_retailers)]
    day_index = pd.Index(range(len(daily_rows)), name="day")
    trace_tables = {
        "daily": pd.DataFrame(daily_rows),
        "inventory_before": pd.DataFrame(retailer_rows["inventory_before"], columns=retailer_columns, index=day_index),
        "replenishment": pd.DataFrame(retailer_rows["replenishment"], columns=retailer_columns, index=day_index),
        "demand": pd.DataFrame(retailer_rows["demand"], columns=retailer_columns, index=day_index),
        "ending_inventory": pd.DataFrame(retailer_rows["ending_inventory"], columns=retailer_columns, index=day_index),
        "lost_sales": pd.DataFrame(retailer_rows["lost_sales"], columns=retailer_columns, index=day_index),
    }
    return summary, trace_tables


# ---------------------------------------------------------------------------
# Legacy aliases kept for backwards compatibility with earlier notebooks.
# ---------------------------------------------------------------------------


def evaluate_heuristic_baseline(
    config: EnvironmentConfig,
    episodes: int = 4,
    baseline: BaselineName = "greedy",
    ga_config: GAConfig | None = None,
    **kwargs,
) -> pd.DataFrame:
    if "route_solver" in kwargs:
        # old API used ``route_solver='greedy'``; silently map to the new baseline name
        baseline = kwargs["route_solver"]
        if baseline == "ortools":
            baseline = "ga_vrp"
    return evaluate_baseline(config, baseline=baseline, episodes=episodes, ga_config=ga_config)


def run_heuristic_episode(
    config: EnvironmentConfig,
    seed: int = 123,
    baseline: BaselineName = "greedy",
    ga_config: GAConfig | None = None,
    **kwargs,
) -> tuple[dict[str, float], dict[str, pd.DataFrame]]:
    if "route_solver" in kwargs:
        baseline = kwargs["route_solver"]
        if baseline == "ortools":
            baseline = "ga_vrp"
    return run_baseline_episode(config, baseline=baseline, seed=seed, ga_config=ga_config)
