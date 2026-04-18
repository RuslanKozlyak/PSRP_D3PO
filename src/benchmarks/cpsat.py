from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.env.instance import MPPSRPInstance


def build_euclidean_weight_matrix(
    points: np.ndarray,
    *,
    distance_scale: float = 100.0,
    round_to_int: bool = True,
) -> np.ndarray:
    coords = np.asarray(points, dtype=np.float64)
    diffs = coords[:, None, :] - coords[None, :, :]
    distances = np.linalg.norm(diffs, axis=-1) * distance_scale
    if round_to_int:
        return np.rint(distances).astype(np.int64)
    return distances.astype(np.float64)


def build_cpsat_bundle(
    weight_matrix: np.ndarray,
    builder_config: dict[str, Any],
    *,
    solve: bool = False,
    solve_config: dict[str, Any] | None = None,
    python_executable: str | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    helper_script = root / "mppsrp_nir" / "export_cpsat_bundle.py"
    if not helper_script.exists():
        raise FileNotFoundError(f"Expected helper script at {helper_script}")

    payload = {
        "weight_matrix": np.asarray(weight_matrix).tolist(),
        "builder_config": builder_config,
        "solve": bool(solve),
        "solve_config": solve_config or {},
    }
    command = [
        python_executable or sys.executable,
        str(helper_script),
        "--payload",
        json.dumps(payload),
    ]
    completed = subprocess.run(
        command,
        cwd=str(helper_script.parent),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "CP-SAT bundle export failed.\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("CP-SAT bundle export produced no JSON output.")
    return json.loads(lines[-1])


def build_graph_generator_weight_matrix(
    graph_config: dict[str, Any],
    *,
    python_executable: str | None = None,
    repo_root: str | Path | None = None,
) -> np.ndarray:
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    helper_script = root / "mppsrp_nir" / "export_graph_generator_weight_matrix.py"
    if not helper_script.exists():
        raise FileNotFoundError(f"Expected helper script at {helper_script}")

    command = [
        python_executable or sys.executable,
        str(helper_script),
        "--payload",
        json.dumps(dict(graph_config)),
    ]
    completed = subprocess.run(
        command,
        cwd=str(helper_script.parent),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Graph generator export failed.\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("Graph generator export produced no JSON output.")
    payload = json.loads(lines[-1])
    return np.asarray(payload["weight_matrix"], dtype=np.int64)


def cpsat_data_model_to_instance(
    data_model: dict[str, Any],
    *,
    max_steps: int | None = None,
    holding_cost: float = 0.01,
    inferred_station_classes: int = 1,
) -> MPPSRPInstance:
    distance_matrix = np.asarray(data_model["distance_matrix"], dtype=np.float32)
    travel_time_matrix = np.asarray(data_model["travel_time_matrix"], dtype=np.float32)
    station_data = np.asarray(data_model["station_data"], dtype=np.int64)
    vehicle_compartments = np.asarray(data_model["vehicle_compartments"], dtype=np.float32)
    vehicle_time_windows = np.asarray(data_model["vehicle_time_windows"], dtype=np.float32)
    restriction_matrix = np.asarray(data_model["restriction_matrix"], dtype=np.int64)
    service_times = np.asarray(data_model["service_times"], dtype=np.float32)

    n_nodes = int(distance_matrix.shape[0])
    n_stations = n_nodes - 1
    station_ids = sorted({int(value) for value in station_data[:, 0]})
    product_ids = sorted({int(value) for value in station_data[:, 1]})
    station_to_idx = {station_id: index for index, station_id in enumerate(station_ids)}
    product_to_idx = {product_id: index for index, product_id in enumerate(product_ids)}

    horizon_days = int(station_data.shape[1] - 5)
    n_products = len(product_ids)
    n_vehicles, n_compartments = vehicle_compartments.shape
    if n_compartments != n_products:
        raise ValueError(
            "Simplified RL environment requires one vehicle compartment per product "
            f"(got {n_compartments} compartments for {n_products} products). "
            "Adjust the CP-SAT data generator so the compartment count matches products_count."
        )

    station_capacity = np.zeros((n_stations, n_products), dtype=np.float32)
    safety_stock = np.zeros((n_stations, n_products), dtype=np.float32)
    initial_inventory = np.zeros((n_stations, n_products), dtype=np.float32)
    daily_demand = np.zeros((horizon_days, n_stations, n_products), dtype=np.float32)
    for row in station_data:
        station_idx = station_to_idx[int(row[0])]
        product_idx = product_to_idx[int(row[1])]
        safety_stock[station_idx, product_idx] = float(row[2])
        station_capacity[station_idx, product_idx] = float(row[3])
        initial_inventory[station_idx, product_idx] = float(row[4])
        daily_demand[:, station_idx, product_idx] = np.asarray(row[5:], dtype=np.float32)

    consumption_rate = daily_demand.mean(axis=0).astype(np.float32)
    global_start = float(vehicle_time_windows[:, 0].min())
    global_end = float(vehicle_time_windows[:, 1].max())
    shift_length = max(global_end - global_start, 1.0)

    coords = _infer_coordinates_from_distance_matrix(distance_matrix)
    depot_coord = coords[0]
    station_coords = coords[1:]

    station_class = np.zeros(n_stations, dtype=np.int64)
    holding_cost_matrix = np.full((n_stations, n_products), holding_cost, dtype=np.float32)
    vehicle_total_capacity = vehicle_compartments.sum(axis=1).astype(np.float32)
    vehicle_class = np.zeros(n_vehicles, dtype=np.int64)
    compartment_products = np.tile(
        np.arange(n_products, dtype=np.int64),
        (n_vehicles, 1),
    )
    compatibility = restriction_matrix[:, 1:].astype(bool)
    service_time = service_times[1:].astype(np.float32)

    if max_steps is None:
        max_trips = int(data_model.get("max_trips_per_day", 2))
        max_steps = int(horizon_days * n_vehicles * max(max_trips, 1) * max(n_stations + 1, 2) * 2)

    return MPPSRPInstance(
        horizon_days=horizon_days,
        n_stations=n_stations,
        n_vehicles=n_vehicles,
        n_products=n_products,
        n_compartments=n_compartments,
        n_station_classes=max(int(inferred_station_classes), 1),
        shift_length=shift_length,
        max_steps=max_steps,
        depot_coord=depot_coord.astype(np.float32),
        station_coords=station_coords.astype(np.float32),
        station_capacity=station_capacity,
        initial_inventory=initial_inventory,
        safety_stock=safety_stock,
        consumption_rate=consumption_rate,
        daily_demand=daily_demand,
        time_window_start=np.zeros(n_stations, dtype=np.float32),
        time_window_end=np.full(n_stations, shift_length, dtype=np.float32),
        station_class=station_class,
        service_time=service_time,
        holding_cost=holding_cost_matrix,
        vehicle_class=vehicle_class,
        vehicle_total_capacity=vehicle_total_capacity,
        compartment_capacity=vehicle_compartments.astype(np.float32),
        compartment_products=compartment_products,
        station_vehicle_compatibility=compatibility,
        distance_matrix=distance_matrix,
        travel_time_matrix=travel_time_matrix,
    )


def evaluate_policy_kpis(
    env: Any,
    policy: torch.nn.Module,
    *,
    episodes: int = 1,
    preferences: torch.Tensor | np.ndarray | None = None,
    deterministic: bool = True,
    device: torch.device | str | None = None,
    max_steps: int | None = None,
    return_reward_components: bool = False,
) -> dict[str, Any]:
    resolved_device = _resolve_device(policy, device)
    was_training = policy.training
    policy.eval()
    pref_tensor = _prepare_preferences(preferences, resolved_device)
    n_eval_episodes = max(int(episodes), 1)
    horizon_days = int(env.instance.horizon_days)
    average_stock_levels_accum = np.zeros(horizon_days, dtype=np.float64)
    reward_components = tuple(getattr(env.reward_fn, "components", ()))
    total_reward = np.zeros(len(reward_components), dtype=np.float64)
    scalar_totals = {
        "total_travel_distance": 0.0,
        "total_travel_time": 0.0,
        "dry_runs": 0.0,
        "average_stock_levels_percent": 0.0,
        "average_vehicle_utilization": 0.0,
        "average_stops_per_trip": 0.0,
    }

    try:
        for _ in range(n_eval_episodes):
            obs, _ = env.reset()
            done = False
            truncated = False
            step_limit = max_steps if max_steps is not None else env.instance.max_steps
            step_idx = 0

            day_end_fill_ratios: list[float] = []
            day_end_inventory: list[np.ndarray] = []
            dry_runs = 0.0
            total_travel_time = 0.0
            total_delivered_amount = 0.0
            total_trip_capacity = 0.0
            total_trips = 0.0
            total_stops = 0.0
            vehicle_trip_state = {
                vehicle_idx: {"active": False, "delivered": 0.0, "stops": 0}
                for vehicle_idx in range(env.instance.n_vehicles)
            }

            while not done and not truncated and step_idx < step_limit:
                obs_t = _to_tensor_dict(obs, resolved_device)
                with torch.no_grad():
                    output = policy.act(obs_t, preferences=pref_tensor, deterministic=deterministic)

                vehicle_idx = int(output.actions["vehicle"].item())
                node_idx = int(output.actions["node"].item())
                quantity_action = output.actions["quantity"].squeeze(0).detach().cpu().numpy()

                prev_day = int(env.state.day)
                prev_node = int(env.state.vehicle_node[vehicle_idx])
                prev_vehicle_volume = float(env.state.compartment_volume[vehicle_idx].sum())
                total_travel_time += float(env.instance.travel_time_matrix[prev_node, node_idx])

                if node_idx > 0:
                    vehicle_trip_state[vehicle_idx]["active"] = True
                    vehicle_trip_state[vehicle_idx]["stops"] += 1

                next_obs, reward_vec, done, truncated, _ = env.step(
                    {
                        "vehicle": vehicle_idx,
                        "node": node_idx,
                        "quantity": quantity_action,
                    }
                )
                if return_reward_components:
                    total_reward += reward_vec

                if node_idx > 0:
                    delivered = max(prev_vehicle_volume - float(env.state.compartment_volume[vehicle_idx].sum()), 0.0)
                    total_delivered_amount += max(delivered, 0.0)
                    vehicle_trip_state[vehicle_idx]["delivered"] += max(delivered, 0.0)

                if node_idx == 0 and vehicle_trip_state[vehicle_idx]["active"]:
                    total_trips += 1.0
                    total_stops += float(vehicle_trip_state[vehicle_idx]["stops"])
                    total_trip_capacity += float(env.instance.vehicle_total_capacity[vehicle_idx])
                    vehicle_trip_state[vehicle_idx] = {"active": False, "delivered": 0.0, "stops": 0}

                if int(env.state.day) > prev_day:
                    current_inventory = env.state.station_inventory.copy()
                    day_end_fill_ratios.extend(
                        (
                            current_inventory / np.maximum(env.instance.station_capacity, 1e-6)
                        ).reshape(-1).tolist()
                    )
                    day_end_inventory.append(current_inventory)
                    dry_runs += float(
                        np.sum(current_inventory <= env.instance.safety_stock)
                    )

                obs = next_obs
                step_idx += 1

            while len(day_end_inventory) < horizon_days:
                day_end_inventory.append(env.state.station_inventory.copy())

            average_stock_levels = _average_stock_levels_like_cpsat(
                day_end_inventory=day_end_inventory,
                n_days=horizon_days,
            )
            average_stock_levels_accum += average_stock_levels

            average_fill_percent = (
                100.0 * float(np.mean(day_end_fill_ratios))
                if day_end_fill_ratios
                else 0.0
            )
            utilization = (
                round(100.0 * (total_delivered_amount / total_trip_capacity), 1)
                if total_trip_capacity > 0
                else 0.0
            )
            avg_stops_per_trip = float(total_stops / total_trips) if total_trips > 0 else 0.0

            scalar_totals["total_travel_distance"] += float(env.state.cumulative_distance)
            scalar_totals["total_travel_time"] += float(total_travel_time)
            scalar_totals["dry_runs"] += float(dry_runs)
            scalar_totals["average_stock_levels_percent"] += float(average_fill_percent)
            scalar_totals["average_vehicle_utilization"] += float(utilization)
            scalar_totals["average_stops_per_trip"] += float(avg_stops_per_trip)
    finally:
        if was_training:
            policy.train()

    result: dict[str, Any] = {
        key: float(value / n_eval_episodes)
        for key, value in scalar_totals.items()
    }
    mean_stock_levels = average_stock_levels_accum / float(n_eval_episodes)
    if n_eval_episodes == 1:
        result["average_stock_levels"] = np.rint(mean_stock_levels).astype(np.int64).tolist()
    else:
        result["average_stock_levels"] = mean_stock_levels.tolist()

    if return_reward_components:
        for idx, component in enumerate(reward_components):
            result[f"reward_{component}"] = float(total_reward[idx] / n_eval_episodes)

    return result


def compare_policy_to_cpsat(
    env: Any,
    policy: torch.nn.Module,
    *,
    cpsat_bundle: dict[str, Any] | None = None,
    weight_matrix: np.ndarray | None = None,
    builder_config: dict[str, Any] | None = None,
    preferences: torch.Tensor | np.ndarray | None = None,
    deterministic: bool = True,
    device: torch.device | str | None = None,
    python_executable: str | None = None,
    repo_root: str | Path | None = None,
    solve_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rl_metrics = evaluate_policy_kpis(
        env,
        policy,
        episodes=1,
        preferences=preferences,
        deterministic=deterministic,
        device=device,
    )
    bundle = cpsat_bundle
    if bundle is None:
        if weight_matrix is None or builder_config is None:
            raise ValueError(
                "Provide either cpsat_bundle or both weight_matrix and builder_config "
                "to compare policy KPIs against CP-SAT."
            )
        bundle = build_cpsat_bundle(
            weight_matrix=np.asarray(weight_matrix),
            builder_config=builder_config,
            solve=True,
            solve_config=solve_config,
            python_executable=python_executable,
            repo_root=repo_root,
        )
    return {
        "rl_kpis": rl_metrics,
        "cpsat_kpis": bundle.get("kpis"),
        "cpsat_data_model": bundle.get("data_model"),
        "cpsat_routes_schedule": bundle.get("routes_schedule"),
        "cpsat_solution_found": bool(bundle.get("solution_found", False)),
    }


def _infer_coordinates_from_distance_matrix(distance_matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(distance_matrix, dtype=np.float64)
    n_nodes = matrix.shape[0]
    if n_nodes <= 1:
        return np.zeros((n_nodes, 2), dtype=np.float64)

    squared = matrix ** 2
    centering = np.eye(n_nodes) - np.ones((n_nodes, n_nodes), dtype=np.float64) / n_nodes
    gram = -0.5 * centering @ squared @ centering
    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order[:2]], 0.0)
    eigvecs = eigvecs[:, order[:2]]
    coords = eigvecs * np.sqrt(eigvals)
    if coords.shape[1] < 2:
        coords = np.pad(coords, ((0, 0), (0, 2 - coords.shape[1])))
    min_vals = coords.min(axis=0, keepdims=True)
    max_vals = coords.max(axis=0, keepdims=True)
    denom = np.where((max_vals - min_vals) > 1e-6, max_vals - min_vals, 1.0)
    return ((coords - min_vals) / denom).astype(np.float64)


def _to_tensor_dict(obs: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    tensor_obs: dict[str, torch.Tensor] = {}
    for key, value in obs.items():
        tensor = torch.as_tensor(value, device=device)
        if tensor.ndim == 0:
            tensor = tensor.unsqueeze(0)
        tensor_obs[key] = tensor.unsqueeze(0)
    return tensor_obs


def _resolve_device(policy: torch.nn.Module, device: torch.device | str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    try:
        return next(policy.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _prepare_preferences(
    preferences: torch.Tensor | np.ndarray | None,
    device: torch.device,
) -> torch.Tensor | None:
    if preferences is None:
        return None
    tensor = torch.as_tensor(preferences, dtype=torch.float32, device=device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


def _average_stock_levels_like_cpsat(
    *,
    day_end_inventory: list[np.ndarray],
    n_days: int,
) -> np.ndarray:
    average_stock_levels = np.zeros((n_days,), dtype=np.float64)
    for day_idx in range(n_days):
        current_inventory = np.asarray(day_end_inventory[day_idx], dtype=np.float64)
        current_levels = current_inventory[:, -1]
        average_stock_levels[day_idx] = float(current_levels[current_levels != 0].sum())
    return average_stock_levels
