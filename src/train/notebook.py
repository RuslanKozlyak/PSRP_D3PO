from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from src.algo.d3po import D3PO
from src.algo.ppo import PPO
from src.algo.ppo_lagrangian import PPOLagrangian
from src.benchmarks import cpsat as cpsat_benchmarks
from src.env.instance import MPPSRPInstance
from src.env.mp_psrp_env import MPPSRPEnv
from src.policy.policy import HierarchicalPolicy
from src.train.trainer import Trainer


def load_config(overrides: list[str] | None = None) -> DictConfig:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return compose(config_name="config", overrides=overrides or [])


def build_components(cfg: DictConfig) -> tuple[MPPSRPEnv, HierarchicalPolicy, Any, Trainer]:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    env_cfg = dict(cfg_dict["env"])
    policy_cfg = dict(cfg_dict["policy"])
    algo_cfg = dict(cfg_dict["algo"])
    exp_cfg = dict(cfg_dict["experiment"])

    env = MPPSRPEnv(env_cfg)
    return _build_components_from_env(env, policy_cfg, algo_cfg, exp_cfg)


def build_components_from_instance(
    cfg: DictConfig,
    instance: MPPSRPInstance,
) -> tuple[MPPSRPEnv, HierarchicalPolicy, Any, Trainer]:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    env_cfg = dict(cfg_dict["env"])
    policy_cfg = dict(cfg_dict["policy"])
    algo_cfg = dict(cfg_dict["algo"])
    exp_cfg = dict(cfg_dict["experiment"])

    env = MPPSRPEnv(env_cfg, instance=instance)
    return _build_components_from_env(env, policy_cfg, algo_cfg, exp_cfg)


def _build_components_from_env(
    env: MPPSRPEnv,
    policy_cfg: dict[str, Any],
    algo_cfg: dict[str, Any],
    exp_cfg: dict[str, Any],
) -> tuple[MPPSRPEnv, HierarchicalPolicy, Any, Trainer]:
    raw_n_objectives = algo_cfg.get("n_objectives", "auto")
    if isinstance(raw_n_objectives, str) and raw_n_objectives.lower() == "auto":
        n_objectives = len(env.reward_fn.components)
    else:
        n_objectives = int(raw_n_objectives)
    policy = HierarchicalPolicy.from_env(env, policy_cfg, n_objectives=n_objectives)
    algo = build_algorithm(policy, algo_cfg)
    trainer = Trainer(env, policy, algo, exp_cfg)
    return env, policy, algo, trainer


def build_algorithm(policy: HierarchicalPolicy, algo_cfg: dict[str, Any]) -> Any:
    name = str(algo_cfg["name"]).lower()
    if name == "ppo":
        return PPO(policy, algo_cfg)
    if name == "ppo_lagrangian":
        return PPOLagrangian(policy, algo_cfg)
    if name == "d3po":
        return D3PO(policy, algo_cfg)
    raise ValueError(f"Unsupported algorithm: {algo_cfg['name']}")


def build_reference_env(cfg: DictConfig) -> MPPSRPEnv:
    cfg_dict = _config_to_dict(cfg)
    env_cfg = dict(cfg_dict["env"])
    return MPPSRPEnv(env_cfg)


def create_reference_instance(cfg: DictConfig) -> MPPSRPInstance:
    return build_reference_env(cfg).instance


def comparison_config_frame(configs: dict[str, DictConfig]) -> Any:
    pd = _require_pandas()
    rows = []
    for label, cfg in configs.items():
        cfg_dict = _config_to_dict(cfg)
        algo_cfg = cfg_dict["algo"]
        exp_cfg = cfg_dict["experiment"]
        rows.append(
            {
                "run": label,
                "env": cfg_dict["env"]["name"],
                "algo": algo_cfg["name"],
                "total_iterations": exp_cfg["total_iterations"],
                "rollout_length": exp_cfg["rollout_length"],
                "eval_interval": exp_cfg["eval_interval"],
                "eval_episodes": exp_cfg.get("eval_episodes", 1),
                "learning_rate": algo_cfg["learning_rate"],
                "clip_eps": algo_cfg["clip_eps"],
                "minibatch_size": algo_cfg["minibatch_size"],
            }
        )
    return pd.DataFrame(rows).set_index("run")


def history_to_frame(history: list[dict[str, float]]) -> Any:
    pd = _require_pandas()
    return pd.DataFrame(history)


def station_summary_frame(env: MPPSRPEnv, obs: dict[str, np.ndarray]) -> Any:
    pd = _require_pandas()
    fill_ratio = env.state.station_inventory / np.maximum(env.instance.station_capacity, 1e-6)
    days_until_stockout = (
        env.state.station_inventory - env.instance.safety_stock
    ) / np.maximum(env.instance.consumption_rate, 1e-6)
    frame = pd.DataFrame(
        {
            "station_id": np.arange(1, env.instance.n_stations + 1, dtype=np.int64),
            "x": env.instance.station_coords[:, 0],
            "y": env.instance.station_coords[:, 1],
            "station_class": env.instance.station_class,
            "days_since_last_delivery": env.state.days_since_last_delivery,
            "valid_vehicles": obs["mask_node"][:, 1:].sum(axis=0),
            "mean_fill_ratio": fill_ratio.mean(axis=1),
            "min_days_until_stockout": days_until_stockout.min(axis=1),
        }
    )
    return frame.sort_values(["min_days_until_stockout", "mean_fill_ratio"]).reset_index(drop=True)


def vehicle_summary_frame(env: MPPSRPEnv, obs: dict[str, np.ndarray]) -> Any:
    pd = _require_pandas()
    return pd.DataFrame(
        {
            "vehicle_id": np.arange(env.instance.n_vehicles, dtype=np.int64),
            "current_node": env.state.vehicle_node,
            "time_in_shift": env.state.vehicle_time,
            "shift_remaining": env.instance.shift_length - env.state.vehicle_time,
            "trip_count": env.state.vehicle_trip_count,
            "is_actionable": obs["mask_vehicle"].astype(bool),
            "valid_nodes": obs["mask_node"].sum(axis=1),
            "loaded_volume": env.state.compartment_volume.sum(axis=1),
            "total_capacity": env.instance.vehicle_total_capacity,
        }
    )


def rollout_episode(
    env: MPPSRPEnv,
    policy: HierarchicalPolicy,
    *,
    deterministic: bool = True,
    preferences: torch.Tensor | np.ndarray | None = None,
    max_steps: int | None = None,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    device = _resolve_device(policy, device)
    obs, reset_info = env.reset()
    pref_tensor = _prepare_preferences(preferences, device)

    steps: list[dict[str, Any]] = []
    inventory_snapshots = [env.state.station_inventory.copy()]
    vehicle_times = [env.state.vehicle_time.copy()]
    daily_inventory = [env.state.station_inventory.copy()]
    delivered_per_station_day = np.zeros(
        (env.instance.horizon_days, env.instance.n_stations, env.instance.n_products),
        dtype=np.float32,
    )
    routes_per_day: dict[int, list[dict[str, Any]]] = {}
    done = False
    truncated = False
    step_idx = 0
    info: dict[str, Any] = {}
    limit = max_steps if max_steps is not None else env.instance.max_steps

    while not done and not truncated and step_idx < limit:
        obs_t = _to_tensor_dict(obs, device)
        chosen_vehicle_mask = obs["mask_vehicle"].astype(bool)

        with torch.no_grad():
            output = policy.act(obs_t, preferences=pref_tensor, deterministic=deterministic)

        vehicle_idx = int(output.actions["vehicle"].item())
        node_idx = int(output.actions["node"].item())
        quantity_action = output.actions["quantity"].squeeze(0).detach().cpu().numpy()
        quantity_upper = output.actions["quantity_upper_bound"].squeeze(0).detach().cpu().numpy()
        delivery_matrix = (
            quantity_upper
            if policy.quantity_mode == "deterministic"
            else np.clip(quantity_action, 0.0, 1.0) * quantity_upper
        )
        delivered_estimate = float(delivery_matrix.sum())

        from_node = int(env.state.vehicle_node[vehicle_idx])
        current_day = int(env.state.day)
        step_record = {
            "step": step_idx,
            "day": current_day,
            "vehicle": vehicle_idx,
            "from_node": from_node,
            "to_node": node_idx,
            "vehicle_time_before": float(env.state.vehicle_time[vehicle_idx]),
            "valid_vehicle_count": int(chosen_vehicle_mask.sum()),
            "valid_node_count": int(obs["mask_node"][vehicle_idx].sum()),
            "estimated_delivery": delivered_estimate,
        }

        action = {
            "vehicle": vehicle_idx,
            "node": node_idx,
            "quantity": quantity_action,
        }
        obs, reward_vec, done, truncated, info = env.step(action)

        actual_delivered = float(info.get("delivered_volume", 0.0))
        if node_idx > 0 and actual_delivered > 0.0:
            station_idx = node_idx - 1
            delivered_per_station_day[current_day, station_idx] += delivery_matrix.sum(axis=0)

        routes_per_day.setdefault(current_day, []).append(
            {
                "step": step_idx,
                "vehicle": vehicle_idx,
                "from_node": from_node,
                "to_node": node_idx,
                "delivered": actual_delivered,
            }
        )

        step_record.update(
            {
                "cumulative_distance": float(info["cumulative_distance"]),
                "cumulative_stockout": float(info["cumulative_stockout"]),
                "delivered_volume": actual_delivered,
                "day_advanced": bool(info.get("day_advanced", False)),
            }
        )
        step_record.update(_reward_components_to_dict(env.reward_fn.components, reward_vec))
        steps.append(step_record)
        inventory_snapshots.append(env.state.station_inventory.copy())
        vehicle_times.append(env.state.vehicle_time.copy())
        if info.get("day_advanced", False):
            daily_inventory.append(env.state.station_inventory.copy())
        step_idx += 1

    while len(daily_inventory) <= env.instance.horizon_days:
        daily_inventory.append(env.state.station_inventory.copy())

    return {
        "reset_info": reset_info,
        "steps": steps,
        "terminated": done,
        "truncated": truncated,
        "terminal_info": info,
        "inventory_snapshots": np.stack(inventory_snapshots, axis=0),
        "vehicle_times": np.stack(vehicle_times, axis=0),
        "daily_inventory": np.stack(daily_inventory, axis=0),
        "delivered_per_station_day": delivered_per_station_day,
        "routes_per_day": routes_per_day,
        "preferences": None if pref_tensor is None else pref_tensor.detach().cpu().numpy(),
        "reward_components": tuple(env.reward_fn.components),
    }


def episode_to_frame(episode: dict[str, Any]) -> Any:
    pd = _require_pandas()
    return pd.DataFrame(episode["steps"])


def episode_summary(episode: dict[str, Any]) -> dict[str, float]:
    steps = episode["steps"]
    reward_components = tuple(episode.get("reward_components", ()))
    if not steps:
        summary = {
            "n_steps": 0.0,
            "distance": 0.0,
            "stockout": 0.0,
        }
        for component in reward_components:
            summary[f"reward_{component}"] = 0.0
        return summary

    summary = {
        "n_steps": float(len(steps)),
        "distance": float(episode["terminal_info"]["cumulative_distance"]),
        "stockout": float(episode["terminal_info"]["cumulative_stockout"]),
    }
    for component in reward_components:
        key = f"reward_{component}"
        summary[key] = float(sum(step.get(key, 0.0) for step in steps))
    return summary


def resolve_eval_preferences(
    cfg: DictConfig,
    preferences: torch.Tensor | np.ndarray | None = None,
    *,
    n_objectives: int | None = None,
) -> torch.Tensor | np.ndarray | None:
    if preferences is not None:
        return preferences
    cfg_dict = _config_to_dict(cfg)
    algo_name = str(cfg_dict["algo"]["name"]).lower()
    if algo_name != "d3po":
        return None
    default_preference = cfg_dict["experiment"].get("d3po_eval_preference")
    if default_preference is None:
        return None
    if isinstance(default_preference, str) and default_preference.lower() == "auto":
        resolved_n_objectives = int(n_objectives or 1)
        if resolved_n_objectives == 3:
            return np.asarray([0.55, 0.15, 0.30], dtype=np.float32)
        if resolved_n_objectives == 2:
            return np.asarray([0.70, 0.30], dtype=np.float32)
        return np.full((resolved_n_objectives,), 1.0 / max(resolved_n_objectives, 1), dtype=np.float32)
    return np.asarray(default_preference, dtype=np.float32)


def evaluate_policy(
    env: MPPSRPEnv,
    policy: HierarchicalPolicy,
    *,
    episodes: int = 1,
    deterministic: bool = True,
    preferences: torch.Tensor | np.ndarray | None = None,
    max_steps: int | None = None,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    return cpsat_benchmarks.evaluate_policy_kpis(
        env,
        policy,
        episodes=episodes,
        preferences=preferences,
        deterministic=deterministic,
        device=device,
        max_steps=max_steps,
        return_reward_components=True,
    )


def run_training_pipeline(
    cfg: DictConfig,
    *,
    instance: MPPSRPInstance | None = None,
    eval_preferences: torch.Tensor | np.ndarray | None = None,
    deterministic_eval: bool = True,
    comparison_eval_episodes: int | None = None,
) -> dict[str, Any]:
    if instance is None:
        env, policy, algo, trainer = build_components(cfg)
    else:
        env, policy, algo, trainer = build_components_from_instance(cfg, instance)

    cfg_dict = _config_to_dict(cfg)
    resolved_preferences = resolve_eval_preferences(
        cfg,
        preferences=eval_preferences,
        n_objectives=policy.n_objectives,
    )
    eval_episodes = int(
        comparison_eval_episodes
        if comparison_eval_episodes is not None
        else cfg_dict["experiment"].get(
            "comparison_eval_episodes",
            cfg_dict["experiment"].get("eval_episodes", 1),
        )
    )

    before_episode = _rollout_for_display(
        env,
        policy,
        deterministic=deterministic_eval,
        preferences=resolved_preferences,
        device=trainer.device,
    )
    history = trainer.train()
    after_episode = _rollout_for_display(
        env,
        policy,
        deterministic=deterministic_eval,
        preferences=resolved_preferences,
        device=trainer.device,
    )
    evaluation = evaluate_policy(
        env,
        policy,
        episodes=eval_episodes,
        deterministic=deterministic_eval,
        preferences=resolved_preferences,
        device=trainer.device,
    )

    result = {
        "cfg": cfg,
        "env": env,
        "policy": policy,
        "algo": algo,
        "trainer": trainer,
        "history": history,
        "history_frame": history_to_frame(history),
        "before_episode": before_episode,
        "after_episode": after_episode,
        "before_summary": episode_kpis(env, before_episode),
        "after_summary": episode_kpis(env, after_episode),
        "evaluation": evaluation,
        "eval_preferences": resolved_preferences,
        "comparison_eval_episodes": eval_episodes,
    }
    result["final_metrics"] = history[-1] if history else {}
    return result


def run_algorithm_suite(
    configs: dict[str, DictConfig],
    *,
    instance: MPPSRPInstance | None = None,
    eval_preferences: dict[str, torch.Tensor | np.ndarray] | None = None,
    deterministic_eval: bool = True,
    comparison_eval_episodes: int | None = None,
) -> dict[str, Any]:
    if not configs:
        raise ValueError("run_algorithm_suite requires at least one config.")

    first_cfg = next(iter(configs.values()))
    reference_env = (
        build_reference_env(first_cfg)
        if instance is None
        else MPPSRPEnv(_config_to_dict(first_cfg)["env"], instance=instance)
    )
    shared_instance = reference_env.instance if instance is None else instance
    preview_obs, preview_info = reference_env.reset()

    results: dict[str, Any] = {}
    for label, cfg in configs.items():
        pref = None if eval_preferences is None else eval_preferences.get(label)
        results[label] = run_training_pipeline(
            cfg,
            instance=shared_instance,
            eval_preferences=pref,
            deterministic_eval=deterministic_eval,
            comparison_eval_episodes=comparison_eval_episodes,
        )

    return {
        "reference_env": reference_env,
        "reference_obs": preview_obs,
        "reference_info": preview_info,
        "instance": shared_instance,
        "results": results,
    }


def comparison_summary_frame(results: dict[str, dict[str, Any]]) -> Any:
    pd = _require_pandas()
    rows: list[dict[str, Any]] = []
    for label, result in results.items():
        cfg_dict = _config_to_dict(result["cfg"])
        before = result["before_summary"]
        after = result["after_summary"]
        row: dict[str, Any] = {
            "algorithm": label,
            "algo_name": cfg_dict["algo"]["name"],
            "env_name": cfg_dict["env"]["name"],
            "before_total_travel_distance": before["total_travel_distance"],
            "after_total_travel_distance": after["total_travel_distance"],
            "before_dry_runs": before["dry_runs"],
            "after_dry_runs": after["dry_runs"],
            "before_average_stock_levels_percent": before["average_stock_levels_percent"],
            "after_average_stock_levels_percent": after["average_stock_levels_percent"],
            "before_average_vehicle_utilization": before["average_vehicle_utilization"],
            "after_average_vehicle_utilization": after["average_vehicle_utilization"],
            "before_average_stops_per_trip": before["average_stops_per_trip"],
            "after_average_stops_per_trip": after["average_stops_per_trip"],
            "before_total_travel_time": before["total_travel_time"],
            "after_total_travel_time": after["total_travel_time"],
        }
        row["before_distance"] = row["before_total_travel_distance"]
        row["after_distance"] = row["after_total_travel_distance"]
        row["untrained_total_travel_distance"] = row["before_total_travel_distance"]
        row["trained_total_travel_distance"] = row["after_total_travel_distance"]
        row["untrained_dry_runs"] = row["before_dry_runs"]
        row["trained_dry_runs"] = row["after_dry_runs"]
        row["untrained_average_stock_levels_percent"] = row["before_average_stock_levels_percent"]
        row["trained_average_stock_levels_percent"] = row["after_average_stock_levels_percent"]
        row["distance_improvement"] = row["before_total_travel_distance"] - row["after_total_travel_distance"]
        row["dry_runs_improvement"] = row["before_dry_runs"] - row["after_dry_runs"]
        row["average_stock_levels_percent_improvement"] = (
            row["after_average_stock_levels_percent"] - row["before_average_stock_levels_percent"]
        )
        row["distance_improvement_pct"] = (
            100.0 * row["distance_improvement"] / max(abs(row["before_total_travel_distance"]), 1e-6)
        )
        row["dry_runs_improvement_pct"] = (
            100.0 * row["dry_runs_improvement"] / max(abs(row["before_dry_runs"]), 1e-6)
            if abs(row["before_dry_runs"]) > 1e-6
            else 0.0
        )
        row["fill_percent_improvement"] = row["average_stock_levels_percent_improvement"]
        for key, value in result.get("evaluation", {}).items():
            if isinstance(value, (int, float)):
                row[f"eval_{key}"] = float(value)
        for key, value in result.get("final_metrics", {}).items():
            if isinstance(value, (int, float)):
                row[f"final_{key}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows).set_index("algorithm").sort_index()


def cpsat_comparison_frame(comparisons: dict[str, dict[str, Any]]) -> Any:
    pd = _require_pandas()
    rows: list[dict[str, Any]] = []
    cpsat_kpis: dict[str, float] | None = None

    for algorithm, comparison in comparisons.items():
        rl_row = {"solver": algorithm, "source": "trained_policy"}
        rl_row.update(comparison.get("rl_kpis") or {})
        rows.append(rl_row)

        if cpsat_kpis is None and comparison.get("cpsat_kpis"):
            cpsat_kpis = comparison["cpsat_kpis"]

    if cpsat_kpis:
        cpsat_row = {"solver": "cpsat", "source": "cp_sat_solver"}
        cpsat_row.update(cpsat_kpis)
        rows.append(cpsat_row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    preferred_order = ["ppo", "ppo_lagrangian", "d3po", "cpsat"]
    categorical_order = [label for label in preferred_order if label in frame["solver"].tolist()]
    remaining = [label for label in frame["solver"].tolist() if label not in categorical_order]
    frame["solver"] = pd.Categorical(
        frame["solver"],
        categories=categorical_order + remaining,
        ordered=True,
    )
    frame = frame.sort_values("solver").set_index("solver")

    if "cpsat" in frame.index:
        unsupported_metrics = [
            column
            for column in frame.columns
            if column != "source" and _is_missing_scalar(frame.loc["cpsat", column], pd)
        ]
        if unsupported_metrics:
            frame = frame.drop(columns=unsupported_metrics)

        for metric in (
            "total_travel_distance",
            "total_travel_time",
            "dry_runs",
            "average_stock_levels_percent",
            "average_vehicle_utilization",
            "average_stops_per_trip",
        ):
            if metric in frame.columns:
                frame[f"{metric}_gap_to_cpsat"] = frame[metric] - float(frame.loc["cpsat", metric])
        if "total_travel_distance_gap_to_cpsat" in frame.columns:
            frame.loc["cpsat", "total_travel_distance_gap_to_cpsat"] = 0.0
        if "total_travel_time_gap_to_cpsat" in frame.columns:
            frame.loc["cpsat", "total_travel_time_gap_to_cpsat"] = 0.0
        if "dry_runs_gap_to_cpsat" in frame.columns:
            frame.loc["cpsat", "dry_runs_gap_to_cpsat"] = 0.0
        if "average_stock_levels_percent_gap_to_cpsat" in frame.columns:
            frame.loc["cpsat", "average_stock_levels_percent_gap_to_cpsat"] = 0.0
        if "average_vehicle_utilization_gap_to_cpsat" in frame.columns:
            frame.loc["cpsat", "average_vehicle_utilization_gap_to_cpsat"] = 0.0
        if "average_stops_per_trip_gap_to_cpsat" in frame.columns:
            frame.loc["cpsat", "average_stops_per_trip_gap_to_cpsat"] = 0.0

    return frame


def comparison_history_frame(results: dict[str, dict[str, Any]]) -> Any:
    pd = _require_pandas()
    frames = []
    for label, result in results.items():
        frame = result["history_frame"].copy()
        frame["algorithm"] = label
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=0, ignore_index=True)


def plot_instance_map(
    env: MPPSRPEnv,
    *,
    ax: Any | None = None,
    annotate: bool = True,
) -> Any:
    plt = _require_matplotlib()
    _, ax = _maybe_create_axis(ax, plt, figsize=(6, 5))

    classes = env.instance.station_class
    scatter = ax.scatter(
        env.instance.station_coords[:, 0],
        env.instance.station_coords[:, 1],
        c=classes,
        cmap="tab10",
        s=110,
        edgecolor="black",
        linewidth=0.7,
    )
    ax.scatter(
        env.instance.depot_coord[0],
        env.instance.depot_coord[1],
        marker="*",
        s=320,
        c="#d1495b",
        edgecolor="black",
        linewidth=0.8,
        label="Depot",
    )
    if annotate:
        for station_idx, (x, y) in enumerate(env.instance.station_coords, start=1):
            ax.text(x + 0.01, y + 0.01, str(station_idx), fontsize=8)

    ax.set_title("Station Layout")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)
    return scatter


def plot_inventory_heatmap(
    env: MPPSRPEnv,
    inventory: np.ndarray,
    *,
    ax: Any | None = None,
    title: str = "Station Inventory Fill Ratio",
) -> Any:
    plt = _require_matplotlib()
    _, ax = _maybe_create_axis(ax, plt, figsize=(7, 4))

    fill_ratio = inventory / np.maximum(env.instance.station_capacity, 1e-6)
    image = ax.imshow(fill_ratio, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel("Product")
    ax.set_ylabel("Station")
    ax.set_xticks(np.arange(env.instance.n_products))
    ax.set_yticks(np.arange(env.instance.n_stations))
    ax.set_yticklabels(np.arange(1, env.instance.n_stations + 1))
    return image


def plot_inventory_dynamics(
    env: MPPSRPEnv,
    episode: dict[str, Any],
    *,
    ax: Any | None = None,
    product: int | None = None,
    title: str | None = None,
) -> Any:
    plt = _require_matplotlib()
    fig, ax = _maybe_create_axis(ax, plt, figsize=(10, 5))

    daily = episode.get("daily_inventory")
    if daily is None or len(daily) == 0:
        ax.set_title(title or "Inventory Dynamics (no data)")
        return ax

    daily_arr = np.asarray(daily)
    capacity = env.instance.station_capacity
    if product is None:
        fill = daily_arr.sum(axis=-1) / np.maximum(capacity.sum(axis=-1, keepdims=False), 1e-6)
        ylabel = "station fill ratio (all products)"
    else:
        fill = daily_arr[..., product] / np.maximum(capacity[:, product], 1e-6)
        ylabel = f"station fill ratio (product {product})"

    days = np.arange(fill.shape[0])
    n_stations = fill.shape[1]
    cmap = plt.cm.get_cmap("nipy_spectral", max(n_stations, 1))
    for station_idx in range(n_stations):
        ax.plot(
            days,
            fill[:, station_idx],
            linewidth=2.0,
            alpha=0.9,
            color=cmap(station_idx),
            label=f"Station {station_idx + 1}",
        )

    ax.axhline(1.0, color="black", linewidth=0.8, linestyle=":", alpha=0.5, label="capacity")
    ax.axhline(0.0, color="#d1495b", linewidth=0.8, linestyle="--", alpha=0.5, label="empty")

    if title is None:
        if product is None:
            title = "Station Fill Ratio by Day (Day 0 = Initial Inventory)"
        else:
            title = f"Station Fill Ratio by Day for Product {product} (Day 0 = Initial Inventory)"
    ax.set_title(title)
    ax.set_xlabel("Day")
    ax.set_ylabel(ylabel)
    ax.set_xticks(days)
    if len(days) > 0:
        ax.set_xlim(float(days.min()), float(days.max()))
    ax.set_ylim(-0.05, 1.1)
    ax.grid(alpha=0.25)
    legend_cols = 1 if n_stations <= 12 else 2
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=8,
        frameon=False,
        ncol=legend_cols,
    )
    return ax


def plot_daily_delivered(
    env: MPPSRPEnv,
    episode: dict[str, Any],
    *,
    ax: Any | None = None,
) -> Any:
    plt = _require_matplotlib()
    fig, ax = _maybe_create_axis(ax, plt, figsize=(10, 4))

    delivered = episode.get("delivered_per_station_day")
    if delivered is None or delivered.size == 0:
        ax.set_title("Delivered Fuel per Day (no data)")
        return ax

    per_day_total = delivered.sum(axis=(1, 2))
    per_day_product = delivered.sum(axis=1)

    days = np.arange(per_day_total.shape[0])
    bottom = np.zeros_like(per_day_total)
    cmap = plt.cm.get_cmap("Set2", max(per_day_product.shape[1], 1))
    for product_idx in range(per_day_product.shape[1]):
        ax.bar(
            days,
            per_day_product[:, product_idx],
            bottom=bottom,
            label=f"Product {product_idx}",
            color=cmap(product_idx),
            edgecolor="black",
            linewidth=0.5,
        )
        bottom = bottom + per_day_product[:, product_idx]

    for day_idx, total in enumerate(per_day_total):
        if total > 0:
            ax.text(day_idx, total, f"{total:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_title("Delivered Fuel per Day (stacked by product)")
    ax.set_xlabel("Day")
    ax.set_ylabel("Delivered volume")
    ax.set_xticks(days)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(loc="upper right")
    return ax


def plot_route_map(
    env: MPPSRPEnv,
    episode: dict[str, Any],
    *,
    ax: Any | None = None,
    day: int | None = None,
) -> Any:
    plt = _require_matplotlib()
    fig, ax = _maybe_create_axis(ax, plt, figsize=(8, 7))

    plot_instance_map(env, ax=ax, annotate=True)

    node_coords = np.vstack([env.instance.depot_coord[None, :], env.instance.station_coords])
    colors = plt.cm.get_cmap("tab10", max(env.instance.n_vehicles, 1))

    routes_per_day = episode.get("routes_per_day", {})
    if day is None:
        selected = [(d, segs) for d, segs in sorted(routes_per_day.items())]
        title = "Routes (all days)"
    else:
        selected = [(day, routes_per_day.get(day, []))]
        title = f"Routes on Day {day}"

    for day_idx, segments in selected:
        for seg in segments:
            start = node_coords[seg["from_node"]]
            end = node_coords[seg["to_node"]]
            line_style = "-" if day_idx % 2 == 0 else "--"
            ax.annotate(
                "",
                xy=end,
                xytext=start,
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=colors(seg["vehicle"]),
                    linewidth=2.0,
                    alpha=0.8,
                    linestyle=line_style,
                    shrinkA=6,
                    shrinkB=6,
                ),
            )

    handles = [
        plt.Line2D([0], [0], color=colors(v), linewidth=2.0, label=f"Vehicle {v}")
        for v in range(env.instance.n_vehicles)
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=9)
    ax.set_title(title)
    return ax


def episode_kpis(env: MPPSRPEnv, episode: dict[str, Any]) -> dict[str, Any]:
    steps = episode.get("steps", [])
    daily_inv = np.asarray(episode.get("daily_inventory"))
    horizon_days = int(env.instance.horizon_days)

    total_travel_time = 0.0
    total_delivered_amount = 0.0
    total_trip_capacity = 0.0
    total_trips = 0.0
    total_stops = 0.0
    vehicle_trip_state = {
        vehicle_idx: {"active": False, "delivered": 0.0, "stops": 0}
        for vehicle_idx in range(env.instance.n_vehicles)
    }

    for step in steps:
        vehicle_idx = int(step["vehicle"])
        from_node = int(step["from_node"])
        to_node = int(step["to_node"])
        total_travel_time += float(env.instance.travel_time_matrix[from_node, to_node])

        if to_node > 0:
            vehicle_trip_state[vehicle_idx]["active"] = True
            vehicle_trip_state[vehicle_idx]["stops"] += 1
            delivered = float(step.get("delivered_volume", 0.0))
            total_delivered_amount += delivered
            vehicle_trip_state[vehicle_idx]["delivered"] += delivered

        if to_node == 0 and vehicle_trip_state[vehicle_idx]["active"]:
            total_trips += 1.0
            total_stops += float(vehicle_trip_state[vehicle_idx]["stops"])
            total_trip_capacity += float(env.instance.vehicle_total_capacity[vehicle_idx])
            vehicle_trip_state[vehicle_idx] = {"active": False, "delivered": 0.0, "stops": 0}

    if daily_inv.ndim == 3 and daily_inv.shape[0] > 1:
        day_end_inventory = [np.asarray(daily_inv[idx], dtype=np.float64) for idx in range(1, daily_inv.shape[0])]
    else:
        final_inventory = np.asarray(env.state.station_inventory, dtype=np.float64)
        day_end_inventory = [final_inventory.copy() for _ in range(horizon_days)]

    while len(day_end_inventory) < horizon_days:
        day_end_inventory.append(day_end_inventory[-1].copy())

    average_stock_levels = _average_stock_levels_like_episode(
        day_end_inventory=day_end_inventory,
        n_days=horizon_days,
    )
    fill_ratios = [
        (
            inventory / np.maximum(env.instance.station_capacity, 1e-6)
        ).reshape(-1)
        for inventory in day_end_inventory[:horizon_days]
    ]
    fill_ratio_values = np.concatenate(fill_ratios, axis=0) if fill_ratios else np.zeros((0,), dtype=np.float64)
    dry_runs = sum(
        float(np.sum(inventory <= env.instance.safety_stock))
        for inventory in day_end_inventory[:horizon_days]
    )
    utilization = (
        round(100.0 * (total_delivered_amount / total_trip_capacity), 1)
        if total_trip_capacity > 0
        else 0.0
    )
    avg_stops_per_trip = float(total_stops / total_trips) if total_trips > 0 else 0.0
    terminal = episode.get("terminal_info", {}) or {}

    return {
        "total_travel_distance": float(terminal.get("cumulative_distance", 0.0)),
        "total_travel_time": float(total_travel_time),
        "average_stock_levels": np.rint(average_stock_levels).astype(np.int64).tolist(),
        "average_stock_levels_percent": (
            100.0 * float(fill_ratio_values.mean()) if fill_ratio_values.size > 0 else 0.0
        ),
        "dry_runs": float(dry_runs),
        "average_vehicle_utilization": float(utilization),
        "average_stops_per_trip": float(avg_stops_per_trip),
    }


def daily_route_frame(episode: dict[str, Any]) -> Any:
    pd = _require_pandas()
    routes_per_day = episode.get("routes_per_day", {})
    rows = []
    for day in sorted(routes_per_day.keys()):
        for seg in routes_per_day[day]:
            rows.append(
                {
                    "day": int(day),
                    "step": int(seg["step"]),
                    "vehicle": int(seg["vehicle"]),
                    "from_node": int(seg["from_node"]),
                    "to_node": int(seg["to_node"]),
                    "delivered": float(seg["delivered"]),
                }
            )
    return pd.DataFrame(rows)


def plot_training_history(history: list[dict[str, float]], *, figsize: tuple[int, int] = (12, 8)) -> Any:
    plt = _require_matplotlib()
    frame = history_to_frame(history)
    fig, axes = plt.subplots(3, 2, figsize=figsize, constrained_layout=True)

    if not frame.empty:
        actor_cols = [
            col
            for col in (
                "policy_loss",
                "policy_surrogate_abs",
            )
            if col in frame.columns
        ]
        if actor_cols:
            frame.plot(x="iteration", y=actor_cols, ax=axes[0, 0], title="Actor Surrogate")
        value_cols = [col for col in ("value_loss", "diversity_loss") if col in frame.columns]
        if value_cols:
            frame.plot(x="iteration", y=value_cols, ax=axes[0, 1], title="Critic / Diversity")
        entropy_cols = [col for col in ("entropy",) if col in frame.columns]
        if entropy_cols:
            frame.plot(x="iteration", y=entropy_cols, ax=axes[1, 0], title="Entropy")
        update_diag_cols = [
            col
            for col in (
                "approx_kl",
                "clip_fraction",
            )
            if col in frame.columns
        ]
        if update_diag_cols:
            frame.plot(x="iteration", y=update_diag_cols, ax=axes[1, 1], title="Update Diagnostics")

        eval_cols = [
            col
            for col in (
                "eval_total_travel_distance",
                "eval_dry_runs",
                "eval_average_stock_levels_percent",
            )
            if col in frame.columns
        ]
        if eval_cols:
            frame.plot(x="iteration", y=eval_cols, ax=axes[2, 0], title="Eval KPIs")
        reward_cols = [
            col for col in frame.columns if col.startswith("eval_reward_")
        ]
        if reward_cols:
            frame.plot(x="iteration", y=reward_cols, ax=axes[2, 1], title="Eval Reward Components")

    for ax in axes.flat:
        ax.grid(alpha=0.2)
    return fig


def plot_algorithm_comparison(
    results: dict[str, dict[str, Any]],
    *,
    metrics: tuple[str, ...] = (
        "eval_total_travel_distance",
        "eval_dry_runs",
        "eval_average_stock_levels_percent",
        "eval_average_vehicle_utilization",
        "policy_surrogate_abs",
        "approx_kl",
        "clip_fraction",
        "value_loss",
        "entropy",
    ),
    figsize: tuple[int, int] = (15, 11),
) -> Any:
    plt = _require_matplotlib()
    frame = comparison_history_frame(results)
    n_cols = 2
    n_rows = max(1, (len(metrics) + n_cols - 1) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, constrained_layout=True)
    axes_array = np.atleast_1d(axes).reshape(n_rows, n_cols)
    axes_list = list(axes_array.flat)

    if frame.empty:
        for ax in axes_list:
            ax.set_visible(False)
        return fig

    grouped = frame.groupby("algorithm", sort=True)
    for metric, ax in zip(metrics, axes_list):
        if metric not in frame.columns:
            ax.set_visible(False)
            continue
        for algorithm, algo_frame in grouped:
            sub = algo_frame[["iteration", metric]].copy()
            sub = sub.sort_values("iteration")
            valid = sub.dropna(subset=[metric])
            if valid.empty:
                continue
            ax.plot(
                valid["iteration"].to_numpy(),
                valid[metric].to_numpy(),
                marker=None,
                linewidth=2.2,
                label=algorithm,
            )
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlabel("iteration")
        ax.set_ylabel(metric.replace("_", " "))
        ax.grid(alpha=0.25)

    for ax in axes_list[len(metrics) :]:
        ax.set_visible(False)

    handles, labels = axes_list[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
    return fig


def plot_algorithm_summary_dashboard(
    results: dict[str, dict[str, Any]],
    *,
    figsize: tuple[int, int] = (16, 10),
) -> Any:
    plt = _require_matplotlib()
    summary = comparison_summary_frame(results)
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)

    if summary.empty:
        for ax in axes.flat:
            ax.set_visible(False)
        return fig

    labels = summary.index.tolist()
    x = np.arange(len(labels))
    width = 0.36
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(labels), 1)))

    axes[0, 0].bar(x - width / 2.0, summary["before_distance"], width=width, color="#b8c0cc", label="untrained policy")
    axes[0, 0].bar(x + width / 2.0, summary["after_distance"], width=width, color="#2a9d8f", label="trained policy")
    axes[0, 0].set_title("Route Distance: Untrained vs Trained Policy")
    axes[0, 0].set_ylabel("total distance per episode")
    axes[0, 0].set_xticks(x, labels)
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.25, axis="y")

    axes[0, 1].bar(x - width / 2.0, summary["before_dry_runs"], width=width, color="#d9d9d9", label="untrained policy")
    axes[0, 1].bar(x + width / 2.0, summary["after_dry_runs"], width=width, color="#e76f51", label="trained policy")
    axes[0, 1].set_title("Dry Runs: Untrained vs Trained Policy")
    axes[0, 1].set_ylabel("count")
    axes[0, 1].set_xticks(x, labels)
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.25, axis="y")

    distance_bars = axes[1, 0].bar(x, summary["distance_improvement_pct"], color=colors)
    axes[1, 0].axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    axes[1, 0].set_title("Distance Improvement % (Trained vs Untrained)")
    axes[1, 0].set_ylabel("% improvement")
    axes[1, 0].set_xticks(x, labels)
    axes[1, 0].grid(alpha=0.25, axis="y")
    axes[1, 0].bar_label(distance_bars, fmt="%.1f", padding=3, fontsize=9)

    eval_cols = [
        "eval_total_travel_distance",
        "eval_dry_runs",
        "eval_average_stock_levels_percent",
    ]
    available_eval_cols = [col for col in eval_cols if col in summary.columns]
    if available_eval_cols:
        eval_frame = summary[available_eval_cols].copy()
        renamed = {
            "eval_total_travel_distance": "total_travel_distance",
            "eval_dry_runs": "dry_runs",
            "eval_average_stock_levels_percent": "average_stock_levels_percent",
        }
        eval_frame = eval_frame.rename(columns=renamed)
        eval_frame.plot(
            kind="bar",
            ax=axes[1, 1],
            color=["#457b9d", "#f4a261", "#2a9d8f"][: len(eval_frame.columns)],
        )
        axes[1, 1].set_title("Final Evaluation KPIs")
        axes[1, 1].set_ylabel("value")
        axes[1, 1].tick_params(axis="x", rotation=0)
        axes[1, 1].grid(alpha=0.2, axis="y")
        axes[1, 1].legend(loc="upper right")
    else:
        axes[1, 1].set_visible(False)

    return fig


def plot_cpsat_comparison(
    comparison_frame: Any,
    *,
    figsize: tuple[int, int] = (14, 5),
) -> Any:
    plt = _require_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    if comparison_frame is None or getattr(comparison_frame, "empty", True):
        for ax in axes.flat:
            ax.set_visible(False)
        return fig

    plot_frame = comparison_frame.copy()

    if "total_travel_distance" in plot_frame.columns:
        metric_frame = plot_frame[["total_travel_distance"]].dropna()
        colors = ["#2a9d8f" if idx != "cpsat" else "#264653" for idx in metric_frame.index]
        bars = axes[0].bar(
            np.arange(len(metric_frame)),
            metric_frame["total_travel_distance"],
            color=colors,
        )
        axes[0].set_title("Total Travel Distance")
        axes[0].set_ylabel("distance")
        axes[0].set_xticks(np.arange(len(metric_frame)), metric_frame.index, rotation=0)
        axes[0].grid(alpha=0.25, axis="y")
        axes[0].bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    else:
        axes[0].set_visible(False)

    if "dry_runs" in plot_frame.columns:
        metric_frame = plot_frame[["dry_runs"]].dropna()
        colors = ["#2a9d8f" if idx != "cpsat" else "#264653" for idx in metric_frame.index]
        bars = axes[1].bar(
            np.arange(len(metric_frame)),
            metric_frame["dry_runs"],
            color=colors,
        )
        axes[1].set_title("Dry Runs")
        axes[1].set_ylabel("count")
        axes[1].set_xticks(np.arange(len(metric_frame)), metric_frame.index, rotation=0)
        axes[1].grid(alpha=0.25, axis="y")
        axes[1].bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    else:
        axes[1].set_visible(False)

    return fig


def plot_algorithm_tradeoffs(
    results: dict[str, dict[str, Any]],
    *,
    x_key: str = "after_total_travel_distance",
    y_key: str = "after_average_stock_levels_percent",
    figsize: tuple[int, int] = (8, 6),
) -> Any:
    plt = _require_matplotlib()
    summary = comparison_summary_frame(results)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    if summary.empty:
        return fig

    ax.scatter(summary[x_key], summary[y_key], s=160, c=np.arange(len(summary)), cmap="tab10")
    for algorithm, row in summary.iterrows():
        ax.text(row[x_key], row[y_key], f" {algorithm}", va="center", fontsize=10)
    ax.set_xlabel(x_key.replace("_", " ").title())
    ax.set_ylabel(y_key.replace("_", " ").title())
    ax.set_title("Trade-Off: Distance vs Average Stock Level %")
    ax.grid(alpha=0.2)
    return fig


def plot_episode_dashboard(
    env: MPPSRPEnv,
    episode: dict[str, Any],
    *,
    figsize: tuple[int, int] = (14, 10),
) -> Any:
    plt = _require_matplotlib()
    step_frame = episode_to_frame(episode)
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)

    _plot_episode_routes(env, episode, ax=axes[0, 0])

    if not step_frame.empty:
        reward_cols = [col for col in step_frame.columns if col.startswith("reward_")]
        step_frame.plot(
            x="step",
            y=reward_cols,
            ax=axes[0, 1],
            title="Reward Components per Step",
        )
        step_frame.plot(
            x="step",
            y=["cumulative_distance", "cumulative_stockout"],
            ax=axes[1, 0],
            title="Cumulative Metrics",
        )
    else:
        axes[0, 1].set_title("Reward Components per Step")
        axes[1, 0].set_title("Cumulative Metrics")

    plot_inventory_dynamics(
        env,
        episode,
        ax=axes[1, 1],
        title="Station Fill Ratio By Day",
    )

    for ax in axes.flat:
        ax.grid(alpha=0.2)
    return fig


def _plot_episode_routes(env: MPPSRPEnv, episode: dict[str, Any], *, ax: Any) -> None:
    plt = _require_matplotlib()
    node_coords = np.vstack([env.instance.depot_coord[None, :], env.instance.station_coords])
    colors = plt.cm.get_cmap("tab10", env.instance.n_vehicles)

    plot_instance_map(env, ax=ax, annotate=True)
    for step in episode["steps"]:
        start = node_coords[step["from_node"]]
        end = node_coords[step["to_node"]]
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=colors(step["vehicle"]),
            linewidth=2.2,
            alpha=0.75,
        )
    ax.set_title("Episode Route Trace")


def _to_tensor_dict(obs: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    tensor_obs: dict[str, torch.Tensor] = {}
    for key, value in obs.items():
        tensor = torch.as_tensor(value, device=device)
        if tensor.ndim == 0:
            tensor = tensor.unsqueeze(0)
        tensor_obs[key] = tensor.unsqueeze(0)
    return tensor_obs


def _resolve_device(policy: HierarchicalPolicy, device: torch.device | str | None) -> torch.device:
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


def _rollout_for_display(
    env: MPPSRPEnv,
    policy: HierarchicalPolicy,
    *,
    deterministic: bool,
    preferences: torch.Tensor | np.ndarray | None,
    device: torch.device | str | None,
) -> dict[str, Any]:
    was_training = policy.training
    policy.eval()
    try:
        return rollout_episode(
            env,
            policy,
            deterministic=deterministic,
            preferences=preferences,
            device=device,
        )
    finally:
        if was_training:
            policy.train()


def _reward_components_to_dict(
    components: tuple[str, ...] | list[str],
    reward_vec: np.ndarray | torch.Tensor,
) -> dict[str, float]:
    values = np.asarray(reward_vec, dtype=np.float32)
    return {
        f"reward_{component}": float(values[idx])
        for idx, component in enumerate(components)
    }


def _require_matplotlib() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised in notebook runtime
        raise ImportError(
            "matplotlib is required for notebook visualizations. Install requirements.txt."
        ) from exc
    return plt


def _require_pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - exercised in notebook runtime
        raise ImportError(
            "pandas is required for notebook dataframes. Install requirements.txt."
        ) from exc
    return pd


def _config_to_dict(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    return dict(cfg)


def _maybe_create_axis(ax: Any | None, plt: Any, *, figsize: tuple[int, int]) -> tuple[Any, Any]:
    if ax is None:
        fig, axis = plt.subplots(figsize=figsize, constrained_layout=True)
        return fig, axis
    return ax.figure, ax


def _is_missing_scalar(value: Any, pd: Any) -> bool:
    if isinstance(value, (list, tuple, np.ndarray, dict)):
        return False
    try:
        return bool(pd.isna(value))
    except TypeError:
        return False


def _average_stock_levels_like_episode(
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
