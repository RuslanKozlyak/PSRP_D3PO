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
    n_objectives = int(algo_cfg.get("n_objectives", 4))
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
    done = False
    truncated = False
    step_idx = 0
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
        delivered_estimate = float(
            quantity_upper.sum() if policy.quantity_mode == "deterministic" else (quantity_action * quantity_upper).sum()
        )

        from_node = int(env.state.vehicle_node[vehicle_idx])
        step_record = {
            "step": step_idx,
            "day": int(env.state.day),
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
        step_record.update(
            {
                "reward_distance": float(reward_vec[0]),
                "reward_holding": float(reward_vec[1]),
                "reward_safety": float(reward_vec[2]),
                "reward_time_window": float(reward_vec[3]),
                "cumulative_distance": float(info["cumulative_distance"]),
                "cumulative_stockout": float(info["cumulative_stockout"]),
            }
        )
        steps.append(step_record)
        inventory_snapshots.append(env.state.station_inventory.copy())
        vehicle_times.append(env.state.vehicle_time.copy())
        step_idx += 1

    return {
        "reset_info": reset_info,
        "steps": steps,
        "terminated": done,
        "truncated": truncated,
        "terminal_info": info,
        "inventory_snapshots": np.stack(inventory_snapshots, axis=0),
        "vehicle_times": np.stack(vehicle_times, axis=0),
        "preferences": None if pref_tensor is None else pref_tensor.detach().cpu().numpy(),
    }


def episode_to_frame(episode: dict[str, Any]) -> Any:
    pd = _require_pandas()
    return pd.DataFrame(episode["steps"])


def episode_summary(episode: dict[str, Any]) -> dict[str, float]:
    steps = episode["steps"]
    if not steps:
        return {
            "n_steps": 0.0,
            "distance": 0.0,
            "stockout": 0.0,
            "reward_distance": 0.0,
            "reward_holding": 0.0,
            "reward_safety": 0.0,
            "reward_time_window": 0.0,
        }

    return {
        "n_steps": float(len(steps)),
        "distance": float(episode["terminal_info"]["cumulative_distance"]),
        "stockout": float(episode["terminal_info"]["cumulative_stockout"]),
        "reward_distance": float(sum(step["reward_distance"] for step in steps)),
        "reward_holding": float(sum(step["reward_holding"] for step in steps)),
        "reward_safety": float(sum(step["reward_safety"] for step in steps)),
        "reward_time_window": float(sum(step["reward_time_window"] for step in steps)),
    }


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


def plot_action_masks(
    obs: dict[str, np.ndarray],
    *,
    axes: tuple[Any, Any] | None = None,
) -> tuple[Any, Any]:
    plt = _require_matplotlib()
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    else:
        fig = axes[0].figure

    node_image = axes[0].imshow(obs["mask_node"], aspect="auto", cmap="Greys", vmin=0, vmax=1)
    axes[0].set_title("Node Mask per Vehicle")
    axes[0].set_xlabel("Node (0 = depot)")
    axes[0].set_ylabel("Vehicle")
    axes[0].set_yticks(np.arange(obs["mask_node"].shape[0]))

    quantity_slots = obs["mask_quantity"].sum(axis=(1, 2))
    axes[1].bar(np.arange(obs["mask_vehicle"].shape[0]), quantity_slots, color="#4c956c")
    axes[1].set_title("Available Quantity Slots")
    axes[1].set_xlabel("Vehicle")
    axes[1].set_ylabel("Compartment/Product slots")
    axes[1].set_ylim(0, max(float(quantity_slots.max()) + 1.0, 1.0))

    fig.colorbar(node_image, ax=axes[0], fraction=0.046, pad=0.04)
    return fig, axes


def plot_training_history(history: list[dict[str, float]], *, figsize: tuple[int, int] = (12, 8)) -> Any:
    plt = _require_matplotlib()
    frame = history_to_frame(history)
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)

    if not frame.empty:
        frame.plot(x="iteration", y=["policy_loss", "value_loss"], ax=axes[0, 0], title="Losses")
        frame.plot(x="iteration", y=["entropy"], ax=axes[0, 1], title="Entropy")

        eval_cols = [col for col in ("eval_distance", "eval_stockout") if col in frame.columns]
        if eval_cols:
            frame.plot(x="iteration", y=eval_cols, ax=axes[1, 0], title="Eval Metrics")
        reward_cols = [
            col
            for col in (
                "eval_reward_distance",
                "eval_reward_holding",
                "eval_reward_safety",
                "eval_reward_time_window",
            )
            if col in frame.columns
        ]
        if reward_cols:
            frame.plot(x="iteration", y=reward_cols, ax=axes[1, 1], title="Eval Reward Components")

    for ax in axes.flat:
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
        step_frame.plot(
            x="step",
            y=["reward_distance", "reward_holding", "reward_safety", "reward_time_window"],
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

    plot_inventory_heatmap(
        env,
        episode["inventory_snapshots"][-1],
        ax=axes[1, 1],
        title="Final Inventory Fill Ratio",
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


def _maybe_create_axis(ax: Any | None, plt: Any, *, figsize: tuple[int, int]) -> tuple[Any, Any]:
    if ax is None:
        fig, axis = plt.subplots(figsize=figsize, constrained_layout=True)
        return fig, axis
    return ax.figure, ax
