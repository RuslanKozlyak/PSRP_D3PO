from __future__ import annotations

from dataclasses import replace
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm

from .baselines import evaluate_heuristic_baseline
from .config import EnvironmentConfig, ModelConfig, TrainingConfig
from .rl.trainer import MTPPOTrainer


def run_joint_experiment(
    env_config: EnvironmentConfig,
    model_config: ModelConfig | None = None,
    training_config: TrainingConfig | None = None,
    show_progress: bool = False,
) -> tuple[MTPPOTrainer, pd.DataFrame]:
    model_config = model_config or ModelConfig()
    training_config = training_config or TrainingConfig()
    trainer = MTPPOTrainer(env_config, model_config, training_config)
    history = trainer.train(show_progress=show_progress)
    return trainer, history


def run_scale_sweep(
    scales: Iterable[int],
    base_env_config: EnvironmentConfig | None = None,
    model_config: ModelConfig | None = None,
    training_config: TrainingConfig | None = None,
    baseline_solver: str = "ortools",
    show_progress: bool = False,
) -> pd.DataFrame:
    base_env_config = base_env_config or EnvironmentConfig()
    model_config = model_config or ModelConfig()
    training_config = training_config or TrainingConfig()
    rows: list[dict[str, float | str]] = []

    scale_iterator = tqdm(
        list(scales),
        desc="Scale sweep",
        leave=True,
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for scale in scale_iterator:
        env_config = replace(base_env_config, num_retailers=scale)
        trainer, history = run_joint_experiment(
            env_config,
            model_config,
            training_config,
            show_progress=show_progress,
        )
        joint_metrics = trainer.evaluate(training_config.evaluation_episodes)
        baseline_metrics = evaluate_heuristic_baseline(
            env_config,
            episodes=training_config.evaluation_episodes,
            route_solver=baseline_solver,
        ).mean(numeric_only=True)
        rows.append(
            {
                "algorithm": "MTPPO",
                "retailers": scale,
                "inventory_cost": joint_metrics["eval_inventory_cost"],
                "route_distance": joint_metrics["eval_route_distance"],
                "route_cost": joint_metrics["eval_route_cost"],
                "fill_rate": joint_metrics["eval_fill_rate"],
                "sum_cost": joint_metrics["eval_sum_cost"],
                "last_mean_return": float(history["mean_return"].iloc[-1]),
            }
        )
        rows.append(
            {
                "algorithm": f"Heuristic-{baseline_solver}",
                "retailers": scale,
                "inventory_cost": float(baseline_metrics["inventory_cost"]),
                "route_distance": float(baseline_metrics["route_distance"]),
                "route_cost": float(baseline_metrics["route_cost"]),
                "fill_rate": float(baseline_metrics["fill_rate"]),
                "sum_cost": float(baseline_metrics["sum_cost"]),
                "last_mean_return": float("nan"),
            }
        )
        scale_iterator.set_postfix(
            retailers=scale,
            mtppo_sum_cost=f"{joint_metrics['eval_sum_cost']:.2f}",
            baseline_sum_cost=f"{float(baseline_metrics['sum_cost']):.2f}",
        )

    return pd.DataFrame(rows)


def plot_training_history(history: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    history.plot(x="iteration", y="mean_return", ax=axes[0], title="Mean Return")
    history.plot(x="iteration", y="eval_sum_cost", ax=axes[1], title="Eval Sum Cost")
    history.plot(x="iteration", y="eval_fill_rate", ax=axes[2], title="Eval Fill Rate")
    for axis in axes:
        axis.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_training_dashboard(history: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    plotting_plan = [
        ("mean_return", "Mean Return"),
        ("eval_sum_cost", "Evaluation Sum Cost"),
        ("eval_fill_rate", "Evaluation Fill Rate"),
        ("inventory_actor_loss", "Inventory Actor Loss"),
        ("routing_actor_loss", "Routing Actor Loss"),
        ("critic_loss", "Critic Loss"),
    ]
    for axis, (column, title) in zip(axes.flatten(), plotting_plan):
        history.plot(x="iteration", y=column, marker="o", ax=axis, title=title, legend=False)
        axis.grid(alpha=0.3)
        axis.set_xlabel("Iteration")
    fig.tight_layout()
    return fig


def plot_episode_dashboard(daily_trace: pd.DataFrame, title_prefix: str = "Episode") -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    x = daily_trace["day"]

    axes[0, 0].plot(x, daily_trace["reward"], marker="o", label="Reward")
    axes[0, 0].plot(x, -(daily_trace["inventory_cost"] + daily_trace["route_cost"]), linestyle="--", label="-Total Cost")
    axes[0, 0].set_title(f"{title_prefix}: Reward vs Cost")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(x, daily_trace["inventory_cost"], marker="o", label="Inventory Cost")
    axes[0, 1].plot(x, daily_trace["route_cost"], marker="s", label="Route Cost")
    axes[0, 1].plot(x, daily_trace["holding_cost"], linestyle="--", label="Holding Cost")
    axes[0, 1].plot(x, daily_trace["sales_loss_cost"], linestyle="--", label="Sales Loss Cost")
    axes[0, 1].set_title(f"{title_prefix}: Cost Decomposition")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    axes[0, 2].plot(x, daily_trace["route_distance"], marker="o", label="Route Distance")
    axes[0, 2].plot(x, daily_trace["route_stops"], marker="s", label="Route Stops")
    axes[0, 2].plot(x, daily_trace["active_retailers"], marker="^", label="Active Retailers")
    axes[0, 2].set_title(f"{title_prefix}: Routing Load")
    axes[0, 2].legend()
    axes[0, 2].grid(alpha=0.3)

    axes[1, 0].plot(x, daily_trace["fill_rate"], marker="o", label="Fill Rate")
    axes[1, 0].plot(x, daily_trace["lost_sales"], marker="s", label="Lost Sales")
    axes[1, 0].set_title(f"{title_prefix}: Service Level")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(x, daily_trace["avg_inventory_before"], marker="o", label="Avg Inventory Before")
    axes[1, 1].plot(x, daily_trace["avg_inventory_after"], marker="s", label="Avg Inventory After")
    axes[1, 1].plot(x, daily_trace["min_inventory_after"], linestyle="--", label="Min Inventory After")
    axes[1, 1].set_title(f"{title_prefix}: Inventory Levels")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    axes[1, 2].plot(x, daily_trace["total_replenishment"], marker="o", label="Replenishment")
    axes[1, 2].plot(x, daily_trace["total_demand"], marker="s", label="Demand")
    axes[1, 2].plot(x, daily_trace["fulfilled"], linestyle="--", label="Fulfilled")
    axes[1, 2].set_title(f"{title_prefix}: Flow Volume")
    axes[1, 2].legend()
    axes[1, 2].grid(alpha=0.3)

    fig.tight_layout()
    return fig


def plot_retailer_heatmaps(
    ending_inventory: pd.DataFrame,
    replenishment: pd.DataFrame,
    demand: pd.DataFrame,
    title_prefix: str = "Episode",
) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    matrices = [
        (ending_inventory.T, "Ending Inventory"),
        (replenishment.T, "Replenishment"),
        (demand.T, "Demand"),
    ]
    for axis, (matrix, title) in zip(axes, matrices):
        image = axis.imshow(matrix, aspect="auto", interpolation="nearest")
        axis.set_title(f"{title_prefix}: {title}")
        axis.set_xlabel("Day")
        axis.set_ylabel("Retailer")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_scale_sweep(results: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for metric, axis in zip(["inventory_cost", "route_distance", "sum_cost"], axes):
        pivot = results.pivot(index="retailers", columns="algorithm", values=metric)
        pivot.plot(marker="o", ax=axis, title=metric.replace("_", " ").title())
        axis.grid(alpha=0.3)
    fig.tight_layout()
    return fig
