from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm.auto import tqdm

from ..config import EnvironmentConfig, ModelConfig, TrainingConfig
from ..env import IRPVMIEnv, JointAction
from ..models.mtppo import MTPPOModel
from ..utils import set_seed, stack_observations


@dataclass(slots=True)
class RolloutBatch:
    observations: list[dict[str, np.ndarray]] = field(default_factory=list)
    inventory_actions: list[np.ndarray] = field(default_factory=list)
    routes: list[list[int]] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    inventory_log_probs: list[float] = field(default_factory=list)
    routing_log_probs: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    returns: list[float] = field(default_factory=list)
    advantages: list[float] = field(default_factory=list)
    infos: list[dict[str, Any]] = field(default_factory=list)


class MTPPOTrainer:
    def __init__(
        self,
        env_config: EnvironmentConfig,
        model_config: ModelConfig,
        training_config: TrainingConfig,
    ) -> None:
        self.env_config = env_config
        self.model_config = model_config
        self.training_config = training_config
        device_name = training_config.device
        if device_name == "cuda" and not torch.cuda.is_available():
            device_name = "cpu"
        self.device = torch.device(device_name)
        set_seed(training_config.seed)

        self.model = MTPPOModel(env_config, model_config).to(self.device)
        self.inventory_optimizer = torch.optim.Adam(
            self.model.inventory_actor.parameters(),
            lr=training_config.learning_rate,
        )
        self.routing_optimizer = torch.optim.Adam(
            self.model.routing_actor.parameters(),
            lr=training_config.learning_rate,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.model.critic.parameters(),
            lr=training_config.learning_rate,
        )
        self.history: list[dict[str, float]] = []

    def _build_trace_artifacts(
        self,
        daily_rows: list[dict[str, float]],
        inventory_before_rows: list[np.ndarray],
        replenishment_rows: list[np.ndarray],
        demand_rows: list[np.ndarray],
        ending_inventory_rows: list[np.ndarray],
        lost_sales_rows: list[np.ndarray],
    ) -> dict[str, pd.DataFrame]:
        retailer_columns = [f"retailer_{idx + 1}" for idx in range(self.env_config.num_retailers)]
        day_index = pd.Index(range(len(daily_rows)), name="day")
        return {
            "daily": pd.DataFrame(daily_rows),
            "inventory_before": pd.DataFrame(inventory_before_rows, columns=retailer_columns, index=day_index),
            "replenishment": pd.DataFrame(replenishment_rows, columns=retailer_columns, index=day_index),
            "demand": pd.DataFrame(demand_rows, columns=retailer_columns, index=day_index),
            "ending_inventory": pd.DataFrame(ending_inventory_rows, columns=retailer_columns, index=day_index),
            "lost_sales": pd.DataFrame(lost_sales_rows, columns=retailer_columns, index=day_index),
        }

    def rollout_episode(
        self,
        seed: int = 123,
        greedy: bool = True,
        show_progress: bool = False,
    ) -> tuple[dict[str, float], dict[str, pd.DataFrame]]:
        env = IRPVMIEnv(self.env_config)
        obs, _ = env.reset(seed=seed)
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
        iterator = tqdm(
            total=self.env_config.horizon_days,
            desc="Episode trace",
            leave=False,
            dynamic_ncols=True,
            disable=not show_progress,
        )

        try:
            while not terminated:
                tensor_obs = {
                    key: torch.as_tensor(value, dtype=torch.float32, device=self.device).unsqueeze(0)
                    for key, value in obs.items()
                }
                with torch.no_grad():
                    policy_output = self.model.act(tensor_obs, greedy=greedy)
                replenishment = policy_output.replenishment.squeeze(0).cpu().numpy().astype(np.float32)
                next_obs, reward, terminated, _truncated, info = env.step(
                    JointAction(replenishment=replenishment, route=policy_output.route)
                )

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
                obs = next_obs
                iterator.update(1)
                iterator.set_postfix(
                    day=info["day"] + 1,
                    reward=f"{summary['reward']:.1f}",
                    sum_cost=f"{summary['inventory_cost'] + summary['route_cost']:.1f}",
                )
        finally:
            iterator.close()

        trace_tables = self._build_trace_artifacts(
            daily_rows=daily_rows,
            inventory_before_rows=inventory_before_rows,
            replenishment_rows=replenishment_rows,
            demand_rows=demand_rows,
            ending_inventory_rows=ending_inventory_rows,
            lost_sales_rows=lost_sales_rows,
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
        return episode_summary, trace_tables

    def collect_batch(self, batch_size: int | None = None, show_progress: bool = False) -> RolloutBatch:
        target_steps = batch_size or self.training_config.train_batch_size
        batch = RolloutBatch()
        progress = tqdm(
            total=target_steps,
            desc="Collect rollout",
            leave=False,
            dynamic_ncols=True,
            disable=not show_progress,
        )
        episodes_collected = 0

        try:
            while len(batch.rewards) < target_steps:
                env = IRPVMIEnv(self.env_config)
                obs, _info = env.reset(seed=int(np.random.randint(0, 10_000)))
                episode_rewards: list[float] = []
                episode_values: list[float] = []
                episode_start = len(batch.rewards)

                terminated = False
                while not terminated:
                    tensor_obs = {
                        key: torch.as_tensor(value, dtype=torch.float32, device=self.device).unsqueeze(0)
                        for key, value in obs.items()
                    }
                    with torch.no_grad():
                        policy_output = self.model.act(tensor_obs, greedy=False)

                    replenishment = policy_output.replenishment.squeeze(0).cpu().numpy().astype(np.float32)
                    joint_action = JointAction(replenishment=replenishment, route=policy_output.route)
                    next_obs, reward, terminated, _truncated, info = env.step(joint_action)

                    batch.observations.append(obs)
                    batch.inventory_actions.append(replenishment)
                    batch.routes.append(policy_output.route)
                    batch.rewards.append(float(reward))
                    batch.inventory_log_probs.append(float(policy_output.inventory_log_prob.item()))
                    batch.routing_log_probs.append(float(policy_output.routing_log_prob.item()))
                    batch.values.append(float(policy_output.value.item()))
                    batch.infos.append(info)
                    episode_rewards.append(float(reward))
                    episode_values.append(float(policy_output.value.item()))
                    obs = next_obs

                returns = []
                running_return = 0.0
                for reward in reversed(episode_rewards):
                    running_return = reward + self.training_config.gamma * running_return
                    returns.append(running_return)
                returns.reverse()
                advantages = [ret - val for ret, val in zip(returns, episode_values)]

                batch.returns.extend(returns)
                batch.advantages.extend(advantages)

                episode_end = len(batch.rewards)
                assert episode_end - episode_start == len(returns)
                progress.update(episode_end - episode_start)
                episodes_collected += 1
                progress.set_postfix(
                    steps=len(batch.rewards),
                    episodes=episodes_collected,
                    last_episode_return=round(sum(episode_rewards), 2),
                )
        finally:
            progress.close()

        return batch

    def _critic_loss(
        self,
        predicted_values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
    ) -> torch.Tensor:
        value_clipped = old_values + (predicted_values - old_values).clamp(
            -self.training_config.value_clip_param,
            self.training_config.value_clip_param,
        )
        loss_unclipped = (predicted_values - returns).pow(2)
        loss_clipped = (value_clipped - returns).pow(2)
        return 0.5 * torch.maximum(loss_unclipped, loss_clipped).mean()

    def update(self, batch: RolloutBatch, show_progress: bool = False) -> dict[str, float]:
        observations = stack_observations(batch.observations, self.device)
        inventory_actions = torch.as_tensor(
            np.stack(batch.inventory_actions),
            dtype=torch.float32,
            device=self.device,
        )
        returns = torch.as_tensor(batch.returns, dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(batch.advantages, dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-6)
        old_inventory_log_probs = torch.as_tensor(
            batch.inventory_log_probs,
            dtype=torch.float32,
            device=self.device,
        )
        old_routing_log_probs = torch.as_tensor(
            batch.routing_log_probs,
            dtype=torch.float32,
            device=self.device,
        )
        old_values = torch.as_tensor(batch.values, dtype=torch.float32, device=self.device)

        num_items = returns.shape[0]
        indices = np.arange(num_items)
        losses: dict[str, list[float]] = {
            "inventory_actor_loss": [],
            "routing_actor_loss": [],
            "critic_loss": [],
            "mean_return": [float(returns.mean().item())],
        }
        total_minibatches = self.training_config.ppo_epochs * ceil(num_items / self.training_config.minibatch_size)
        progress = tqdm(
            total=total_minibatches,
            desc="PPO update",
            leave=False,
            dynamic_ncols=True,
            disable=not show_progress,
        )

        try:
            for epoch in range(self.training_config.ppo_epochs):
                np.random.shuffle(indices)
                for start in range(0, num_items, self.training_config.minibatch_size):
                    batch_indices = indices[start : start + self.training_config.minibatch_size]
                    if len(batch_indices) == 0:
                        continue

                    mini_obs = {key: value[batch_indices] for key, value in observations.items()}
                    mini_inventory_actions = inventory_actions[batch_indices]
                    mini_advantages = advantages[batch_indices]
                    mini_returns = returns[batch_indices]
                    mini_old_inventory_log_probs = old_inventory_log_probs[batch_indices]
                    mini_old_routing_log_probs = old_routing_log_probs[batch_indices]
                    mini_old_values = old_values[batch_indices]
                    mini_routes = [batch.routes[idx] for idx in batch_indices]

                    current_inventory_log_prob, inventory_entropy = self.model.inventory_actor.evaluate_actions(
                        mini_obs,
                        mini_inventory_actions,
                    )
                    inventory_ratio = torch.exp(current_inventory_log_prob - mini_old_inventory_log_probs)
                    inventory_loss = -torch.min(
                        inventory_ratio * mini_advantages,
                        inventory_ratio.clamp(
                            1.0 - self.training_config.clip_param,
                            1.0 + self.training_config.clip_param,
                        )
                        * mini_advantages,
                    ).mean()
                    inventory_kl = (mini_old_inventory_log_probs - current_inventory_log_prob).mean()
                    inventory_loss = inventory_loss + self.training_config.kl_coefficient * inventory_kl
                    inventory_loss = inventory_loss - self.training_config.entropy_coefficient * inventory_entropy.mean()

                    routing_log_probs = []
                    routing_entropies = []
                    for obs_index, route in enumerate(mini_routes):
                        single_obs = {key: value[obs_index : obs_index + 1] for key, value in mini_obs.items()}
                        log_prob, entropy = self.model.routing_actor.evaluate_sequence(
                            single_obs,
                            mini_inventory_actions[obs_index : obs_index + 1],
                            route,
                        )
                        routing_log_probs.append(log_prob)
                        routing_entropies.append(entropy)
                    current_routing_log_prob = torch.cat(routing_log_probs, dim=0)
                    routing_entropy = torch.cat(routing_entropies, dim=0)
                    routing_ratio = torch.exp(current_routing_log_prob - mini_old_routing_log_probs)
                    routing_loss = -torch.min(
                        routing_ratio * mini_advantages,
                        routing_ratio.clamp(
                            1.0 - self.training_config.clip_param,
                            1.0 + self.training_config.clip_param,
                        )
                        * mini_advantages,
                    ).mean()
                    routing_kl = (mini_old_routing_log_probs - current_routing_log_prob).mean()
                    routing_loss = routing_loss + self.training_config.kl_coefficient * routing_kl
                    routing_loss = routing_loss - self.training_config.entropy_coefficient * routing_entropy.mean()

                    predicted_values = self.model.critic(mini_obs)
                    critic_loss = self._critic_loss(predicted_values, mini_old_values, mini_returns)

                    self.inventory_optimizer.zero_grad()
                    inventory_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.model.inventory_actor.parameters(),
                        self.training_config.max_grad_norm,
                    )
                    self.inventory_optimizer.step()

                    self.routing_optimizer.zero_grad()
                    routing_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.model.routing_actor.parameters(),
                        self.training_config.max_grad_norm,
                    )
                    self.routing_optimizer.step()

                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(self.model.critic.parameters(), self.training_config.max_grad_norm)
                    self.critic_optimizer.step()

                    losses["inventory_actor_loss"].append(float(inventory_loss.item()))
                    losses["routing_actor_loss"].append(float(routing_loss.item()))
                    losses["critic_loss"].append(float(critic_loss.item()))
                    progress.update(1)
                    progress.set_postfix(
                        epoch=epoch + 1,
                        inv_loss=f"{inventory_loss.item():.3f}",
                        route_loss=f"{routing_loss.item():.3f}",
                        critic=f"{critic_loss.item():.3f}",
                    )
        finally:
            progress.close()

        return {key: float(np.mean(value)) for key, value in losses.items()}

    def train(self, show_progress: bool = False) -> pd.DataFrame:
        history_rows: list[dict[str, float]] = []
        progress = tqdm(
            range(self.training_config.training_iterations),
            desc=f"MTPPO train ({self.env_config.num_retailers} retailers)",
            leave=True,
            dynamic_ncols=True,
            disable=not show_progress,
        )
        for iteration in progress:
            batch = self.collect_batch(show_progress=show_progress)
            update_metrics = self.update(batch, show_progress=show_progress)
            eval_metrics = self.evaluate(self.training_config.evaluation_episodes, show_progress=show_progress)
            row = {"iteration": float(iteration), **update_metrics, **eval_metrics}
            self.history.append(row)
            history_rows.append(row)
            progress.set_postfix(
                mean_return=f"{row['mean_return']:.2f}",
                eval_cost=f"{row['eval_sum_cost']:.2f}",
                fill_rate=f"{row['eval_fill_rate']:.3f}",
            )
        return pd.DataFrame(history_rows)

    def evaluate(self, episodes: int = 4, show_progress: bool = False) -> dict[str, float]:
        metrics: list[dict[str, float]] = []
        iterator = tqdm(
            range(episodes),
            desc="Evaluate",
            leave=False,
            dynamic_ncols=True,
            disable=not show_progress,
        )
        for episode_idx in iterator:
            env = IRPVMIEnv(self.env_config)
            obs, _ = env.reset(seed=episode_idx + 123)
            terminated = False
            episode_metrics = {
                "reward": 0.0,
                "inventory_cost": 0.0,
                "route_distance": 0.0,
                "route_cost": 0.0,
                "fill_rate": 0.0,
                "days": 0.0,
            }
            while not terminated:
                tensor_obs = {
                    key: torch.as_tensor(value, dtype=torch.float32, device=self.device).unsqueeze(0)
                    for key, value in obs.items()
                }
                with torch.no_grad():
                    policy_output = self.model.act(tensor_obs, greedy=True)
                next_obs, reward, terminated, _truncated, info = env.step(
                    JointAction(
                        replenishment=policy_output.replenishment.squeeze(0).cpu().numpy(),
                        route=policy_output.route,
                    )
                )
                episode_metrics["reward"] += reward
                episode_metrics["inventory_cost"] += float(info["inventory_cost"])
                episode_metrics["route_distance"] += float(info["route_distance"])
                episode_metrics["route_cost"] += float(info["route_cost"])
                episode_metrics["fill_rate"] += float(info["fill_rate"])
                episode_metrics["days"] += 1.0
                obs = next_obs

            episode_metrics["fill_rate"] /= max(episode_metrics["days"], 1.0)
            episode_metrics["sum_cost"] = episode_metrics["inventory_cost"] + episode_metrics["route_cost"]
            metrics.append(episode_metrics)
            iterator.set_postfix(
                reward=f"{episode_metrics['reward']:.2f}",
                sum_cost=f"{episode_metrics['sum_cost']:.2f}",
                fill_rate=f"{episode_metrics['fill_rate']:.3f}",
            )

        summary = pd.DataFrame(metrics).mean(numeric_only=True).to_dict()
        return {f"eval_{key}": float(value) for key, value in summary.items()}
