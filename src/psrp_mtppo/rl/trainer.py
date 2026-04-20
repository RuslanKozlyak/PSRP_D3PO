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
    inventory_latents: list[np.ndarray] = field(default_factory=list)
    routes: list[list[int]] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    inventory_log_probs: list[float] = field(default_factory=list)
    routing_log_probs: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    returns: list[float] = field(default_factory=list)
    advantages: list[float] = field(default_factory=list)
    infos: list[dict[str, Any]] = field(default_factory=list)


def _stack_obs_dict(observations: list[dict[str, np.ndarray]], device: torch.device) -> dict[str, torch.Tensor]:
    keys = observations[0].keys()
    out: dict[str, torch.Tensor] = {}
    for key in keys:
        arr = np.stack([np.asarray(obs[key]) for obs in observations])
        out[key] = torch.as_tensor(arr, dtype=torch.float32, device=device)
    return out


class ParallelEnvRunner:
    """Synchronous vector env wrapper that keeps the policy forward batched."""

    def __init__(self, env_config: EnvironmentConfig, num_envs: int, seed_offset: int = 0) -> None:
        self.env_config = env_config
        self.envs = [IRPVMIEnv(env_config) for _ in range(num_envs)]
        self.seed_offset = seed_offset
        self.active = [True] * num_envs
        self._observations: list[dict[str, np.ndarray]] = [None] * num_envs  # type: ignore[list-item]

    def reset(self, base_seed: int) -> list[dict[str, np.ndarray]]:
        self.active = [True] * len(self.envs)
        for idx, env in enumerate(self.envs):
            obs, _info = env.reset(seed=base_seed + idx + self.seed_offset)
            self._observations[idx] = obs
        return list(self._observations)

    def step(
        self,
        replenishments: np.ndarray,
        routes: list[list[int]],
    ) -> tuple[list[dict[str, np.ndarray]], list[float], list[bool], list[dict[str, Any]]]:
        new_obs: list[dict[str, np.ndarray]] = []
        rewards: list[float] = []
        dones: list[bool] = []
        infos: list[dict[str, Any]] = []
        for idx, env in enumerate(self.envs):
            if not self.active[idx]:
                new_obs.append(self._observations[idx])
                rewards.append(0.0)
                dones.append(True)
                infos.append({})
                continue
            obs, reward, terminated, _truncated, info = env.step(
                JointAction(replenishment=replenishments[idx].astype(np.float32), route=routes[idx])
            )
            new_obs.append(obs)
            rewards.append(float(reward))
            dones.append(bool(terminated))
            infos.append(info)
            self._observations[idx] = obs
            if terminated:
                self.active[idx] = False
        return new_obs, rewards, dones, infos

    def any_active(self) -> bool:
        return any(self.active)


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
        device_name = training_config.resolved_device()
        if device_name.startswith("cuda") and not torch.cuda.is_available():
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

    # ------------------------------------------------------------------
    # Trace helpers
    # ------------------------------------------------------------------

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

        self.model.eval()
        try:
            while not terminated:
                tensor_obs = _stack_obs_dict([obs], self.device)
                with torch.no_grad():
                    policy_output = self.model.act(tensor_obs, greedy=greedy)
                replenishment = policy_output.replenishment[0].cpu().numpy().astype(np.float32)
                route = policy_output.routes[0]
                next_obs, reward, terminated, _truncated, info = env.step(
                    JointAction(replenishment=replenishment, route=route)
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
        self.model.train()

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

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def _num_parallel_envs(self) -> int:
        return max(1, int(getattr(self.training_config, "num_envs", 1) or 1))

    def collect_batch(self, batch_size: int | None = None, show_progress: bool = False) -> RolloutBatch:
        target_steps = batch_size or self.training_config.train_batch_size
        batch = RolloutBatch()
        num_envs = self._num_parallel_envs()
        runner = ParallelEnvRunner(self.env_config, num_envs=num_envs)

        episode_rewards: list[list[float]] = [[] for _ in range(num_envs)]
        episode_values: list[list[float]] = [[] for _ in range(num_envs)]
        episode_obs: list[list[dict[str, np.ndarray]]] = [[] for _ in range(num_envs)]
        episode_inventory: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
        episode_inventory_latents: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
        episode_routes: list[list[list[int]]] = [[] for _ in range(num_envs)]
        episode_inventory_lp: list[list[float]] = [[] for _ in range(num_envs)]
        episode_routing_lp: list[list[float]] = [[] for _ in range(num_envs)]
        episode_infos: list[list[dict[str, Any]]] = [[] for _ in range(num_envs)]

        progress = tqdm(
            total=target_steps,
            desc="Collect rollout",
            leave=False,
            dynamic_ncols=True,
            disable=not show_progress,
        )

        rng = np.random.default_rng(int(np.random.randint(0, 10_000)))
        episodes_collected = 0

        try:
            while len(batch.rewards) < target_steps:
                base_seed = int(rng.integers(0, 10_000_000))
                observations = runner.reset(base_seed)
                active_mask = [True] * num_envs

                while any(active_mask):
                    active_indices = [idx for idx, flag in enumerate(active_mask) if flag]
                    active_obs = [observations[idx] for idx in active_indices]
                    tensor_obs = _stack_obs_dict(active_obs, self.device)
                    with torch.no_grad():
                        policy_output = self.model.act(tensor_obs, greedy=False)
                    replenishments_np = policy_output.replenishment.cpu().numpy().astype(np.float32)
                    latents_np = policy_output.inventory_latent.cpu().numpy().astype(np.float32)
                    routes = policy_output.routes
                    inv_logp = policy_output.inventory_log_prob.detach().cpu().numpy()
                    route_logp = policy_output.routing_log_prob.detach().cpu().numpy()
                    values = policy_output.value.detach().cpu().numpy()

                    all_repl = np.zeros((num_envs, self.env_config.num_retailers), dtype=np.float32)
                    all_latents = np.zeros((num_envs, self.env_config.num_retailers), dtype=np.float32)
                    all_routes: list[list[int]] = [[0, 0] for _ in range(num_envs)]
                    for local_idx, env_idx in enumerate(active_indices):
                        all_repl[env_idx] = replenishments_np[local_idx]
                        all_latents[env_idx] = latents_np[local_idx]
                        all_routes[env_idx] = routes[local_idx]

                    for local_idx, env_idx in enumerate(active_indices):
                        episode_obs[env_idx].append(observations[env_idx])
                        episode_inventory[env_idx].append(all_repl[env_idx].copy())
                        episode_inventory_latents[env_idx].append(all_latents[env_idx].copy())
                        episode_routes[env_idx].append(all_routes[env_idx])
                        episode_inventory_lp[env_idx].append(float(inv_logp[local_idx]))
                        episode_routing_lp[env_idx].append(float(route_logp[local_idx]))
                        episode_values[env_idx].append(float(values[local_idx]))

                    new_observations, rewards, dones, infos = runner.step(all_repl, all_routes)
                    for local_idx, env_idx in enumerate(active_indices):
                        episode_rewards[env_idx].append(rewards[env_idx])
                        episode_infos[env_idx].append(infos[env_idx])
                        observations[env_idx] = new_observations[env_idx]
                        if dones[env_idx]:
                            active_mask[env_idx] = False

                for env_idx in range(num_envs):
                    if not episode_rewards[env_idx]:
                        continue
                    rewards_arr = episode_rewards[env_idx]
                    values_arr = episode_values[env_idx]
                    returns: list[float] = []
                    running = 0.0
                    for reward in reversed(rewards_arr):
                        running = reward + self.training_config.gamma * running
                        returns.append(running)
                    returns.reverse()
                    advantages = [ret - val for ret, val in zip(returns, values_arr)]

                    batch.observations.extend(episode_obs[env_idx])
                    batch.inventory_actions.extend(episode_inventory[env_idx])
                    batch.inventory_latents.extend(episode_inventory_latents[env_idx])
                    batch.routes.extend(episode_routes[env_idx])
                    batch.rewards.extend(rewards_arr)
                    batch.inventory_log_probs.extend(episode_inventory_lp[env_idx])
                    batch.routing_log_probs.extend(episode_routing_lp[env_idx])
                    batch.values.extend(values_arr)
                    batch.returns.extend(returns)
                    batch.advantages.extend(advantages)
                    batch.infos.extend(episode_infos[env_idx])

                    progress.update(len(rewards_arr))
                    episodes_collected += 1
                    progress.set_postfix(
                        steps=len(batch.rewards),
                        episodes=episodes_collected,
                        last_episode_return=round(float(sum(rewards_arr)), 2),
                    )

                for env_idx in range(num_envs):
                    episode_rewards[env_idx] = []
                    episode_values[env_idx] = []
                    episode_obs[env_idx] = []
                    episode_inventory[env_idx] = []
                    episode_inventory_latents[env_idx] = []
                    episode_routes[env_idx] = []
                    episode_inventory_lp[env_idx] = []
                    episode_routing_lp[env_idx] = []
                    episode_infos[env_idx] = []
        finally:
            progress.close()

        return batch

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

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
        inventory_latents = torch.as_tensor(
            np.stack(batch.inventory_latents),
            dtype=torch.float32,
            device=self.device,
        )
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
                    mini_inventory_latents = inventory_latents[batch_indices]
                    mini_advantages = advantages[batch_indices]
                    mini_returns = returns[batch_indices]
                    mini_old_inventory_log_probs = old_inventory_log_probs[batch_indices]
                    mini_old_routing_log_probs = old_routing_log_probs[batch_indices]
                    mini_old_values = old_values[batch_indices]
                    mini_routes = [batch.routes[idx] for idx in batch_indices]

                    current_inventory_log_prob, inventory_entropy = self.model.inventory_actor.evaluate_actions(
                        mini_obs,
                        mini_inventory_latents,
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

                    current_routing_log_prob, routing_entropy = self.model.routing_actor.evaluate_routes(
                        mini_obs,
                        mini_inventory_actions,
                        mini_routes,
                    )
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
        self.model.eval()
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
                tensor_obs = _stack_obs_dict([obs], self.device)
                with torch.no_grad():
                    policy_output = self.model.act(tensor_obs, greedy=True)
                replenishment = policy_output.replenishment[0].cpu().numpy().astype(np.float32)
                route = policy_output.routes[0]
                next_obs, reward, terminated, _truncated, info = env.step(
                    JointAction(replenishment=replenishment, route=route)
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

        self.model.train()
        summary = pd.DataFrame(metrics).mean(numeric_only=True).to_dict()
        return {f"eval_{key}": float(value) for key, value in summary.items()}
