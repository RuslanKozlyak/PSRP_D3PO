from __future__ import annotations

from typing import Any

import torch

from src.algo.base import BaseAlgorithm
from src.algo.rollout import RolloutBatch, index_tree


class PIDController:
    def __init__(self, kp: float, ki: float, kd: float) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0

    def step(self, error: float) -> float:
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative


class PPOLagrangian(BaseAlgorithm):
    def __init__(self, policy: torch.nn.Module, config: dict[str, Any]) -> None:
        super().__init__(policy, config)
        (
            self.reward_indices,
            self.reward_weights,
            self.cost_indices,
            self.cost_targets,
        ) = self._resolve_reward_structure()
        self.lambda_lr = float(self.config.get("lambda_lr", 0.01))
        self.use_pid = bool(self.config.get("use_pid", True))
        pid_cfg = self.config.get("pid", {"kp": 0.1, "ki": 0.01, "kd": 0.01})
        self.pid = PIDController(
            kp=float(pid_cfg.get("kp", 0.1)),
            ki=float(pid_cfg.get("ki", 0.01)),
            kd=float(pid_cfg.get("kd", 0.01)),
        )
        self.lambdas = torch.zeros(len(self.cost_indices), dtype=torch.float32)

    def compute_advantages(self, rollouts: RolloutBatch) -> torch.Tensor:
        reward_weights = self.reward_weights.to(rollouts.rewards_vec.device)
        reward_indices = torch.tensor(self.reward_indices, dtype=torch.long, device=rollouts.rewards_vec.device)
        reward_stream = rollouts.rewards_vec.index_select(dim=-1, index=reward_indices)
        reward_values_stream = rollouts.values_vec.index_select(dim=-1, index=reward_indices)
        reward_scalar = (reward_stream * reward_weights).sum(dim=-1)
        reward_value_scalar = (reward_values_stream * reward_weights).sum(dim=-1)
        bootstrap_value = None
        if rollouts.bootstrap_value is not None:
            bootstrap_value = (
                rollouts.bootstrap_value.index_select(dim=-1, index=reward_indices) * reward_weights
            ).sum(dim=-1)
        reward_adv = self.compute_gae(
            rewards=reward_scalar,
            values=reward_value_scalar,
            dones=rollouts.dones,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            bootstrap_value=bootstrap_value,
        )
        cost_advantages = []
        reward_returns = reward_adv + reward_value_scalar
        cost_returns = []
        for cost_idx in self.cost_indices:
            cost_bootstrap = None
            if rollouts.bootstrap_value is not None:
                cost_bootstrap = rollouts.bootstrap_value[..., cost_idx]
            cost_adv = self.compute_gae(
                rewards=rollouts.rewards_vec[..., cost_idx],
                values=rollouts.values_vec[..., cost_idx],
                dones=rollouts.dones,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                bootstrap_value=cost_bootstrap,
            )
            cost_advantages.append(cost_adv)
            cost_returns.append(cost_adv + rollouts.values_vec[..., cost_idx])

        advantages = torch.stack([reward_adv, *cost_advantages], dim=-1)
        reduce_dims = tuple(range(advantages.ndim - 1))
        adv_mean = advantages.mean(dim=reduce_dims, keepdim=True)
        adv_std = advantages.std(dim=reduce_dims, keepdim=True).clamp_min(1e-6)
        advantages = (advantages - adv_mean) / adv_std
        returns = torch.stack([reward_returns, *cost_returns], dim=-1)
        rollouts.advantages = advantages
        rollouts.returns = returns
        return advantages

    def compute_loss(self, batch: RolloutBatch) -> dict[str, torch.Tensor]:
        new_log_prob, entropy, values_vec = self.policy.evaluate_actions(
            batch.observations,
            batch.actions,
            preferences=batch.preferences,
        )
        assert batch.advantages is not None
        assert batch.returns is not None

        reward_adv = batch.advantages[..., 0]
        cost_adv = batch.advantages[..., 1:]
        combined_adv = reward_adv
        if self.cost_indices:
            combined_adv = reward_adv - (cost_adv * self.lambdas.to(cost_adv.device)).sum(dim=-1)

        log_ratio = new_log_prob - batch.log_probs
        ratio = torch.exp(log_ratio)
        surr1 = ratio * combined_adv
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * combined_adv
        surrogate = torch.min(surr1, surr2)
        policy_loss = -surrogate.mean()
        policy_surrogate_abs = surrogate.abs().mean()
        approx_kl = ((ratio - 1.0) - log_ratio).mean()
        clip_fraction = ((ratio - 1.0).abs() > self.clip_eps).float().mean()

        tracked_values = torch.stack(
            [
                (
                    values_vec[..., self.reward_indices]
                    * self.reward_weights.to(values_vec.device)
                ).sum(dim=-1),
                *[values_vec[..., idx] for idx in self.cost_indices],
            ],
            dim=-1,
        )
        value_loss = torch.nn.functional.mse_loss(tracked_values, batch.returns)
        entropy_loss = entropy.mean()
        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "policy_surrogate_abs": policy_surrogate_abs,
            "value_loss": value_loss,
            "entropy": entropy_loss,
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
        }

    def update(self, rollouts: RolloutBatch) -> dict[str, float]:
        if rollouts.advantages is None or rollouts.returns is None:
            self.compute_advantages(rollouts)

        flat = rollouts.flatten()
        metrics_sum: dict[str, float] = {}
        update_steps = 0
        num_samples = flat.batch_size
        early_stop = False
        was_training = self.policy.training
        self.policy.eval()

        try:
            for _ in range(self.update_epochs):
                indices = torch.randperm(num_samples, device=flat.log_probs.device)
                for start in range(0, num_samples, self.minibatch_size):
                    batch_idx = indices[start : start + self.minibatch_size]
                    batch = RolloutBatch(
                        observations=index_tree(flat.observations, batch_idx),
                        actions=index_tree(flat.actions, batch_idx),
                        log_probs=flat.log_probs[batch_idx],
                        rewards_vec=flat.rewards_vec[batch_idx],
                        dones=flat.dones[batch_idx],
                        values_vec=flat.values_vec[batch_idx],
                        masks=index_tree(flat.masks, batch_idx),
                        preferences=None if flat.preferences is None else flat.preferences[batch_idx],
                        advantages=None if flat.advantages is None else flat.advantages[batch_idx],
                        returns=None if flat.returns is None else flat.returns[batch_idx],
                    )
                    losses = self.compute_loss(batch)
                    approx_kl = float(losses["approx_kl"].detach().cpu())
                    if approx_kl > self.target_kl:
                        early_stop = True
                        break
                    self.optimizer.zero_grad()
                    losses["total_loss"].backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    step_metrics = {key: float(value.detach().cpu()) for key, value in losses.items()}
                    for key, value in step_metrics.items():
                        metrics_sum[key] = metrics_sum.get(key, 0.0) + value
                    update_steps += 1
                if early_stop:
                    break
        finally:
            if was_training:
                self.policy.train()

        if update_steps == 0:
            return {"early_stop": float(early_stop)}
        averaged = {key: value / update_steps for key, value in metrics_sum.items()}
        averaged["early_stop"] = float(early_stop)
        return averaged

    def post_update(self, rollouts: RolloutBatch) -> dict[str, float]:
        if not self.cost_indices:
            return {}
        costs = rollouts.rewards_vec[..., self.cost_indices].mean(dim=tuple(range(rollouts.rewards_vec.ndim - 1)))
        costs = costs.to(self.lambdas.dtype)
        targets = self.cost_targets.to(costs.device)
        updates = []
        for idx in range(len(self.cost_indices)):
            violation = float(costs[idx] - targets[idx])
            delta = self.pid.step(violation) if self.use_pid else self.lambda_lr * violation
            self.lambdas[idx] = torch.clamp(self.lambdas[idx] + delta, min=0.0)
            updates.append(float(self.lambdas[idx].cpu()))
        return {f"lambda_{idx}": value for idx, value in enumerate(updates)}

    def _resolve_reward_structure(
        self,
    ) -> tuple[list[int], torch.Tensor, list[int], torch.Tensor]:
        n_objectives = int(getattr(self.policy, "n_objectives", 1))
        reward_indices_cfg = self.config.get("reward_indices", "auto")
        reward_weights_cfg = self.config.get("reward_weights", "auto")
        cost_indices_cfg = self.config.get("cost_indices", "auto")
        cost_targets_cfg = self.config.get("cost_targets", "auto")

        if isinstance(reward_indices_cfg, str) and reward_indices_cfg.lower() == "auto":
            if n_objectives == 3:
                reward_indices = [0, 1]
            else:
                reward_indices = [0]
        else:
            reward_indices = [int(idx) for idx in reward_indices_cfg]

        if isinstance(reward_weights_cfg, str) and reward_weights_cfg.lower() == "auto":
            if len(reward_indices) == 2 and reward_indices == [0, 1]:
                reward_weights = [1.0, 0.1]
            else:
                reward_weights = [1.0] * len(reward_indices)
        else:
            reward_weights = [float(weight) for weight in reward_weights_cfg]
        if len(reward_weights) != len(reward_indices):
            raise ValueError(
                "PPOLagrangian reward_weights must match reward_indices length."
            )

        if isinstance(cost_indices_cfg, str) and cost_indices_cfg.lower() == "auto":
            if n_objectives == 3:
                cost_indices = [2]
            elif n_objectives == 2:
                cost_indices = [1]
            else:
                cost_indices = [idx for idx in range(n_objectives) if idx not in reward_indices]
        else:
            cost_indices = [int(idx) for idx in cost_indices_cfg]

        if isinstance(cost_targets_cfg, str) and cost_targets_cfg.lower() == "auto":
            cost_targets = [0.0 for _ in cost_indices]
        else:
            cost_targets = [float(value) for value in cost_targets_cfg]
        if len(cost_targets) != len(cost_indices):
            raise ValueError(
                "PPOLagrangian cost_targets must match cost_indices length."
            )

        return (
            reward_indices,
            torch.tensor(reward_weights, dtype=torch.float32),
            cost_indices,
            torch.tensor(cost_targets, dtype=torch.float32),
        )
