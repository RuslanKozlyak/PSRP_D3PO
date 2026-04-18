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
        self.reward_index = int(self.config.get("reward_index", 0))
        self.cost_indices = list(self.config.get("cost_indices", [2, 3]))
        self.cost_targets = torch.tensor(
            self.config.get("cost_targets", [0.0 for _ in self.cost_indices]),
            dtype=torch.float32,
        )
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
        reward_adv = self.compute_gae(
            rewards=rollouts.rewards_vec[..., self.reward_index],
            values=rollouts.values_vec[..., self.reward_index],
            dones=rollouts.dones,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        cost_advantages = []
        reward_returns = reward_adv + rollouts.values_vec[..., self.reward_index]
        cost_returns = []
        for cost_idx in self.cost_indices:
            cost_adv = self.compute_gae(
                rewards=rollouts.rewards_vec[..., cost_idx],
                values=rollouts.values_vec[..., cost_idx],
                dones=rollouts.dones,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
            )
            cost_advantages.append(cost_adv)
            cost_returns.append(cost_adv + rollouts.values_vec[..., cost_idx])

        advantages = torch.stack([reward_adv, *cost_advantages], dim=-1)
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
        combined_adv = reward_adv - (cost_adv * self.lambdas.to(cost_adv.device)).sum(dim=-1)

        ratio = torch.exp(new_log_prob - batch.log_probs)
        surr1 = ratio * combined_adv
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * combined_adv
        policy_loss = -torch.min(surr1, surr2).mean()

        tracked_values = torch.stack(
            [values_vec[..., self.reward_index], *[values_vec[..., idx] for idx in self.cost_indices]],
            dim=-1,
        )
        value_loss = torch.nn.functional.mse_loss(tracked_values, batch.returns)
        entropy_loss = entropy.mean()
        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy_loss,
        }

    def update(self, rollouts: RolloutBatch) -> dict[str, float]:
        if rollouts.advantages is None or rollouts.returns is None:
            self.compute_advantages(rollouts)

        flat = rollouts.flatten()
        metrics: dict[str, float] = {}
        num_samples = flat.batch_size

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
                self.optimizer.zero_grad()
                losses["total_loss"].backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                metrics = {key: float(value.detach().cpu()) for key, value in losses.items()}

        return metrics

    def post_update(self, rollouts: RolloutBatch) -> dict[str, float]:
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
