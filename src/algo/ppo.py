from __future__ import annotations

from typing import Any

import torch

from src.algo.base import BaseAlgorithm
from src.algo.rollout import RolloutBatch, index_tree


class PPO(BaseAlgorithm):
    def __init__(self, policy: torch.nn.Module, config: dict[str, Any]) -> None:
        super().__init__(policy, config)
        self.reward_weights = torch.tensor(
            self.config.get("reward_weights", [1.0, 1.0, 1.0, 1.0]),
            dtype=torch.float32,
        )

    def compute_advantages(self, rollouts: RolloutBatch) -> torch.Tensor:
        weights = self.reward_weights.to(rollouts.rewards_vec.device)
        rewards_scalar = (rollouts.rewards_vec * weights).sum(dim=-1)
        values_scalar = (rollouts.values_vec * weights).sum(dim=-1)
        advantages = self.compute_gae(
            rewards=rewards_scalar,
            values=values_scalar,
            dones=rollouts.dones,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        returns = advantages + values_scalar
        rollouts.advantages = advantages
        rollouts.returns = returns
        return advantages

    def compute_loss(self, batch: RolloutBatch) -> dict[str, torch.Tensor]:
        weights = self.reward_weights.to(batch.values_vec.device)
        new_log_prob, entropy, values_vec = self.policy.evaluate_actions(
            batch.observations,
            batch.actions,
            preferences=batch.preferences,
        )
        values_scalar = (values_vec * weights).sum(dim=-1)
        ratio = torch.exp(new_log_prob - batch.log_probs)
        advantages = batch.advantages
        assert advantages is not None
        assert batch.returns is not None

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = torch.nn.functional.mse_loss(values_scalar, batch.returns)
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
