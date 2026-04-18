from __future__ import annotations

from typing import Any

import torch

from src.algo.base import BaseAlgorithm
from src.algo.rollout import RolloutBatch, index_tree


class PPO(BaseAlgorithm):
    def __init__(self, policy: torch.nn.Module, config: dict[str, Any]) -> None:
        super().__init__(policy, config)
        self.reward_weights_config = self.config.get("reward_weights", "auto")

    def compute_advantages(self, rollouts: RolloutBatch) -> torch.Tensor:
        weights = self._resolve_reward_weights(
            device=rollouts.rewards_vec.device,
            n_objectives=rollouts.rewards_vec.shape[-1],
        )
        rewards_scalar = (rollouts.rewards_vec * weights).sum(dim=-1)
        values_scalar = (rollouts.values_vec * weights).sum(dim=-1)
        bootstrap_value = None
        if rollouts.bootstrap_value is not None:
            bootstrap_value = (rollouts.bootstrap_value * weights).sum(dim=-1)
        raw_advantages = self.compute_gae(
            rewards=rewards_scalar,
            values=values_scalar,
            dones=rollouts.dones,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            bootstrap_value=bootstrap_value,
        )
        advantages = (raw_advantages - raw_advantages.mean()) / raw_advantages.std().clamp_min(1e-6)
        returns = raw_advantages + values_scalar
        rollouts.advantages = advantages
        rollouts.returns = returns
        return advantages

    def compute_loss(self, batch: RolloutBatch) -> dict[str, torch.Tensor]:
        weights = self._resolve_reward_weights(
            device=batch.values_vec.device,
            n_objectives=batch.values_vec.shape[-1],
        )
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

        log_ratio = new_log_prob - batch.log_probs
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        surrogate = torch.min(surr1, surr2)
        policy_loss = -surrogate.mean()
        policy_surrogate_abs = surrogate.abs().mean()
        approx_kl = ((ratio - 1.0) - log_ratio).mean()
        clip_fraction = ((ratio - 1.0).abs() > self.clip_eps).float().mean()
        value_loss = torch.nn.functional.mse_loss(values_scalar, batch.returns)
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

    def _resolve_reward_weights(
        self,
        *,
        device: torch.device,
        n_objectives: int,
    ) -> torch.Tensor:
        cfg = self.reward_weights_config
        if isinstance(cfg, str) and cfg.lower() == "auto":
            if n_objectives == 3:
                values = [1.0, 0.1, 10.0]
            elif n_objectives == 2:
                values = [1.0, 10.0]
            else:
                values = [1.0] * n_objectives
        else:
            values = list(cfg)
            if len(values) != n_objectives:
                raise ValueError(
                    f"PPO reward_weights has length {len(values)}, "
                    f"but the reward vector has {n_objectives} objectives."
                )
        return torch.tensor(values, dtype=torch.float32, device=device)

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
