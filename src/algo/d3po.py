from __future__ import annotations

from typing import Any

import torch
from torch.distributions import kl_divergence

from src.algo.base import BaseAlgorithm
from src.algo.rollout import RolloutBatch, index_tree


class D3PO(BaseAlgorithm):
    def __init__(self, policy: torch.nn.Module, config: dict[str, Any]) -> None:
        super().__init__(policy, config)
        self.n_objectives = int(self.config.get("n_objectives", 4))
        self.lambda_div = float(self.config.get("lambda_div", 0.01))
        self.alpha = float(self.config.get("alpha", 1.0))
        self.preference_neighborhood_radius = float(
            self.config.get("preference_neighborhood_radius", 0.05)
        )

    def sample_preferences(self, batch_size: int, device: torch.device | None = None) -> torch.Tensor:
        concentration = torch.ones(self.n_objectives, device=device)
        return torch.distributions.Dirichlet(concentration).sample((batch_size,))

    def sample_nearby_preferences(self, preferences: torch.Tensor) -> torch.Tensor:
        noise = (
            (torch.rand_like(preferences) * 2.0) - 1.0
        ) * self.preference_neighborhood_radius
        perturbed = torch.clamp(preferences + noise, min=1e-6)
        return perturbed / perturbed.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def compute_advantages(self, rollouts: RolloutBatch) -> torch.Tensor:
        advantages = self.compute_gae(
            rewards=rollouts.rewards_vec,
            values=rollouts.values_vec,
            dones=rollouts.dones.unsqueeze(-1),
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        returns = advantages + rollouts.values_vec

        reduce_dims = tuple(range(advantages.ndim - 1))
        adv_mean = advantages.mean(dim=reduce_dims, keepdim=True)
        adv_std = advantages.std(dim=reduce_dims, keepdim=True).clamp_min(1e-6)
        advantages = (advantages - adv_mean) / adv_std

        rollouts.advantages = advantages
        rollouts.returns = returns
        return advantages

    def compute_loss(self, batch: RolloutBatch) -> dict[str, torch.Tensor]:
        if batch.preferences is None:
            raise ValueError("D3PO requires preference vectors in the rollout batch.")
        assert batch.advantages is not None
        assert batch.returns is not None

        new_log_prob, entropy, values_vec = self.policy.evaluate_actions(
            batch.observations,
            batch.actions,
            preferences=batch.preferences,
        )
        ratio = torch.exp(new_log_prob - batch.log_probs)

        per_objective_terms = []
        for objective_idx in range(self.n_objectives):
            adv_i = batch.advantages[..., objective_idx]
            surr1 = ratio * adv_i
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_i
            per_objective_terms.append(-torch.min(surr1, surr2))
        per_objective_loss = torch.stack(per_objective_terms, dim=-1)
        policy_loss = (batch.preferences * per_objective_loss).sum(dim=-1).mean()

        value_loss = torch.nn.functional.mse_loss(values_vec, batch.returns)

        nearby_preferences = self.sample_nearby_preferences(batch.preferences)
        dist_a = self.policy.distribution_components(batch.observations, batch.preferences)
        dist_b = self.policy.distribution_components(batch.observations, nearby_preferences)

        vehicle_kl = kl_divergence(dist_a["vehicle_dist"], dist_b["vehicle_dist"])
        node_kl = kl_divergence(dist_a["node_dist"], dist_b["node_dist"])
        total_kl = vehicle_kl + node_kl

        quantity_dist_a = dist_a["quantity_dist"]
        quantity_dist_b = dist_b["quantity_dist"]
        if quantity_dist_a is not None and quantity_dist_b is not None:
            quantity_mask = dist_a["quantity_mask"].float()
            quantity_kl = kl_divergence(quantity_dist_a, quantity_dist_b) * quantity_mask
            total_kl = total_kl + quantity_kl.sum(dim=(-2, -1))

        total_kl = torch.nan_to_num(total_kl, nan=0.0, posinf=1e3, neginf=0.0)
        preference_distance = torch.abs(batch.preferences - nearby_preferences).sum(dim=-1)
        preference_distance = torch.nan_to_num(preference_distance, nan=0.0, posinf=1e3, neginf=0.0)
        diversity_loss = ((total_kl - self.alpha * preference_distance).pow(2)).mean()
        diversity_loss = torch.nan_to_num(diversity_loss, nan=0.0, posinf=1e3, neginf=0.0)

        entropy_loss = entropy.mean()
        entropy_loss = torch.nan_to_num(entropy_loss, nan=0.0, posinf=0.0, neginf=0.0)
        policy_loss = torch.nan_to_num(policy_loss, nan=0.0, posinf=1e3, neginf=-1e3)
        value_loss = torch.nan_to_num(value_loss, nan=0.0, posinf=1e3, neginf=0.0)
        total_loss = (
            policy_loss
            + self.value_coef * value_loss
            + self.lambda_div * diversity_loss
            - self.entropy_coef * entropy_loss
        )
        total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=1e3, neginf=-1e3)
        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "diversity_loss": diversity_loss,
            "entropy": entropy_loss,
        }

    def update(self, rollouts: RolloutBatch) -> dict[str, float]:
        if rollouts.advantages is None or rollouts.returns is None:
            self.compute_advantages(rollouts)

        flat = rollouts.flatten()
        if flat.preferences is None:
            raise ValueError("D3PO update requires rollout preferences.")

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
                    preferences=flat.preferences[batch_idx],
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
