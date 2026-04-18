from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import optim

from src.algo.rollout import RolloutBatch


class BaseAlgorithm(ABC):
    """Common interface for PPO-family updates."""

    def __init__(self, policy: torch.nn.Module, config: dict[str, Any]) -> None:
        self.policy = policy
        self.config = dict(config)
        self.gamma = float(self.config.get("gamma", 0.99))
        self.gae_lambda = float(self.config.get("gae_lambda", 0.95))
        self.clip_eps = float(self.config.get("clip_eps", 0.2))
        self.value_coef = float(self.config.get("value_coef", 0.5))
        self.entropy_coef = float(self.config.get("entropy_coef", 0.01))
        self.max_grad_norm = float(self.config.get("max_grad_norm", 0.5))
        self.update_epochs = int(self.config.get("update_epochs", 4))
        self.minibatch_size = int(self.config.get("minibatch_size", 128))
        self.target_kl = float(self.config.get("target_kl", 0.05))
        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=float(self.config.get("learning_rate", 3e-4)),
        )

    @staticmethod
    def compute_gae(
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        gamma: float,
        gae_lambda: float,
        bootstrap_value: torch.Tensor | None = None,
    ) -> torch.Tensor:
        advantages = torch.zeros_like(rewards)
        last_advantage = torch.zeros_like(rewards[-1])
        if bootstrap_value is None:
            next_value = torch.zeros_like(values[-1])
        else:
            next_value = bootstrap_value.to(values.device, values.dtype)

        for step in reversed(range(rewards.size(0))):
            not_done = 1.0 - dones[step].float()
            delta = rewards[step] + gamma * next_value * not_done - values[step]
            last_advantage = delta + gamma * gae_lambda * not_done * last_advantage
            advantages[step] = last_advantage
            next_value = values[step]

        return advantages

    def post_update(self, rollouts: RolloutBatch) -> dict[str, float]:
        return {}

    @abstractmethod
    def compute_advantages(self, rollouts: RolloutBatch) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def compute_loss(self, batch: RolloutBatch) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def update(self, rollouts: RolloutBatch) -> dict[str, float]:
        raise NotImplementedError
