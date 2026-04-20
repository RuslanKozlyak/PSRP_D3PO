from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..config import EnvironmentConfig, ModelConfig
from .actors import InventoryActor, RoutingActor
from .critic import JointCritic


@dataclass(slots=True)
class PolicyOutput:
    replenishment: torch.Tensor
    route: list[int]
    inventory_log_prob: torch.Tensor
    routing_log_prob: torch.Tensor
    value: torch.Tensor


class MTPPOModel(nn.Module):
    def __init__(self, env_config: EnvironmentConfig, model_config: ModelConfig) -> None:
        super().__init__()
        self.inventory_actor = InventoryActor(env_config, model_config)
        self.routing_actor = RoutingActor(env_config, model_config)
        self.critic = JointCritic(env_config, model_config)

    def act(self, obs: dict[str, torch.Tensor], greedy: bool = False) -> PolicyOutput:
        inventory_sample = self.inventory_actor.sample(obs, greedy=greedy)
        routing_sample = self.routing_actor.sample_route(obs, inventory_sample.action, greedy=greedy)
        value = self.critic(obs)
        return PolicyOutput(
            replenishment=inventory_sample.action,
            route=routing_sample.route,
            inventory_log_prob=inventory_sample.log_prob,
            routing_log_prob=routing_sample.log_prob,
            value=value,
        )
