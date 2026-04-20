from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Normal, TransformedDistribution
from torch.distributions.transforms import AffineTransform, SigmoidTransform

from ..config import EnvironmentConfig, ModelConfig
from ..utils import complete_adjacency
from .attention import TransformerBlock
from .gin import GINEncoder, make_mlp


@dataclass(slots=True)
class InventorySample:
    action: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor


@dataclass(slots=True)
class RoutingSample:
    route: list[int]
    log_prob: torch.Tensor
    entropy: torch.Tensor


class InventoryActor(nn.Module):
    def __init__(self, env_config: EnvironmentConfig, model_config: ModelConfig) -> None:
        super().__init__()
        self.env_config = env_config
        self.node_encoder = GINEncoder(5, model_config.gin_dims, model_config.dropout)
        self.state_encoder = make_mlp(
            1 + 2 * env_config.history_length,
            (model_config.state_embed_dim,),
            model_config.state_embed_dim,
            model_config.dropout,
        )
        hidden_dim = self.node_encoder.output_dim + model_config.state_embed_dim
        self.project = nn.Linear(hidden_dim, model_config.mlp_dims[0])
        self.attention_layers = nn.ModuleList(
            TransformerBlock(model_config.mlp_dims[0], model_config.attention_heads, model_config.dropout)
            for _ in range(model_config.attention_layers)
        )
        self.decoder = make_mlp(
            model_config.mlp_dims[0],
            model_config.mlp_dims[1:],
            model_config.mlp_dims[-1],
            model_config.dropout,
        )
        self.mean_head = nn.Linear(model_config.mlp_dims[-1], 1)
        self.log_std_head = nn.Linear(model_config.mlp_dims[-1], 1)

    def _distribution(
        self,
        obs: dict[str, torch.Tensor],
    ) -> tuple[TransformedDistribution, torch.Tensor, torch.Tensor]:
        features = obs["inventory_node_features"]
        state = obs["inventory_state"]
        batch_size, num_retailers = features.shape[:2]
        adjacency = complete_adjacency(num_retailers).to(features.device).expand(batch_size, -1, -1)
        graph_hidden = self.node_encoder(features, adjacency)
        state_hidden = self.state_encoder(state)
        hidden = torch.cat([graph_hidden, state_hidden], dim=-1)
        hidden = self.project(hidden)
        for layer in self.attention_layers:
            hidden = layer(hidden)
        hidden = self.decoder(hidden)

        mean = self.mean_head(hidden).squeeze(-1)
        log_std = self.log_std_head(hidden).squeeze(-1).clamp(-4.0, 2.0)
        std = log_std.exp()
        max_replenishment = torch.minimum(
            obs["remaining_capacity"],
            torch.full_like(obs["remaining_capacity"], self.env_config.vehicle_capacity),
        ).clamp_min(1e-6)
        distribution = TransformedDistribution(
            Normal(mean, std),
            [SigmoidTransform(cache_size=1), AffineTransform(loc=0.0, scale=max_replenishment)],
        )
        return distribution, mean, std

    def sample(self, obs: dict[str, torch.Tensor], greedy: bool = False) -> InventorySample:
        distribution, mean, std = self._distribution(obs)
        if greedy:
            latent_mean = torch.sigmoid(mean)
            max_replenishment = torch.minimum(
                obs["remaining_capacity"],
                torch.full_like(obs["remaining_capacity"], self.env_config.vehicle_capacity),
            )
            action = latent_mean * max_replenishment
        else:
            action = distribution.rsample()
        action = action.clamp_min(0.0)
        log_prob = distribution.log_prob(action).sum(dim=-1)
        entropy = std.log().sum(dim=-1)
        return InventorySample(action=action, log_prob=log_prob, entropy=entropy, mean=mean, std=std)

    def evaluate_actions(
        self,
        obs: dict[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distribution, _mean, std = self._distribution(obs)
        log_prob = distribution.log_prob(actions).sum(dim=-1)
        entropy = std.log().sum(dim=-1)
        return log_prob, entropy


class RoutingActor(nn.Module):
    def __init__(self, env_config: EnvironmentConfig, model_config: ModelConfig) -> None:
        super().__init__()
        self.env_config = env_config
        self.node_encoder = GINEncoder(3, model_config.gin_dims, model_config.dropout)
        self.route_project = nn.Linear(self.node_encoder.output_dim, model_config.mlp_dims[0])
        self.attention_layers = nn.ModuleList(
            TransformerBlock(model_config.mlp_dims[0], model_config.attention_heads, model_config.dropout)
            for _ in range(model_config.attention_layers)
        )
        scorer_input_dim = model_config.mlp_dims[0] * 2 + 4
        self.node_scorer = make_mlp(
            scorer_input_dim,
            model_config.mlp_dims,
            1,
            model_config.dropout,
        )
        self.depot_scorer = make_mlp(
            model_config.mlp_dims[0] + 4,
            model_config.mlp_dims,
            1,
            model_config.dropout,
        )

    def _encode_nodes(self, routing_features: torch.Tensor) -> torch.Tensor:
        batch_size, num_retailers = routing_features.shape[:2]
        adjacency = complete_adjacency(num_retailers).to(routing_features.device).expand(batch_size, -1, -1)
        hidden = self.node_encoder(routing_features, adjacency)
        hidden = self.route_project(hidden)
        for layer in self.attention_layers:
            hidden = layer(hidden)
        return hidden

    def _score_candidates(
        self,
        node_embeddings: torch.Tensor,
        coords: torch.Tensor,
        current_coord: torch.Tensor,
        remaining: torch.Tensor,
        visited: torch.Tensor,
        current_load: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_retailers, hidden_dim = node_embeddings.shape
        global_context = node_embeddings.mean(dim=1, keepdim=True).expand(-1, num_retailers, -1)
        retailer_coords = current_coord[:, None, :].expand(-1, num_retailers, -1)
        remaining_ratio = (remaining / self.env_config.vehicle_capacity).unsqueeze(-1)
        load_ratio = (current_load / self.env_config.vehicle_capacity).view(batch_size, 1, 1).expand(-1, num_retailers, -1)
        current_distance = torch.norm(coords - retailer_coords, dim=-1, keepdim=True)
        node_input = torch.cat(
            [
                node_embeddings,
                global_context,
                remaining_ratio,
                load_ratio,
                current_distance,
                visited.unsqueeze(-1),
            ],
            dim=-1,
        )
        retailer_logits = self.node_scorer(node_input).squeeze(-1)
        depot_input = torch.cat(
            [
                node_embeddings.mean(dim=1),
                (current_load / self.env_config.vehicle_capacity).unsqueeze(-1),
                remaining.sum(dim=1, keepdim=True) / max(self.env_config.vehicle_capacity, 1.0),
                (visited.float().mean(dim=1, keepdim=True)),
                torch.ones(batch_size, 1, device=node_embeddings.device),
            ],
            dim=-1,
        )
        depot_logits = self.depot_scorer(depot_input)
        return torch.cat([depot_logits, retailer_logits], dim=1)

    def _make_logits(
        self,
        node_embeddings: torch.Tensor,
        coords: torch.Tensor,
        depot_coord: torch.Tensor,
        remaining: torch.Tensor,
        visited: torch.Tensor,
        current_node: int,
        current_load: float,
    ) -> torch.Tensor:
        batch_size = node_embeddings.shape[0]
        current_coord = depot_coord if current_node == 0 else coords[:, current_node - 1, :]
        logits = self._score_candidates(
            node_embeddings=node_embeddings,
            coords=coords,
            current_coord=current_coord,
            remaining=remaining,
            visited=visited.float(),
            current_load=torch.full((batch_size,), current_load, device=node_embeddings.device),
        )

        valid_customers = (~visited.bool()) & (remaining > 1e-6)
        feasible_customers = valid_customers & (remaining <= current_load + 1e-6)
        allow_depot = torch.ones((batch_size, 1), dtype=torch.bool, device=node_embeddings.device)
        allow_depot[:] = current_node != 0
        if feasible_customers.sum() == 0 and valid_customers.sum() > 0:
            allow_depot[:] = True
        if valid_customers.sum() == 0:
            allow_depot[:] = False

        mask = torch.cat([allow_depot, feasible_customers], dim=1)
        masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min / 4.0)
        return masked_logits

    def sample_route(
        self,
        obs: dict[str, torch.Tensor],
        replenishment: torch.Tensor,
        greedy: bool = False,
    ) -> RoutingSample:
        coords = obs["retailer_coords"]
        depot_coord = obs["depot_coord"]
        routing_features = torch.cat([coords, replenishment.unsqueeze(-1)], dim=-1)
        node_embeddings = self._encode_nodes(routing_features)
        remaining = replenishment.detach().clone()
        visited = torch.zeros_like(remaining, dtype=torch.bool)

        route = [0]
        current_node = 0
        current_load = float(self.env_config.vehicle_capacity)
        total_log_prob = torch.zeros(replenishment.shape[0], device=replenishment.device)
        total_entropy = torch.zeros_like(total_log_prob)

        max_steps = self.env_config.max_route_length
        for _ in range(max_steps):
            if bool((remaining <= 1e-6).all()):
                break

            logits = self._make_logits(
                node_embeddings=node_embeddings,
                coords=coords,
                depot_coord=depot_coord,
                remaining=remaining,
                visited=visited,
                current_node=current_node,
                current_load=current_load,
            )
            distribution = Categorical(logits=logits)
            if greedy:
                action = torch.argmax(logits, dim=-1)
            else:
                action = distribution.sample()
            total_log_prob = total_log_prob + distribution.log_prob(action)
            total_entropy = total_entropy + distribution.entropy()
            chosen = int(action.item())
            route.append(chosen)

            if chosen == 0:
                current_node = 0
                current_load = float(self.env_config.vehicle_capacity)
                continue

            visited[:, chosen - 1] = True
            current_load -= float(remaining[:, chosen - 1].item())
            remaining[:, chosen - 1] = 0.0
            current_node = chosen

        if route[-1] != 0:
            route.append(0)

        return RoutingSample(route=route, log_prob=total_log_prob, entropy=total_entropy)

    def evaluate_sequence(
        self,
        obs: dict[str, torch.Tensor],
        replenishment: torch.Tensor,
        route: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coords = obs["retailer_coords"]
        depot_coord = obs["depot_coord"]
        routing_features = torch.cat([coords, replenishment.unsqueeze(-1)], dim=-1)
        node_embeddings = self._encode_nodes(routing_features)
        remaining = replenishment.detach().clone()
        visited = torch.zeros_like(remaining, dtype=torch.bool)
        current_node = 0
        current_load = float(self.env_config.vehicle_capacity)
        total_log_prob = torch.zeros(replenishment.shape[0], device=replenishment.device)
        total_entropy = torch.zeros_like(total_log_prob)

        for chosen in route[1:]:
            if bool((remaining <= 1e-6).all()):
                break
            logits = self._make_logits(
                node_embeddings=node_embeddings,
                coords=coords,
                depot_coord=depot_coord,
                remaining=remaining,
                visited=visited,
                current_node=current_node,
                current_load=current_load,
            )
            distribution = Categorical(logits=logits)
            action = torch.as_tensor([chosen], device=replenishment.device)
            total_log_prob = total_log_prob + distribution.log_prob(action)
            total_entropy = total_entropy + distribution.entropy()

            if chosen == 0:
                current_node = 0
                current_load = float(self.env_config.vehicle_capacity)
                continue

            if remaining[:, chosen - 1].item() <= current_load + 1e-6:
                visited[:, chosen - 1] = True
                current_load -= float(remaining[:, chosen - 1].item())
                remaining[:, chosen - 1] = 0.0
                current_node = chosen
            else:
                current_node = 0
                current_load = float(self.env_config.vehicle_capacity)

        return total_log_prob, total_entropy
