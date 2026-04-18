from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from src.policy.decoder import HierarchicalDecoder
from src.policy.encoder import GlobalEncoder, StationEncoder, VehicleEncoder
from src.policy.heads import MultiCritic, beta_from_params, categorical_from_logits


@dataclass
class PolicyOutput:
    actions: dict[str, torch.Tensor]
    log_prob: torch.Tensor
    entropy: torch.Tensor
    values_vec: torch.Tensor


class HierarchicalPolicy(nn.Module):
    """Hierarchical masked policy with vector critic heads."""

    def __init__(
        self,
        station_input_dim: int,
        vehicle_input_dim: int,
        compartment_input_dim: int,
        global_input_dim: int,
        n_products: int,
        config: dict[str, Any],
        n_objectives: int = 4,
    ) -> None:
        super().__init__()
        hidden_dim = int(config["hidden_dim"])
        n_heads = int(config["n_heads"])
        n_layers = int(config["n_layers"])
        ffn_dim = int(config["ffn_dim"])
        dropout = float(config["dropout"])
        tanh_clip = float(config.get("tanh_clip_C", 10.0))
        critic_hidden = list(config.get("critic_hidden", [hidden_dim, hidden_dim]))

        self.n_products = n_products
        self.n_objectives = n_objectives
        self.quantity_mode = str(config.get("quantity_mode", "deterministic"))

        self.station_encoder = StationEncoder(
            input_dim=station_input_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        self.vehicle_encoder = VehicleEncoder(
            vehicle_input_dim=vehicle_input_dim,
            compartment_input_dim=compartment_input_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        self.global_encoder = GlobalEncoder(global_input_dim, hidden_dim)
        self.preference_projection = nn.Linear(n_objectives, hidden_dim)
        self.decoder = HierarchicalDecoder(
            hidden_dim=hidden_dim,
            n_products=n_products,
            n_glimpse_heads=int(config.get("n_glimpse_heads", n_heads)),
            tanh_clip=tanh_clip,
        )
        self.critic = MultiCritic(hidden_dim * 3, critic_hidden, n_objectives)

    @classmethod
    def from_env(
        cls,
        env: Any,
        config: dict[str, Any],
        n_objectives: int = 4,
    ) -> "HierarchicalPolicy":
        obs_space = env.observation_space
        return cls(
            station_input_dim=obs_space["stations"].shape[-1],
            vehicle_input_dim=obs_space["vehicles"].shape[-1],
            compartment_input_dim=obs_space["compartments"].shape[-1],
            global_input_dim=obs_space["global"].shape[-1],
            n_products=env.instance.n_products,
            config=config,
            n_objectives=n_objectives,
        )

    def act(
        self,
        obs: dict[str, torch.Tensor],
        preferences: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> PolicyOutput:
        encoded = self.encode(obs, preferences=preferences)
        mask_vehicle = obs["mask_vehicle"].bool()
        mask_node = obs["mask_node"].bool()
        mask_quantity = obs["mask_quantity"].bool()
        quantity_upper = obs["quantity_upper_bound"]

        vehicle_logits = self.decoder.vehicle_logits(
            encoded["stations"],
            encoded["vehicles"],
            encoded["global"],
            mask_vehicle,
        )
        vehicle_dist = categorical_from_logits(vehicle_logits)
        vehicle_action = torch.argmax(vehicle_logits, dim=-1) if deterministic else vehicle_dist.sample()

        selected_node_mask = mask_node[torch.arange(mask_node.size(0), device=mask_node.device), vehicle_action]
        node_logits = self.decoder.node_logits(
            encoded["stations"],
            encoded["vehicles"],
            encoded["global"],
            vehicle_action,
            selected_node_mask,
        )
        node_dist = categorical_from_logits(node_logits)
        node_action = torch.argmax(node_logits, dim=-1) if deterministic else node_dist.sample()

        quantity_alpha, quantity_beta = self.decoder.quantity_params(
            encoded["stations"],
            encoded["vehicles"],
            encoded["compartments"],
            encoded["global"],
            mask_quantity,
            vehicle_action,
            node_action,
        )

        batch_index = torch.arange(quantity_upper.size(0), device=quantity_upper.device)
        selected_upper = quantity_upper[batch_index, vehicle_action, node_action]
        selected_mask_quantity = mask_quantity[batch_index, vehicle_action]

        if self.quantity_mode == "deterministic":
            quantity_action = selected_mask_quantity.float()
            quantity_log_prob = torch.zeros(quantity_action.size(0), device=quantity_action.device)
            quantity_entropy = torch.zeros_like(quantity_log_prob)
        else:
            quantity_dist = beta_from_params(quantity_alpha, quantity_beta)
            quantity_action = quantity_dist.mean if deterministic else quantity_dist.rsample()
            quantity_action = quantity_action * selected_mask_quantity.float()
            quantity_log_prob = (
                quantity_dist.log_prob(torch.clamp(quantity_action, 1e-6, 1.0 - 1e-6))
                * selected_mask_quantity.float()
            ).sum(dim=(-2, -1))
            quantity_entropy = (
                quantity_dist.entropy() * selected_mask_quantity.float()
            ).sum(dim=(-2, -1))

        values_vec = self._critic_values(encoded)
        valid_vehicle = mask_vehicle.sum(dim=-1).clamp_min(2).float()
        valid_nodes = selected_node_mask.sum(dim=-1).clamp_min(2).float()
        entropy = (
            vehicle_dist.entropy() / torch.log(valid_vehicle)
            + node_dist.entropy() / torch.log(valid_nodes)
            + quantity_entropy
        )
        log_prob = vehicle_dist.log_prob(vehicle_action) + node_dist.log_prob(node_action) + quantity_log_prob

        return PolicyOutput(
            actions={
                "vehicle": vehicle_action,
                "node": node_action,
                "quantity": quantity_action,
                "quantity_upper_bound": selected_upper,
            },
            log_prob=log_prob,
            entropy=entropy,
            values_vec=values_vec,
        )

    def evaluate_actions(
        self,
        obs: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor],
        preferences: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.encode(obs, preferences=preferences)
        mask_vehicle = obs["mask_vehicle"].bool()
        mask_node = obs["mask_node"].bool()
        mask_quantity = obs["mask_quantity"].bool()

        vehicle_logits = self.decoder.vehicle_logits(
            encoded["stations"],
            encoded["vehicles"],
            encoded["global"],
            mask_vehicle,
        )
        vehicle_dist = categorical_from_logits(vehicle_logits)
        vehicle_action = actions["vehicle"].long()

        selected_node_mask = mask_node[torch.arange(mask_node.size(0), device=mask_node.device), vehicle_action]
        node_logits = self.decoder.node_logits(
            encoded["stations"],
            encoded["vehicles"],
            encoded["global"],
            vehicle_action,
            selected_node_mask,
        )
        node_dist = categorical_from_logits(node_logits)
        node_action = actions["node"].long()

        quantity_alpha, quantity_beta = self.decoder.quantity_params(
            encoded["stations"],
            encoded["vehicles"],
            encoded["compartments"],
            encoded["global"],
            mask_quantity,
            vehicle_action,
            node_action,
        )
        selected_mask_quantity = mask_quantity[
            torch.arange(mask_quantity.size(0), device=mask_quantity.device),
            vehicle_action,
        ]

        if self.quantity_mode == "deterministic":
            quantity_log_prob = torch.zeros(vehicle_action.size(0), device=vehicle_action.device)
            quantity_entropy = torch.zeros_like(quantity_log_prob)
        else:
            quantity_dist = beta_from_params(quantity_alpha, quantity_beta)
            quantity_action = torch.clamp(actions["quantity"], 1e-6, 1.0 - 1e-6)
            quantity_log_prob = (
                quantity_dist.log_prob(quantity_action) * selected_mask_quantity.float()
            ).sum(dim=(-2, -1))
            quantity_entropy = (
                quantity_dist.entropy() * selected_mask_quantity.float()
            ).sum(dim=(-2, -1))

        valid_vehicle = mask_vehicle.sum(dim=-1).clamp_min(2).float()
        valid_nodes = selected_node_mask.sum(dim=-1).clamp_min(2).float()
        entropy = (
            vehicle_dist.entropy() / torch.log(valid_vehicle)
            + node_dist.entropy() / torch.log(valid_nodes)
            + quantity_entropy
        )
        log_prob = vehicle_dist.log_prob(vehicle_action) + node_dist.log_prob(node_action) + quantity_log_prob
        values_vec = self._critic_values(encoded)
        return log_prob, entropy, values_vec

    def preference_policy_signature(
        self,
        obs: dict[str, torch.Tensor],
        preferences: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.encode(obs, preferences=preferences)
        return self.decoder.vehicle_logits(
            encoded["stations"],
            encoded["vehicles"],
            encoded["global"],
            obs["mask_vehicle"].bool(),
        )

    def encode(
        self,
        obs: dict[str, torch.Tensor],
        preferences: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        station_embeddings = self.station_encoder(obs["stations"])
        global_embedding = self.global_encoder(obs["global"]) + station_embeddings.mean(dim=1)
        if preferences is not None:
            global_embedding = global_embedding + self.preference_projection(preferences)
        vehicle_embeddings, compartment_embeddings = self.vehicle_encoder(
            obs["vehicles"],
            obs["compartments"],
            global_context=global_embedding,
        )
        return {
            "stations": station_embeddings,
            "vehicles": vehicle_embeddings,
            "compartments": compartment_embeddings,
            "global": global_embedding,
        }

    def distribution_components(
        self,
        obs: dict[str, torch.Tensor],
        preferences: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | object]:
        encoded = self.encode(obs, preferences=preferences)
        mask_vehicle = obs["mask_vehicle"].bool()
        mask_node = obs["mask_node"].bool()
        mask_quantity = obs["mask_quantity"].bool()

        vehicle_logits = self.decoder.vehicle_logits(
            encoded["stations"],
            encoded["vehicles"],
            encoded["global"],
            mask_vehicle,
        )
        vehicle_dist = categorical_from_logits(vehicle_logits)
        vehicle_action = torch.argmax(vehicle_logits, dim=-1)

        batch_index = torch.arange(mask_node.size(0), device=mask_node.device)
        selected_node_mask = mask_node[batch_index, vehicle_action]
        node_logits = self.decoder.node_logits(
            encoded["stations"],
            encoded["vehicles"],
            encoded["global"],
            vehicle_action,
            selected_node_mask,
        )
        node_dist = categorical_from_logits(node_logits)
        node_action = torch.argmax(node_logits, dim=-1)

        selected_mask_quantity = mask_quantity[batch_index, vehicle_action]
        quantity_dist = None
        if self.quantity_mode != "deterministic":
            quantity_alpha, quantity_beta = self.decoder.quantity_params(
                encoded["stations"],
                encoded["vehicles"],
                encoded["compartments"],
                encoded["global"],
                mask_quantity,
                vehicle_action,
                node_action,
            )
            quantity_dist = beta_from_params(quantity_alpha, quantity_beta)

        return {
            "vehicle_logits": vehicle_logits,
            "vehicle_dist": vehicle_dist,
            "vehicle_action": vehicle_action,
            "node_logits": node_logits,
            "node_dist": node_dist,
            "node_action": node_action,
            "quantity_dist": quantity_dist,
            "quantity_mask": selected_mask_quantity,
        }

    def _critic_values(self, encoded: dict[str, torch.Tensor]) -> torch.Tensor:
        pooled = torch.cat(
            [
                encoded["stations"].mean(dim=1),
                encoded["vehicles"].mean(dim=1),
                encoded["global"],
            ],
            dim=-1,
        )
        return self.critic(pooled)
