from __future__ import annotations

import torch
from torch import nn

from ..config import EnvironmentConfig, ModelConfig
from ..utils import complete_adjacency
from .attention import TransformerBlock
from .gin import GINEncoder, make_mlp


class JointCritic(nn.Module):
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
        self.value_head = make_mlp(
            model_config.mlp_dims[0] + 4,
            model_config.mlp_dims,
            1,
            model_config.dropout,
        )

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        features = obs["inventory_node_features"]
        state = obs["inventory_state"]
        batch_size, num_retailers = features.shape[:2]
        adjacency = complete_adjacency(num_retailers).to(features.device).expand(batch_size, -1, -1)
        graph_hidden = self.node_encoder(features, adjacency)
        state_hidden = self.state_encoder(state)
        hidden = self.project(torch.cat([graph_hidden, state_hidden], dim=-1))
        for layer in self.attention_layers:
            hidden = layer(hidden)
        pooled = hidden.mean(dim=1)
        extras = torch.cat(
            [
                obs["supplier_inventory"] / max(self.env_config.supplier_initial_inventory, 1.0),
                obs["day_fraction"],
                obs["remaining_capacity"].mean(dim=1, keepdim=True) / max(self.env_config.retailer_capacity, 1.0),
                obs["inventory_state"][:, :, 0].mean(dim=1, keepdim=True) / max(self.env_config.retailer_capacity, 1.0),
            ],
            dim=-1,
        )
        return self.value_head(torch.cat([pooled, extras], dim=-1)).squeeze(-1)
