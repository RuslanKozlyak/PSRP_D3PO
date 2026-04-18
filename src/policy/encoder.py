from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TokenBatchNorm(nn.Module):
    """BatchNorm over the embedding dimension for token sequences."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        transposed = x.transpose(1, 2)
        # BatchNorm needs more than one value per channel in training mode.
        # For tiny instances such as a single vehicle token, fall back to
        # the accumulated running statistics so evaluation utilities keep working.
        if self.training and (transposed.size(0) * transposed.size(2) <= 1):
            normalized = F.batch_norm(
                transposed,
                self.norm.running_mean,
                self.norm.running_var,
                self.norm.weight,
                self.norm.bias,
                training=False,
                momentum=0.0,
                eps=self.norm.eps,
            )
            return normalized.transpose(1, 2)
        return self.norm(transposed).transpose(1, 2)


class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = TokenBatchNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.norm2 = TokenBatchNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        return self.norm2(x + ffn_out)


class StationEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_heads: int,
        n_layers: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [AttentionBlock(hidden_dim, n_heads, ffn_dim, dropout) for _ in range(n_layers)]
        )

    def forward(self, station_features: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(station_features)
        for block in self.blocks:
            x = block(x)
        return x


class VehicleEncoder(nn.Module):
    def __init__(
        self,
        vehicle_input_dim: int,
        compartment_input_dim: int,
        hidden_dim: int,
        n_heads: int,
        n_layers: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.vehicle_projection = nn.Linear(vehicle_input_dim, hidden_dim)
        self.compartment_projection = nn.Linear(compartment_input_dim, hidden_dim)
        self.compartment_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.gate_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        self.blocks = nn.ModuleList(
            [AttentionBlock(hidden_dim, n_heads, ffn_dim, dropout) for _ in range(max(1, n_layers - 1))]
        )

    def forward(
        self,
        vehicle_features: torch.Tensor,
        compartment_features: torch.Tensor,
        global_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, n_vehicles, _, hidden_dim = (
            compartment_features.size(0),
            compartment_features.size(1),
            compartment_features.size(2),
            self.vehicle_projection.out_features,
        )
        vehicle_tokens = self.vehicle_projection(vehicle_features)
        compartment_tokens = self.compartment_projection(compartment_features)
        flat_query = vehicle_tokens.reshape(batch_size * n_vehicles, 1, hidden_dim)
        flat_compartments = compartment_tokens.reshape(batch_size * n_vehicles, -1, hidden_dim)
        attn_summary, _ = self.compartment_attention(
            flat_query,
            flat_compartments,
            flat_compartments,
            need_weights=False,
        )
        vehicle_tokens = vehicle_tokens + attn_summary.reshape(batch_size, n_vehicles, hidden_dim)

        if global_context is not None:
            expanded_global = global_context.unsqueeze(1).expand(batch_size, n_vehicles, hidden_dim)
            gate = torch.sigmoid(
                self.gate_projection(torch.cat([vehicle_tokens, expanded_global], dim=-1))
            )
            vehicle_tokens = gate * vehicle_tokens + (1.0 - gate) * expanded_global

        for block in self.blocks:
            vehicle_tokens = block(vehicle_tokens)

        return vehicle_tokens, compartment_tokens


class GlobalEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, global_features: torch.Tensor) -> torch.Tensor:
        return self.net(global_features)
