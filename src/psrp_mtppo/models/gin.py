from __future__ import annotations

import torch
from torch import nn


def make_mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend([nn.Linear(prev_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class GINLayer(nn.Module):
    """GIN layer, eq. (19) of Lu et al. 2025: ``(1 + eps) h_i + sum_{u in N(i)} h_u``.

    ``adjacency`` is expected to contain the raw (unnormalized) 0/1 connectivity so
    the matmul realises a true SUM aggregator. If a normalized adjacency is passed,
    the aggregator degenerates into a mean.
    """

    def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = make_mlp(input_dim, (output_dim,), output_dim, dropout)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        aggregated = torch.matmul(adjacency, features)
        return self.mlp((1.0 + self.eps) * features + aggregated)


class GINEncoder(nn.Module):
    def __init__(self, input_dim: int, dims: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in dims:
            layers.append(GINLayer(prev_dim, dim, dropout))
            prev_dim = dim
        self.layers = nn.ModuleList(layers)
        self.output_dim = prev_dim

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        hidden = features
        for layer in self.layers:
            hidden = layer(hidden, adjacency)
        return hidden
