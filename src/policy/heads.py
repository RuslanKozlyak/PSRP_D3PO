from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Beta, Categorical


class PointerHead(nn.Module):
    def __init__(self, hidden_dim: int, tanh_clip: float = 10.0) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.tanh_clip = tanh_clip

    def forward(
        self,
        context: torch.Tensor,
        candidates: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        query = self.query(context).unsqueeze(1)
        key = self.key(candidates)
        logits = (query * key).sum(dim=-1) / math.sqrt(key.size(-1))
        logits = self.tanh_clip * torch.tanh(logits)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        return logits


class MaskedGlimpse(nn.Module):
    """AM-style masked attention glimpse used before pointer selection."""

    def __init__(self, hidden_dim: int, n_heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            batch_first=True,
        )

    def forward(
        self,
        context: torch.Tensor,
        candidates: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        attn_out, _ = self.attn(
            context.unsqueeze(1),
            candidates,
            candidates,
            key_padding_mask=~mask,
            need_weights=False,
        )
        return attn_out.squeeze(1)


class BetaHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.alpha = nn.Linear(hidden_dim, 1)
        self.beta = nn.Linear(hidden_dim, 1)

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.net(context)
        alpha = torch.nn.functional.softplus(self.alpha(hidden)) + 1.0
        beta = torch.nn.functional.softplus(self.beta(hidden)) + 1.0
        return alpha.squeeze(-1), beta.squeeze(-1)


class MultiCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: list[int], n_objectives: int) -> None:
        super().__init__()
        heads = []
        for _ in range(n_objectives):
            layers = []
            prev = input_dim
            for hidden in hidden_sizes:
                layers.extend([nn.Linear(prev, hidden), nn.ReLU()])
                prev = hidden
            layers.append(nn.Linear(prev, 1))
            heads.append(nn.Sequential(*layers))
        self.heads = nn.ModuleList(heads)

    def forward(self, pooled_embedding: torch.Tensor) -> torch.Tensor:
        values = [head(pooled_embedding) for head in self.heads]
        return torch.cat(values, dim=-1)


def categorical_from_logits(logits: torch.Tensor) -> Categorical:
    return Categorical(logits=logits)


def beta_from_params(alpha: torch.Tensor, beta: torch.Tensor) -> Beta:
    return Beta(alpha, beta)
