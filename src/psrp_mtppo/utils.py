from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def complete_adjacency(num_nodes: int, normalize: bool = False) -> torch.Tensor:
    """Adjacency of the complete graph on ``num_nodes`` nodes (no self-loops).

    GIN in the paper (eq. 19) uses *sum* aggregation over neighbours, so by
    default we return the raw 0/1 adjacency. ``normalize=True`` divides by the
    neighbour count for a mean aggregator (kept for experimentation).
    """
    adjacency = torch.ones(num_nodes, num_nodes, dtype=torch.float32)
    adjacency.fill_diagonal_(0.0)
    if normalize:
        adjacency /= max(num_nodes - 1, 1)
    return adjacency


def numpy_observation_to_tensors(obs: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            tensors[key] = torch.as_tensor(value, dtype=torch.float32, device=device)
        elif isinstance(value, (float, int)):
            tensors[key] = torch.as_tensor([value], dtype=torch.float32, device=device)
    return tensors


def stack_observations(observations: list[dict[str, Any]], device: torch.device) -> dict[str, torch.Tensor]:
    stacked: dict[str, list[np.ndarray]] = {}
    for obs in observations:
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                stacked.setdefault(key, []).append(value)
            elif isinstance(value, (float, int)):
                stacked.setdefault(key, []).append(np.asarray([value], dtype=np.float32))

    return {
        key: torch.as_tensor(np.stack(value), dtype=torch.float32, device=device)
        for key, value in stacked.items()
    }


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    masked = tensor * mask
    denom = mask.sum(dim=dim).clamp_min(1.0)
    return masked.sum(dim=dim) / denom
