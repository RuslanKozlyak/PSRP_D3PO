from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from typing import Any

import torch


@dataclass
class RolloutBatch:
    observations: dict[str, torch.Tensor]
    actions: dict[str, torch.Tensor]
    log_probs: torch.Tensor
    rewards_vec: torch.Tensor
    dones: torch.Tensor
    values_vec: torch.Tensor
    masks: dict[str, torch.Tensor]
    preferences: torch.Tensor | None = None
    advantages: torch.Tensor | None = None
    returns: torch.Tensor | None = None

    def flatten(self) -> "RolloutBatch":
        n_leading_dims = len(self.dones.shape)
        flat_obs = _flatten_tree(self.observations, n_leading_dims)
        flat_actions = _flatten_tree(self.actions, n_leading_dims)
        flat_masks = _flatten_tree(self.masks, n_leading_dims)

        return RolloutBatch(
            observations=flat_obs,
            actions=flat_actions,
            log_probs=_flatten_tensor(self.log_probs, n_leading_dims),
            rewards_vec=_flatten_tensor(self.rewards_vec, n_leading_dims),
            dones=_flatten_tensor(self.dones, n_leading_dims),
            values_vec=_flatten_tensor(self.values_vec, n_leading_dims),
            masks=flat_masks,
            preferences=_flatten_tensor(self.preferences, n_leading_dims),
            advantages=_flatten_tensor(self.advantages, n_leading_dims),
            returns=_flatten_tensor(self.returns, n_leading_dims),
        )

    @property
    def batch_size(self) -> int:
        return int(prod(self.dones.shape))


@dataclass
class RolloutBuffer:
    storage: list[dict[str, Any]] = field(default_factory=list)

    def add(
        self,
        *,
        observation: dict[str, torch.Tensor],
        action: dict[str, torch.Tensor],
        log_prob: torch.Tensor,
        reward_vec: torch.Tensor,
        done: torch.Tensor,
        value_vec: torch.Tensor,
        masks: dict[str, torch.Tensor],
        preference: torch.Tensor | None = None,
    ) -> None:
        self.storage.append(
            {
                "observation": {k: v.detach().clone() for k, v in observation.items()},
                "action": {k: v.detach().clone() for k, v in action.items()},
                "log_prob": log_prob.detach().clone(),
                "reward_vec": reward_vec.detach().clone(),
                "done": done.detach().clone(),
                "value_vec": value_vec.detach().clone(),
                "masks": {k: v.detach().clone() for k, v in masks.items()},
                "preference": None if preference is None else preference.detach().clone(),
            }
        )

    def as_batch(self) -> RolloutBatch:
        if not self.storage:
            raise ValueError("Cannot materialize an empty rollout buffer.")

        observations = _stack_tree([step["observation"] for step in self.storage])
        actions = _stack_tree([step["action"] for step in self.storage])
        masks = _stack_tree([step["masks"] for step in self.storage])
        preferences = None
        if any(step["preference"] is not None for step in self.storage):
            preferences = torch.stack(
                [
                    step["preference"]
                    if step["preference"] is not None
                    else torch.zeros_like(self.storage[0]["preference"])
                    for step in self.storage
                ],
                dim=0,
            )

        return RolloutBatch(
            observations=observations,
            actions=actions,
            log_probs=torch.stack([step["log_prob"] for step in self.storage], dim=0),
            rewards_vec=torch.stack([step["reward_vec"] for step in self.storage], dim=0),
            dones=torch.stack([step["done"] for step in self.storage], dim=0),
            values_vec=torch.stack([step["value_vec"] for step in self.storage], dim=0),
            masks=masks,
            preferences=preferences,
        )

    def clear(self) -> None:
        self.storage.clear()


def index_tree(tree: dict[str, torch.Tensor], index: torch.Tensor) -> dict[str, torch.Tensor]:
    return {key: value[index] for key, value in tree.items()}


def _stack_tree(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = items[0].keys()
    return {key: torch.stack([item[key] for item in items], dim=0) for key in keys}


def _flatten_tree(tree: dict[str, torch.Tensor], n_leading_dims: int) -> dict[str, torch.Tensor]:
    return {key: _flatten_tensor(value, n_leading_dims) for key, value in tree.items()}


def _flatten_tensor(tensor: torch.Tensor | None, n_leading_dims: int) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor.reshape(-1, *tensor.shape[n_leading_dims:])
