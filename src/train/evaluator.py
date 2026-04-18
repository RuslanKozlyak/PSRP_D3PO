from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.benchmarks import cpsat as cpsat_benchmarks


class Evaluator:
    def __init__(self, env: Any, policy: torch.nn.Module, device: torch.device) -> None:
        self.env = env
        self.policy = policy
        self.device = device

    def evaluate(
        self,
        episodes: int = 1,
        preferences: torch.Tensor | np.ndarray | None = None,
    ) -> dict[str, float]:
        preference_tensor = self._prepare_preferences(preferences)
        metrics = cpsat_benchmarks.evaluate_policy_kpis(
            self.env,
            self.policy,
            episodes=episodes,
            preferences=preference_tensor,
            deterministic=True,
            device=self.device,
            return_reward_components=True,
        )

        result: dict[str, float] = {}
        for key in (
            "total_travel_distance",
            "total_travel_time",
            "dry_runs",
            "average_stock_levels_percent",
            "average_vehicle_utilization",
            "average_stops_per_trip",
        ):
            if key in metrics:
                result[f"eval_{key}"] = float(metrics[key])

        reward_keys = sorted(key for key in metrics if key.startswith("reward_"))
        reward_total = 0.0
        for key in reward_keys:
            result[f"eval_{key}"] = float(metrics[key])
            reward_total += float(metrics[key])
        if reward_keys:
            result["eval_reward_total"] = reward_total
        return result

    def _prepare_preferences(
        self,
        preferences: torch.Tensor | np.ndarray | None,
    ) -> torch.Tensor | None:
        if preferences is None:
            return None
        tensor = torch.as_tensor(preferences, dtype=torch.float32, device=self.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor
