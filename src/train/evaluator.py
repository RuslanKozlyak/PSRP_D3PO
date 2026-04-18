from __future__ import annotations

from typing import Any

import numpy as np
import torch


class Evaluator:
    def __init__(self, env: Any, policy: torch.nn.Module, device: torch.device) -> None:
        self.env = env
        self.policy = policy
        self.device = device

    def evaluate(self, episodes: int = 1) -> dict[str, float]:
        total_reward = np.zeros(4, dtype=np.float64)
        total_distance = 0.0
        total_stockout = 0.0

        for _ in range(episodes):
            obs, info = self.env.reset()
            done = False
            truncated = False
            while not done and not truncated:
                obs_t = {k: torch.as_tensor(v, device=self.device).unsqueeze(0) for k, v in obs.items()}
                with torch.no_grad():
                    output = self.policy.act(obs_t, deterministic=True)
                action = {
                    "vehicle": int(output.actions["vehicle"].item()),
                    "node": int(output.actions["node"].item()),
                    "quantity": output.actions["quantity"].squeeze(0).cpu().numpy(),
                }
                obs, reward_vec, done, truncated, info = self.env.step(action)
                total_reward += reward_vec

            total_distance += info["cumulative_distance"]
            total_stockout += info["cumulative_stockout"]

        return {
            "eval_distance": total_distance / max(episodes, 1),
            "eval_stockout": total_stockout / max(episodes, 1),
            "eval_reward_distance": float(total_reward[0] / max(episodes, 1)),
            "eval_reward_holding": float(total_reward[1] / max(episodes, 1)),
            "eval_reward_safety": float(total_reward[2] / max(episodes, 1)),
            "eval_reward_time_window": float(total_reward[3] / max(episodes, 1)),
        }
