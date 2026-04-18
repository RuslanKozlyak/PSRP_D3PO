from __future__ import annotations

from typing import Any

import torch

from src.algo.rollout import RolloutBatch, RolloutBuffer
from src.train.evaluator import Evaluator
from src.utils.logging import build_logger


class Trainer:
    def __init__(
        self,
        env: Any,
        policy: torch.nn.Module,
        algo: Any,
        config: dict[str, Any],
    ) -> None:
        self.env = env
        self.policy = policy
        self.algo = algo
        self.config = dict(config)
        self.device = torch.device(self.config.get("device", "cpu"))
        self.rollout_length = int(self.config.get("rollout_length", 64))
        self.total_iterations = int(self.config.get("total_iterations", 10))
        self.eval_interval = int(self.config.get("eval_interval", 5))
        self.eval_episodes = int(self.config.get("eval_episodes", 1))
        self.logger = build_logger()
        self.policy.to(self.device)
        self.evaluator = Evaluator(self.env, self.policy, self.device)
        self.eval_preferences = self._resolve_eval_preferences()

    def train(self) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []
        for iteration in range(self.total_iterations):
            rollouts = self.collect_rollouts()
            self.algo.compute_advantages(rollouts)
            metrics = self.algo.update(rollouts)
            metrics.update(self.algo.post_update(rollouts))
            metrics["iteration"] = float(iteration)

            if iteration % self.eval_interval == 0:
                metrics.update(self.evaluate())

            history.append(metrics)
            self.logger.info("iteration=%s metrics=%s", iteration, metrics)
        return history

    def collect_rollouts(self) -> RolloutBatch:
        obs, _ = self.env.reset()
        buffer = RolloutBuffer()
        episode_preference = self._sample_preferences()

        for _ in range(self.rollout_length):
            obs_t = self._to_tensor_dict(obs)
            with torch.no_grad():
                output = self.policy.act(obs_t, preferences=episode_preference, deterministic=False)

            action_np = {
                "vehicle": int(output.actions["vehicle"].item()),
                "node": int(output.actions["node"].item()),
                "quantity": output.actions["quantity"].squeeze(0).cpu().numpy(),
            }
            next_obs, reward_vec, terminated, truncated, _ = self.env.step(action_np)

            buffer.add(
                observation=obs_t,
                action={
                    "vehicle": output.actions["vehicle"],
                    "node": output.actions["node"],
                    "quantity": output.actions["quantity"],
                },
                log_prob=output.log_prob,
                reward_vec=torch.as_tensor(reward_vec, device=self.device).unsqueeze(0),
                done=torch.as_tensor([terminated or truncated], device=self.device),
                value_vec=output.values_vec,
                masks={
                    "mask_vehicle": obs_t["mask_vehicle"],
                    "mask_node": obs_t["mask_node"],
                    "mask_quantity": obs_t["mask_quantity"],
                },
                preference=episode_preference,
            )

            if terminated or truncated:
                obs, _ = self.env.reset()
                episode_preference = self._sample_preferences()
            else:
                obs = next_obs

        return buffer.as_batch()

    def evaluate(self) -> dict[str, float]:
        return self.evaluator.evaluate(
            episodes=self.eval_episodes,
            preferences=self.eval_preferences,
        )

    def _to_tensor_dict(self, obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        tensor_obs: dict[str, torch.Tensor] = {}
        for key, value in obs.items():
            tensor = torch.as_tensor(value, device=self.device)
            if tensor.ndim == 0:
                tensor = tensor.unsqueeze(0)
            tensor_obs[key] = tensor.unsqueeze(0)
        return tensor_obs

    def _sample_preferences(self) -> torch.Tensor | None:
        if hasattr(self.algo, "sample_preferences"):
            return self.algo.sample_preferences(1, device=self.device)
        return None

    def _resolve_eval_preferences(self) -> torch.Tensor | None:
        if not hasattr(self.algo, "sample_preferences"):
            return None
        preference = self.config.get("d3po_eval_preference")
        if preference is None:
            return None
        tensor = torch.as_tensor(preference, dtype=torch.float32, device=self.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor
