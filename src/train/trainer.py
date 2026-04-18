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
        self.reward_component_names = tuple(getattr(self.env.reward_fn, "components", ()))
        self.policy.to(self.device)
        self.evaluator = Evaluator(self.env, self.policy, self.device)
        self.eval_preferences = self._resolve_eval_preferences()

    def train(self) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []
        for iteration in range(self.total_iterations):
            rollouts = self.collect_rollouts()
            train_reward_metrics = self._summarize_train_rewards(rollouts)
            self.algo.compute_advantages(rollouts)
            metrics = self.algo.update(rollouts)
            metrics.update(self.algo.post_update(rollouts))
            metrics.update(train_reward_metrics)
            metrics["iteration"] = float(iteration)

            if iteration % self.eval_interval == 0:
                metrics.update(self.evaluate())

            history.append(metrics)
            self.logger.info(self._format_progress_message(iteration, metrics))
        return history

    def collect_rollouts(self) -> RolloutBatch:
        obs, _ = self.env.reset()
        buffer = RolloutBuffer()
        episode_preference = self._sample_preferences()
        was_training = self.policy.training
        self.policy.eval()
        last_done = True

        try:
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

                done = terminated or truncated
                buffer.add(
                    observation=obs_t,
                    action={
                        "vehicle": output.actions["vehicle"],
                        "node": output.actions["node"],
                        "quantity": output.actions["quantity"],
                    },
                    log_prob=output.log_prob,
                    reward_vec=torch.as_tensor(reward_vec, device=self.device).unsqueeze(0),
                    done=torch.as_tensor([done], device=self.device),
                    value_vec=output.values_vec,
                    masks={
                        "mask_vehicle": obs_t["mask_vehicle"],
                        "mask_node": obs_t["mask_node"],
                        "mask_quantity": obs_t["mask_quantity"],
                    },
                    preference=episode_preference,
                )
                last_done = done

                if done:
                    obs, _ = self.env.reset()
                    episode_preference = self._sample_preferences()
                else:
                    obs = next_obs
            batch = buffer.as_batch()
            if last_done:
                batch.bootstrap_value = torch.zeros_like(batch.values_vec[-1])
            else:
                next_obs_t = self._to_tensor_dict(obs)
                with torch.no_grad():
                    next_output = self.policy.act(
                        next_obs_t,
                        preferences=episode_preference,
                        deterministic=True,
                    )
                batch.bootstrap_value = next_output.values_vec.detach().clone()
            return batch
        finally:
            if was_training:
                self.policy.train()

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
        if isinstance(preference, str) and preference.lower() == "auto":
            preference = self._default_d3po_preference()
        tensor = torch.as_tensor(preference, dtype=torch.float32, device=self.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _summarize_train_rewards(self, rollouts: RolloutBatch) -> dict[str, float]:
        rewards_vec = rollouts.rewards_vec.detach()
        flat_rewards = rewards_vec.reshape(-1, rewards_vec.shape[-1]).float()
        mean_rewards = flat_rewards.mean(dim=0)

        metrics = {
            f"train_reward_{name}": float(mean_rewards[idx].cpu())
            for idx, name in enumerate(self.reward_component_names)
        }
        metrics["train_reward_total"] = float(mean_rewards.sum().cpu())
        return metrics

    def _format_progress_message(self, iteration: int, metrics: dict[str, float]) -> str:
        message = f"iter={iteration}"

        short_names = self._reward_short_names()
        train_block = self._format_reward_block(
            metrics,
            prefix="train",
            total_key="train_reward_total",
            component_keys={short: f"train_reward_{name}" for name, short in short_names.items()},
        )
        if train_block:
            message += f" | {train_block}"

        val_block = self._format_reward_block(
            metrics,
            prefix="val",
            total_key="eval_reward_total",
            component_keys={short: f"eval_reward_{name}" for name, short in short_names.items()},
        )
        if val_block:
            message += f" | {val_block}"

        return message

    def _reward_short_names(self) -> dict[str, str]:
        aliases = {
            "distance": "dist",
            "holding": "hold",
            "safety": "safety",
        }
        return {name: aliases.get(name, name) for name in self.reward_component_names}

    def _default_d3po_preference(self) -> list[float]:
        n_objectives = int(getattr(self.policy, "n_objectives", 1))
        if n_objectives == 3:
            return [0.55, 0.15, 0.30]
        if n_objectives == 2:
            return [0.70, 0.30]
        return [1.0 / max(n_objectives, 1)] * max(n_objectives, 1)

    @staticmethod
    def _format_reward_block(
        metrics: dict[str, float],
        *,
        prefix: str,
        total_key: str,
        component_keys: dict[str, str],
    ) -> str:
        available = []
        for short_name, metric_key in component_keys.items():
            if metric_key in metrics:
                available.append((short_name, float(metrics[metric_key])))

        if not available:
            return ""

        total_value = float(metrics.get(total_key, sum(value for _, value in available)))
        parts = ", ".join(f"{name}={value:.3f}" for name, value in available)
        return f"{prefix}_reward(total={total_value:.3f}; {parts})"
