from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class EnvironmentConfig:
    num_retailers: int = 20
    horizon_days: int = 90
    history_length: int = 7
    holding_cost: float = 2.0
    retailer_capacity: float = 100.0
    initial_retailer_inventory: float = 100.0
    supplier_initial_inventory: float = 1_000_000.0
    demand_low: int = 1
    demand_high: int = 50
    product_price: float = 20.0
    sales_loss_penalty: float = 0.3
    vehicle_capacity: float = 100.0
    transport_cost_per_distance: float = 1000.0
    coord_low: float = 0.0
    coord_high: float = 1.0
    seed: int = 7

    @property
    def max_route_length(self) -> int:
        return 2 * self.num_retailers + 1


@dataclass(slots=True)
class ModelConfig:
    gin_layers: int = 3
    gin_dims: tuple[int, ...] = (64, 128, 128)
    mlp_dims: tuple[int, ...] = (128, 128)
    state_embed_dim: int = 128
    attention_heads: int = 8
    attention_layers: int = 2
    dropout: float = 0.1


@dataclass(slots=True)
class TrainingConfig:
    learning_rate: float = 1e-3
    train_batch_size: int = 256
    gamma: float = 0.9
    ppo_epochs: int = 4
    minibatch_size: int = 64
    clip_param: float = 0.2
    value_clip_param: float = 0.1
    kl_coefficient: float = 0.2
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    max_grad_norm: float = 1.0
    training_iterations: int = 12
    evaluation_episodes: int = 4
    num_envs: int = 8
    device: str = "auto"
    seed: int = 7

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch  # local import to avoid hard dependency at dataclass creation

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"


@dataclass(slots=True)
class ExperimentConfig:
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
