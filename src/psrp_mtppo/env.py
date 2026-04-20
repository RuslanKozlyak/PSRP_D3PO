from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from .config import EnvironmentConfig
from .instances import ProblemInstance, generate_instance


@dataclass(slots=True)
class JointAction:
    replenishment: np.ndarray
    route: list[int]


@dataclass(slots=True)
class RouteExecution:
    executed_route: list[int]
    distance: float
    invalid_moves: int


class IRPVMIEnv(gym.Env[dict[str, np.ndarray], dict[str, Any]]):
    metadata = {"render_modes": []}

    def __init__(self, config: EnvironmentConfig, instance: ProblemInstance | None = None):
        super().__init__()
        self.config = config
        self._base_instance = instance
        self.rng = np.random.default_rng(config.seed)
        self.instance: ProblemInstance | None = None

        self.day = 0
        self.inventory = np.empty(0, dtype=np.float32)
        self.supplier_inventory = float(config.supplier_initial_inventory)
        self.replenishment_history = np.empty((0, config.history_length), dtype=np.float32)
        self.demand_history = np.empty((0, config.history_length), dtype=np.float32)
        self.last_route: list[int] = [0]

        n = config.num_retailers
        history = config.history_length
        self.observation_space = gym.spaces.Dict(
            {
                "inventory_state": gym.spaces.Box(-np.inf, np.inf, shape=(n, 1 + 2 * history), dtype=np.float32),
                "inventory_node_features": gym.spaces.Box(-np.inf, np.inf, shape=(n, 5), dtype=np.float32),
                "retailer_coords": gym.spaces.Box(0.0, 1.0, shape=(n, 2), dtype=np.float32),
                "depot_coord": gym.spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32),
                "distance_matrix": gym.spaces.Box(0.0, np.inf, shape=(n + 1, n + 1), dtype=np.float32),
                "remaining_capacity": gym.spaces.Box(0.0, np.inf, shape=(n,), dtype=np.float32),
                "supplier_inventory": gym.spaces.Box(0.0, np.inf, shape=(1,), dtype=np.float32),
                "day_fraction": gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            }
        )
        self.action_space = gym.spaces.Dict(
            {
                "replenishment": gym.spaces.Box(0.0, config.vehicle_capacity, shape=(n,), dtype=np.float32),
                "route": gym.spaces.MultiDiscrete(np.full(config.max_route_length, n + 1, dtype=np.int64)),
            }
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        instance = self._base_instance
        if options and "instance" in options:
            instance = options["instance"]
        if instance is None:
            instance = generate_instance(self.config, self.rng)

        self.instance = instance
        self.day = 0
        self.inventory = instance.initial_retailer_inventory.copy()
        self.supplier_inventory = float(instance.supplier_initial_inventory)
        self.replenishment_history = np.zeros(
            (self.config.num_retailers, self.config.history_length),
            dtype=np.float32,
        )
        self.demand_history = np.zeros_like(self.replenishment_history)
        self.last_route = [0]

        return self._build_observation(), {"instance": instance}

    def _build_observation(self) -> dict[str, np.ndarray]:
        assert self.instance is not None
        last_replenishment = self.replenishment_history[:, -1]
        last_demand = self.demand_history[:, -1]

        inventory_state = np.concatenate(
            [
                self.inventory[:, None],
                self.replenishment_history,
                self.demand_history,
            ],
            axis=1,
        ).astype(np.float32)
        inventory_node_features = np.concatenate(
            [
                self.instance.retailer_coords,
                self.inventory[:, None],
                last_replenishment[:, None],
                last_demand[:, None],
            ],
            axis=1,
        ).astype(np.float32)

        return {
            "inventory_state": inventory_state,
            "inventory_node_features": inventory_node_features,
            "retailer_coords": self.instance.retailer_coords.copy(),
            "depot_coord": self.instance.depot_coord.copy(),
            "distance_matrix": self.instance.distance_matrix.copy(),
            "remaining_capacity": np.maximum(
                self.instance.retailer_capacity - self.inventory,
                0.0,
            ).astype(np.float32),
            "supplier_inventory": np.asarray([self.supplier_inventory], dtype=np.float32),
            "day_fraction": np.asarray(
                [self.day / max(self.config.horizon_days - 1, 1)],
                dtype=np.float32,
            ),
        }

    def clamp_replenishment(self, replenishment: np.ndarray) -> np.ndarray:
        assert self.instance is not None
        max_replenishment = np.minimum(
            self.config.vehicle_capacity,
            self.instance.retailer_capacity - self.inventory,
        )
        max_replenishment = np.minimum(max_replenishment, self.supplier_inventory)
        return np.clip(replenishment.astype(np.float32), 0.0, max_replenishment).astype(np.float32)

    def routing_features(self, replenishment: np.ndarray) -> np.ndarray:
        assert self.instance is not None
        return np.concatenate(
            [self.instance.retailer_coords, replenishment[:, None]],
            axis=1,
        ).astype(np.float32)

    def execute_route(self, replenishment: np.ndarray, route: list[int]) -> RouteExecution:
        assert self.instance is not None
        remaining = replenishment.copy()
        if route:
            raw_route = route[: self.config.max_route_length]
        else:
            raw_route = []

        distance = 0.0
        invalid_moves = 0
        current_node = 0
        current_load = self.config.vehicle_capacity
        executed = [0]
        positive_nodes = {idx + 1 for idx, value in enumerate(remaining) if value > 1e-6}

        def travel(next_node: int) -> None:
            nonlocal distance, current_node
            distance += float(self.instance.distance_matrix[current_node, next_node])
            current_node = next_node
            executed.append(next_node)

        for token in raw_route:
            if token == executed[-1] and token == 0:
                continue
            if token == 0:
                if current_node != 0:
                    travel(0)
                    current_load = self.config.vehicle_capacity
                continue

            if token not in positive_nodes:
                invalid_moves += 1
                continue

            idx = token - 1
            demand = float(remaining[idx])
            if demand <= 1e-6:
                invalid_moves += 1
                continue

            if demand > current_load + 1e-6 and current_node != 0:
                travel(0)
                current_load = self.config.vehicle_capacity

            if demand > current_load + 1e-6:
                invalid_moves += 1
                continue

            travel(token)
            current_load -= demand
            remaining[idx] = 0.0
            positive_nodes.discard(token)

            if current_load <= 1e-6 and positive_nodes:
                travel(0)
                current_load = self.config.vehicle_capacity

        while positive_nodes:
            feasible = [token for token in positive_nodes if remaining[token - 1] <= current_load + 1e-6]
            if not feasible:
                if current_node != 0:
                    travel(0)
                current_load = self.config.vehicle_capacity
                feasible = [token for token in positive_nodes if remaining[token - 1] <= current_load + 1e-6]
            next_node = min(feasible, key=lambda token: self.instance.distance_matrix[current_node, token])
            travel(next_node)
            current_load -= float(remaining[next_node - 1])
            remaining[next_node - 1] = 0.0
            positive_nodes.discard(next_node)

        if current_node != 0:
            travel(0)

        return RouteExecution(executed_route=executed, distance=distance, invalid_moves=invalid_moves)

    def step(self, action: JointAction) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        assert self.instance is not None
        day_index = self.day
        inventory_before_replenishment = self.inventory.copy()
        replenishment = self.clamp_replenishment(action.replenishment)
        route_execution = self.execute_route(replenishment, action.route)

        demand = self.instance.demands[self.day].astype(np.float32)
        inventory_before_demand = self.inventory + replenishment
        lost_sales = np.maximum(demand - inventory_before_demand, 0.0)
        fulfilled = demand - lost_sales
        ending_inventory = np.maximum(inventory_before_demand - demand, 0.0)

        holding_cost = float(ending_inventory.sum() * self.config.holding_cost)
        sales_loss_cost = float(
            lost_sales.sum() * self.config.product_price * self.config.sales_loss_penalty
        )
        route_cost = float(route_execution.distance * self.config.transport_cost_per_distance)
        total_reward = -(holding_cost + sales_loss_cost + route_cost)

        self.inventory = ending_inventory.astype(np.float32)
        self.supplier_inventory -= float(replenishment.sum())
        self.replenishment_history = np.roll(self.replenishment_history, shift=-1, axis=1)
        self.replenishment_history[:, -1] = replenishment
        self.demand_history = np.roll(self.demand_history, shift=-1, axis=1)
        self.demand_history[:, -1] = demand
        self.last_route = route_execution.executed_route
        self.day += 1

        terminated = self.day >= self.config.horizon_days
        truncated = False
        info = {
            "day": day_index,
            "inventory_cost": holding_cost + sales_loss_cost,
            "holding_cost": holding_cost,
            "sales_loss_cost": sales_loss_cost,
            "route_cost": route_cost,
            "route_distance": route_execution.distance,
            "invalid_moves": route_execution.invalid_moves,
            "fill_rate": float(fulfilled.sum() / np.maximum(demand.sum(), 1.0)),
            "fulfilled": float(fulfilled.sum()),
            "demand": float(demand.sum()),
            "demand_vector": demand.copy(),
            "lost_sales_vector": lost_sales.copy(),
            "inventory_before_replenishment": inventory_before_replenishment.copy(),
            "inventory_after_replenishment": inventory_before_demand.copy(),
            "ending_inventory": ending_inventory.copy(),
            "replenishment": replenishment,
            "executed_route": route_execution.executed_route,
            "active_retailers": int(np.count_nonzero(replenishment > 1e-6)),
            "route_stops": max(len(route_execution.executed_route) - 2, 0),
        }
        return self._build_observation(), total_reward, terminated, truncated, info
