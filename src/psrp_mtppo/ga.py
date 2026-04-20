"""Genetic Algorithm baselines for IRP-VMI.

Three variants are implemented, matching Lu et al. 2025 §5:

* :func:`ga_inventory_action` — GA(INV): optimises the per-day replenishment
  vector (a chromosome of ``n_retailers`` real numbers) given the current
  inventory state.
* :func:`ga_route` — GA(VRP): given a fixed replenishment vector, evolves a
  visit permutation and returns the full route with depot returns inserted
  when the vehicle capacity is exhausted.
* :func:`ga_irp_action` — GA(IRP): jointly evolves the replenishment vector
  using :func:`ga_route` for fitness evaluation of each candidate.

All GAs share the hyperparameters from Table 3 of the paper (population 100,
crossover 0.6, mutation 0.05, parent/survival ratio 0.5, selection pressure 1.5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .config import EnvironmentConfig


@dataclass(slots=True)
class GAConfig:
    population_size: int = 100
    crossover_prob: float = 0.6
    mutation_prob: float = 0.05
    selection_pressure: float = 1.5
    parent_ratio: float = 0.5
    generations: int = 30
    seed: int | None = None


def _linear_ranking_probabilities(size: int, pressure: float) -> np.ndarray:
    """Linear ranking selection weights (Baker 1987) with the given pressure."""
    ranks = np.arange(1, size + 1, dtype=np.float64)
    best = pressure
    worst = 2.0 - pressure
    probs = (best - (best - worst) * (ranks - 1) / max(size - 1, 1)) / size
    return probs / probs.sum()


def _rank_select(fitness: np.ndarray, count: int, rng: np.random.Generator, pressure: float) -> np.ndarray:
    """Return ``count`` indices drawn by rank selection (higher fitness -> higher prob)."""
    order = np.argsort(-fitness)  # descending fitness -> rank 1 = best
    probs = _linear_ranking_probabilities(len(fitness), pressure)
    selected_ranks = rng.choice(len(fitness), size=count, replace=True, p=probs)
    return order[selected_ranks]


# ---------------------------------------------------------------------------
# VRP GA (single vehicle with multiple return trips to the depot)
# ---------------------------------------------------------------------------


def _decode_vrp_chromosome(
    permutation: np.ndarray,
    replenishment: np.ndarray,
    vehicle_capacity: float,
) -> list[int]:
    """Turn a visit permutation into a full depot-anchored route."""
    route = [0]
    load = vehicle_capacity
    for idx in permutation:
        demand = float(replenishment[idx])
        if demand <= 1e-6:
            continue
        if demand > load + 1e-6:
            if route[-1] != 0:
                route.append(0)
            load = vehicle_capacity
        route.append(int(idx) + 1)
        load -= demand
    if route[-1] != 0:
        route.append(0)
    return route


def _route_distance(route: list[int], distance_matrix: np.ndarray) -> float:
    if len(route) < 2:
        return 0.0
    idx = np.asarray(route, dtype=np.int64)
    return float(distance_matrix[idx[:-1], idx[1:]].sum())


def _vrp_order_crossover(parent_a: np.ndarray, parent_b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    size = len(parent_a)
    if size < 2:
        return parent_a.copy()
    start, end = sorted(rng.integers(0, size, size=2).tolist())
    if end == start:
        end = min(size, start + 1)
    child = np.full(size, -1, dtype=parent_a.dtype)
    child[start:end] = parent_a[start:end]
    insert = [gene for gene in parent_b if gene not in child[start:end]]
    pointer = 0
    for position in range(size):
        if child[position] == -1:
            child[position] = insert[pointer]
            pointer += 1
    return child


def _vrp_mutate(permutation: np.ndarray, prob: float, rng: np.random.Generator) -> np.ndarray:
    out = permutation.copy()
    if len(out) < 2:
        return out
    for position in range(len(out)):
        if rng.random() < prob:
            swap_with = int(rng.integers(0, len(out)))
            out[position], out[swap_with] = out[swap_with], out[position]
    return out


def ga_route(
    replenishment: np.ndarray,
    distance_matrix: np.ndarray,
    vehicle_capacity: float,
    config: GAConfig | None = None,
) -> tuple[list[int], float]:
    """Evolve a delivery route for the given replenishment plan."""
    config = config or GAConfig()
    rng = np.random.default_rng(config.seed)
    active = np.flatnonzero(replenishment > 1e-6)
    if len(active) == 0:
        return [0], 0.0
    if len(active) == 1:
        route = _decode_vrp_chromosome(active, replenishment, vehicle_capacity)
        return route, _route_distance(route, distance_matrix)

    population = np.stack([rng.permutation(active) for _ in range(config.population_size)])
    fitness = np.array(
        [
            -_route_distance(_decode_vrp_chromosome(ind, replenishment, vehicle_capacity), distance_matrix)
            for ind in population
        ]
    )

    for _ in range(config.generations):
        parent_count = max(2, int(config.population_size * config.parent_ratio))
        parent_idx = _rank_select(fitness, parent_count, rng, config.selection_pressure)
        parents = population[parent_idx]

        offspring: list[np.ndarray] = []
        while len(offspring) < config.population_size:
            a, b = parents[rng.integers(0, parent_count)], parents[rng.integers(0, parent_count)]
            if rng.random() < config.crossover_prob:
                child = _vrp_order_crossover(a, b, rng)
            else:
                child = a.copy()
            child = _vrp_mutate(child, config.mutation_prob, rng)
            offspring.append(child)
        offspring_array = np.stack(offspring[: config.population_size])
        offspring_fitness = np.array(
            [
                -_route_distance(_decode_vrp_chromosome(ind, replenishment, vehicle_capacity), distance_matrix)
                for ind in offspring_array
            ]
        )

        pool = np.concatenate([population, offspring_array])
        pool_fitness = np.concatenate([fitness, offspring_fitness])
        top = np.argsort(-pool_fitness)[: config.population_size]
        population = pool[top]
        fitness = pool_fitness[top]

    best = int(np.argmax(fitness))
    route = _decode_vrp_chromosome(population[best], replenishment, vehicle_capacity)
    return route, _route_distance(route, distance_matrix)


# ---------------------------------------------------------------------------
# INV GA (per-day replenishment vector)
# ---------------------------------------------------------------------------


def _inv_cost(
    replenishment: np.ndarray,
    inventory: np.ndarray,
    demand_forecast: np.ndarray,
    config: EnvironmentConfig,
) -> float:
    after = inventory + replenishment
    fulfilled = np.minimum(after, demand_forecast)
    lost = np.maximum(demand_forecast - after, 0.0)
    ending = np.maximum(after - demand_forecast, 0.0)
    holding = float(ending.sum()) * config.holding_cost
    sales_loss = float(lost.sum()) * config.product_price * config.sales_loss_penalty
    return holding + sales_loss


def _sample_inventory_chromosome(
    inventory: np.ndarray,
    config: EnvironmentConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    max_per_retailer = np.minimum(config.retailer_capacity - inventory, config.vehicle_capacity).clip(min=0.0)
    chromosome = rng.uniform(0.0, 1.0, size=inventory.shape) * max_per_retailer
    chromosome = _repair_inventory_chromosome(chromosome, inventory, config)
    return chromosome


def _repair_inventory_chromosome(
    chromosome: np.ndarray,
    inventory: np.ndarray,
    config: EnvironmentConfig,
) -> np.ndarray:
    max_per_retailer = np.minimum(config.retailer_capacity - inventory, config.vehicle_capacity).clip(min=0.0)
    chromosome = np.clip(chromosome, 0.0, max_per_retailer)
    total = float(chromosome.sum())
    if total > config.vehicle_capacity and total > 0:
        chromosome = chromosome * (config.vehicle_capacity / total)
    return chromosome


def _blend_crossover(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    alpha = rng.uniform(0.0, 1.0, size=a.shape)
    return alpha * a + (1.0 - alpha) * b


def _gaussian_mutate(
    chromosome: np.ndarray,
    prob: float,
    scale: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    mask = rng.random(chromosome.shape) < prob
    noise = rng.normal(0.0, scale, size=chromosome.shape)
    return chromosome + mask.astype(np.float64) * noise


def ga_inventory_action(
    inventory: np.ndarray,
    demand_forecast: np.ndarray,
    config: EnvironmentConfig,
    ga_config: GAConfig | None = None,
) -> np.ndarray:
    """GA(INV): evolve a single-day replenishment vector against the forecast."""
    ga_config = ga_config or GAConfig()
    rng = np.random.default_rng(ga_config.seed)

    population = np.stack(
        [_sample_inventory_chromosome(inventory, config, rng) for _ in range(ga_config.population_size)]
    )
    fitness = np.array([-_inv_cost(ind, inventory, demand_forecast, config) for ind in population])
    scale = np.minimum(config.retailer_capacity - inventory, config.vehicle_capacity).clip(min=1.0) * 0.15

    for _ in range(ga_config.generations):
        parent_count = max(2, int(ga_config.population_size * ga_config.parent_ratio))
        parent_idx = _rank_select(fitness, parent_count, rng, ga_config.selection_pressure)
        parents = population[parent_idx]

        offspring: list[np.ndarray] = []
        while len(offspring) < ga_config.population_size:
            a = parents[rng.integers(0, parent_count)]
            b = parents[rng.integers(0, parent_count)]
            child = _blend_crossover(a, b, rng) if rng.random() < ga_config.crossover_prob else a.copy()
            child = _gaussian_mutate(child, ga_config.mutation_prob, scale, rng)
            child = _repair_inventory_chromosome(child, inventory, config)
            offspring.append(child)
        offspring_array = np.stack(offspring[: ga_config.population_size])
        offspring_fitness = np.array(
            [-_inv_cost(ind, inventory, demand_forecast, config) for ind in offspring_array]
        )

        pool = np.concatenate([population, offspring_array])
        pool_fitness = np.concatenate([fitness, offspring_fitness])
        top = np.argsort(-pool_fitness)[: ga_config.population_size]
        population = pool[top]
        fitness = pool_fitness[top]

    best = int(np.argmax(fitness))
    return population[best].astype(np.float32)


# ---------------------------------------------------------------------------
# IRP GA (joint replenishment + routing)
# ---------------------------------------------------------------------------


def ga_irp_action(
    inventory: np.ndarray,
    demand_forecast: np.ndarray,
    distance_matrix: np.ndarray,
    config: EnvironmentConfig,
    ga_config: GAConfig | None = None,
    inner_route_config: GAConfig | None = None,
) -> tuple[np.ndarray, list[int]]:
    """GA(IRP): joint replenishment + routing evaluated with GA(VRP) as inner loop."""
    ga_config = ga_config or GAConfig()
    inner_route_config = inner_route_config or GAConfig(generations=10, population_size=40)
    rng = np.random.default_rng(ga_config.seed)

    def route_cost(replenishment: np.ndarray) -> tuple[list[int], float]:
        cfg = GAConfig(
            population_size=inner_route_config.population_size,
            crossover_prob=inner_route_config.crossover_prob,
            mutation_prob=inner_route_config.mutation_prob,
            selection_pressure=inner_route_config.selection_pressure,
            parent_ratio=inner_route_config.parent_ratio,
            generations=inner_route_config.generations,
            seed=int(rng.integers(0, 10_000)),
        )
        return ga_route(replenishment, distance_matrix, config.vehicle_capacity, cfg)

    def cost(replenishment: np.ndarray) -> tuple[float, list[int], float]:
        inv_cost = _inv_cost(replenishment, inventory, demand_forecast, config)
        route, distance = route_cost(replenishment)
        total = inv_cost + distance * config.transport_cost_per_distance
        return total, route, distance

    population = np.stack(
        [_sample_inventory_chromosome(inventory, config, rng) for _ in range(ga_config.population_size)]
    )
    evaluations = [cost(ind) for ind in population]
    fitness = np.array([-ev[0] for ev in evaluations])
    routes = [ev[1] for ev in evaluations]
    scale = np.minimum(config.retailer_capacity - inventory, config.vehicle_capacity).clip(min=1.0) * 0.15

    for _ in range(ga_config.generations):
        parent_count = max(2, int(ga_config.population_size * ga_config.parent_ratio))
        parent_idx = _rank_select(fitness, parent_count, rng, ga_config.selection_pressure)
        parents = population[parent_idx]

        offspring: list[np.ndarray] = []
        while len(offspring) < ga_config.population_size:
            a = parents[rng.integers(0, parent_count)]
            b = parents[rng.integers(0, parent_count)]
            child = _blend_crossover(a, b, rng) if rng.random() < ga_config.crossover_prob else a.copy()
            child = _gaussian_mutate(child, ga_config.mutation_prob, scale, rng)
            child = _repair_inventory_chromosome(child, inventory, config)
            offspring.append(child)
        offspring_array = np.stack(offspring[: ga_config.population_size])
        offspring_eval = [cost(ind) for ind in offspring_array]
        offspring_fitness = np.array([-ev[0] for ev in offspring_eval])
        offspring_routes = [ev[1] for ev in offspring_eval]

        pool = np.concatenate([population, offspring_array])
        pool_fitness = np.concatenate([fitness, offspring_fitness])
        pool_routes = routes + offspring_routes
        top = np.argsort(-pool_fitness)[: ga_config.population_size]
        population = pool[top]
        fitness = pool_fitness[top]
        routes = [pool_routes[idx] for idx in top]

    best = int(np.argmax(fitness))
    return population[best].astype(np.float32), routes[best]


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def build_forecast(obs: dict[str, np.ndarray]) -> np.ndarray:
    """Simple moving-average forecast built from the per-retailer demand history."""
    history = obs["inventory_state"][:, 1 + (obs["inventory_state"].shape[1] - 1) // 2 :]
    if history.size == 0:
        return np.zeros(obs["inventory_state"].shape[0], dtype=np.float32)
    return history.mean(axis=1).astype(np.float32)


Policy = Callable[[dict[str, np.ndarray]], tuple[np.ndarray, list[int]]]
