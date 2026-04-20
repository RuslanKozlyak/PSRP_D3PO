from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical, Normal

from ..config import EnvironmentConfig, ModelConfig
from ..utils import complete_adjacency
from .attention import TransformerBlock
from .gin import GINEncoder, make_mlp


@dataclass(slots=True)
class InventorySample:
    """Inventory actor output.

    ``latent`` is the pre-squash Normal sample — this is what the PPO ratio
    should be evaluated on. ``action`` is the bounded replenishment vector fed
    to the environment (``sigmoid(latent) * max_replenishment``). Storing the
    latent avoids TransformedDistribution log-prob blow-ups at the sigmoid
    saturation boundary.
    """

    action: torch.Tensor
    latent: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor


@dataclass(slots=True)
class RoutingSample:
    """Vectorised routing sample.

    ``route_ids`` is an ``[B, max_route_length]`` long tensor padded with 0 (depot).
    ``route_lengths`` stores the number of non-padded steps taken for each batch
    element (excluding the initial depot). Use :meth:`routes_as_lists` to convert
    to python lists (e.g. to feed the environment).
    """

    route_ids: torch.Tensor
    route_lengths: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor

    def routes_as_lists(self) -> list[list[int]]:
        lengths = self.route_lengths.detach().cpu().tolist()
        ids = self.route_ids.detach().cpu().tolist()
        routes: list[list[int]] = []
        for row, length in zip(ids, lengths):
            route = [0] + [int(x) for x in row[:length]]
            if not route or route[-1] != 0:
                route.append(0)
            routes.append(route)
        return routes


class InventoryActor(nn.Module):
    def __init__(self, env_config: EnvironmentConfig, model_config: ModelConfig) -> None:
        super().__init__()
        self.env_config = env_config
        self.node_encoder = GINEncoder(5, model_config.gin_dims, model_config.dropout)
        self.state_encoder = make_mlp(
            1 + 2 * env_config.history_length,
            (model_config.state_embed_dim,),
            model_config.state_embed_dim,
            model_config.dropout,
        )
        hidden_dim = self.node_encoder.output_dim + model_config.state_embed_dim
        self.project = nn.Linear(hidden_dim, model_config.mlp_dims[0])
        self.attention_layers = nn.ModuleList(
            TransformerBlock(model_config.mlp_dims[0], model_config.attention_heads, model_config.dropout)
            for _ in range(model_config.attention_layers)
        )
        self.decoder = make_mlp(
            model_config.mlp_dims[0],
            model_config.mlp_dims[1:],
            model_config.mlp_dims[-1],
            model_config.dropout,
        )
        self.mean_head = nn.Linear(model_config.mlp_dims[-1], 1)
        self.log_std_head = nn.Linear(model_config.mlp_dims[-1], 1)
        self.register_buffer(
            "_adjacency",
            complete_adjacency(env_config.num_retailers),
            persistent=False,
        )

    def _distribution(
        self,
        obs: dict[str, torch.Tensor],
    ) -> tuple[Normal, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = obs["inventory_node_features"]
        state = obs["inventory_state"]
        batch_size = features.shape[0]
        adjacency = self._adjacency.expand(batch_size, -1, -1)
        graph_hidden = self.node_encoder(features, adjacency)
        state_hidden = self.state_encoder(state)
        hidden = torch.cat([graph_hidden, state_hidden], dim=-1)
        hidden = self.project(hidden)
        for layer in self.attention_layers:
            hidden = layer(hidden)
        hidden = self.decoder(hidden)

        mean = self.mean_head(hidden).squeeze(-1)
        log_std = self.log_std_head(hidden).squeeze(-1).clamp(-4.0, 2.0)
        std = log_std.exp().clamp_min(1e-4)
        max_replenishment = torch.minimum(
            obs["remaining_capacity"],
            torch.full_like(obs["remaining_capacity"], self.env_config.vehicle_capacity),
        ).clamp_min(1e-6)
        normal = Normal(mean, std)
        return normal, mean, std, max_replenishment

    def sample(self, obs: dict[str, torch.Tensor], greedy: bool = False) -> InventorySample:
        normal, mean, std, max_replenishment = self._distribution(obs)
        if greedy:
            latent = mean
        else:
            latent = normal.rsample()
        action = torch.sigmoid(latent) * max_replenishment
        log_prob = normal.log_prob(latent).sum(dim=-1)
        entropy = normal.entropy().sum(dim=-1)
        return InventorySample(
            action=action,
            latent=latent.detach(),
            log_prob=log_prob,
            entropy=entropy,
            mean=mean,
            std=std,
        )

    def evaluate_actions(
        self,
        obs: dict[str, torch.Tensor],
        latents: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normal, _mean, _std, _max = self._distribution(obs)
        log_prob = normal.log_prob(latents).sum(dim=-1)
        entropy = normal.entropy().sum(dim=-1)
        return log_prob, entropy


class RoutingActor(nn.Module):
    def __init__(self, env_config: EnvironmentConfig, model_config: ModelConfig) -> None:
        super().__init__()
        self.env_config = env_config
        self.node_encoder = GINEncoder(3, model_config.gin_dims, model_config.dropout)
        self.route_project = nn.Linear(self.node_encoder.output_dim, model_config.mlp_dims[0])
        self.attention_layers = nn.ModuleList(
            TransformerBlock(model_config.mlp_dims[0], model_config.attention_heads, model_config.dropout)
            for _ in range(model_config.attention_layers)
        )
        scorer_input_dim = model_config.mlp_dims[0] * 2 + 4
        self.node_scorer = make_mlp(
            scorer_input_dim,
            model_config.mlp_dims,
            1,
            model_config.dropout,
        )
        self.depot_scorer = make_mlp(
            model_config.mlp_dims[0] + 4,
            model_config.mlp_dims,
            1,
            model_config.dropout,
        )
        self.register_buffer(
            "_adjacency",
            complete_adjacency(env_config.num_retailers),
            persistent=False,
        )

    def _encode_nodes(self, routing_features: torch.Tensor) -> torch.Tensor:
        batch_size = routing_features.shape[0]
        adjacency = self._adjacency.expand(batch_size, -1, -1)
        hidden = self.node_encoder(routing_features, adjacency)
        hidden = self.route_project(hidden)
        for layer in self.attention_layers:
            hidden = layer(hidden)
        return hidden

    def _build_logits(
        self,
        node_embeddings: torch.Tensor,
        coords: torch.Tensor,
        depot_coord: torch.Tensor,
        remaining: torch.Tensor,
        visited: torch.Tensor,
        current_node: torch.Tensor,
        current_load: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_retailers, hidden_dim = node_embeddings.shape
        Q = max(self.env_config.vehicle_capacity, 1.0)

        # Gather the current coordinate: depot for 0, else retailer coords.
        current_retailer_index = (current_node.clamp_min(1) - 1).view(batch_size, 1, 1).expand(-1, 1, 2)
        current_retailer_coord = coords.gather(1, current_retailer_index).squeeze(1)
        at_depot = (current_node == 0).unsqueeze(-1)
        current_coord = torch.where(at_depot, depot_coord, current_retailer_coord)

        retailer_coords = current_coord.unsqueeze(1).expand(-1, num_retailers, -1)
        remaining_ratio = (remaining / Q).unsqueeze(-1)
        load_ratio = (current_load / Q).view(batch_size, 1, 1).expand(-1, num_retailers, -1)
        current_distance = torch.norm(coords - retailer_coords, dim=-1, keepdim=True)
        global_context = node_embeddings.mean(dim=1, keepdim=True).expand(-1, num_retailers, -1)

        node_input = torch.cat(
            [
                node_embeddings,
                global_context,
                remaining_ratio,
                load_ratio,
                current_distance,
                visited.float().unsqueeze(-1),
            ],
            dim=-1,
        )
        retailer_logits = self.node_scorer(node_input).squeeze(-1)
        depot_input = torch.cat(
            [
                node_embeddings.mean(dim=1),
                (current_load / Q).unsqueeze(-1),
                remaining.sum(dim=1, keepdim=True) / Q,
                visited.float().mean(dim=1, keepdim=True),
                torch.ones(batch_size, 1, device=node_embeddings.device),
            ],
            dim=-1,
        )
        depot_logits = self.depot_scorer(depot_input)
        return torch.cat([depot_logits, retailer_logits], dim=1)

    def _build_mask(
        self,
        remaining: torch.Tensor,
        visited: torch.Tensor,
        current_node: torch.Tensor,
        current_load: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_retailers = remaining.shape
        valid = (~visited) & (remaining > 1e-6)
        feasible = valid & (remaining <= current_load.unsqueeze(-1) + 1e-6)
        any_valid = valid.any(dim=1, keepdim=True)
        any_feasible = feasible.any(dim=1, keepdim=True)
        at_depot = (current_node == 0).unsqueeze(-1)

        depot_allowed = (~at_depot) | (any_valid & ~any_feasible)
        # if nothing left to serve and we are at the depot -> terminal; depot not needed
        depot_allowed = depot_allowed & any_valid
        # when already everywhere served and at depot, allow depot to act as no-op sink
        no_more_actions = (~any_valid) & at_depot
        depot_allowed = depot_allowed | no_more_actions
        return torch.cat([depot_allowed, feasible], dim=1)

    def _rollout_with_actions(
        self,
        obs: dict[str, torch.Tensor],
        replenishment: torch.Tensor,
        greedy: bool = False,
        forced_actions: torch.Tensor | None = None,
        forced_mask: torch.Tensor | None = None,
    ) -> RoutingSample:
        coords = obs["retailer_coords"]
        depot_coord = obs["depot_coord"]
        routing_features = torch.cat([coords, replenishment.unsqueeze(-1)], dim=-1)
        node_embeddings = self._encode_nodes(routing_features)

        batch_size, num_retailers = replenishment.shape
        device = replenishment.device
        Q = float(self.env_config.vehicle_capacity)
        max_steps = self.env_config.max_route_length

        remaining = replenishment.detach().clone()
        visited = torch.zeros_like(remaining, dtype=torch.bool)
        current_node = torch.zeros(batch_size, dtype=torch.long, device=device)
        current_load = torch.full((batch_size,), Q, device=device)
        done = torch.zeros(batch_size, dtype=torch.bool, device=device)

        total_log_prob = torch.zeros(batch_size, device=device)
        total_entropy = torch.zeros(batch_size, device=device)
        action_history: list[torch.Tensor] = []
        route_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)

        if forced_actions is not None:
            assert forced_mask is not None
            max_steps = min(max_steps, forced_actions.shape[1])

        for step in range(max_steps):
            logits = self._build_logits(
                node_embeddings=node_embeddings,
                coords=coords,
                depot_coord=depot_coord,
                remaining=remaining,
                visited=visited,
                current_node=current_node,
                current_load=current_load,
            )
            mask = self._build_mask(
                remaining=remaining,
                visited=visited,
                current_node=current_node,
                current_load=current_load,
            )
            # Guarantee at least one valid index per row by allowing depot for fully-done rows.
            mask = mask.clone()
            mask[done, :] = False
            mask[done, 0] = True

            masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min / 4.0)
            distribution = Categorical(logits=masked_logits)

            if forced_actions is not None:
                action = forced_actions[:, step]
                step_mask = forced_mask[:, step] & (~done)
            else:
                if greedy:
                    action = masked_logits.argmax(dim=-1)
                else:
                    action = distribution.sample()
                step_mask = ~done

            log_prob_step = distribution.log_prob(action)
            entropy_step = distribution.entropy()
            active_float = step_mask.float()
            total_log_prob = total_log_prob + log_prob_step * active_float
            total_entropy = total_entropy + entropy_step * active_float

            safe_action = torch.where(step_mask, action, torch.zeros_like(action))
            action_history.append(safe_action)
            route_lengths = route_lengths + step_mask.long()

            is_depot = (safe_action == 0) & step_mask
            # At depot: reset load, current node to 0
            current_load = torch.where(is_depot, torch.full_like(current_load, Q), current_load)
            current_node = torch.where(is_depot, torch.zeros_like(current_node), current_node)

            # At retailer: mark visited and reduce load
            is_retailer = step_mask & (~is_depot)
            retailer_idx = (safe_action - 1).clamp_min(0)
            retailer_amt = remaining.gather(1, retailer_idx.unsqueeze(-1)).squeeze(-1)
            retailer_amt = torch.where(is_retailer, retailer_amt, torch.zeros_like(retailer_amt))

            current_load = current_load - retailer_amt
            current_node = torch.where(is_retailer, safe_action, current_node)

            # scatter updates gated by is_retailer
            retailer_idx_col = retailer_idx.unsqueeze(-1)
            visited_update = visited.clone()
            visited_update.scatter_(
                1,
                retailer_idx_col,
                (visited.gather(1, retailer_idx_col) | is_retailer.unsqueeze(-1)),
            )
            visited = visited_update
            remaining_update = remaining.clone()
            remaining_update.scatter_(
                1,
                retailer_idx_col,
                torch.where(
                    is_retailer.unsqueeze(-1),
                    torch.zeros_like(retailer_amt.unsqueeze(-1)),
                    remaining.gather(1, retailer_idx_col),
                ),
            )
            remaining = remaining_update

            # done = nothing left to serve AND vehicle is back at depot
            empty = (remaining <= 1e-6).all(dim=1)
            done = done | (empty & (current_node == 0))

            if bool(done.all()):
                break

        if not action_history:
            route_ids = torch.zeros((batch_size, 0), dtype=torch.long, device=device)
        else:
            route_ids = torch.stack(action_history, dim=1)

        return RoutingSample(
            route_ids=route_ids,
            route_lengths=route_lengths,
            log_prob=total_log_prob,
            entropy=total_entropy,
        )

    def sample_route(
        self,
        obs: dict[str, torch.Tensor],
        replenishment: torch.Tensor,
        greedy: bool = False,
    ) -> RoutingSample:
        return self._rollout_with_actions(obs, replenishment, greedy=greedy)

    def evaluate_routes(
        self,
        obs: dict[str, torch.Tensor],
        replenishment: torch.Tensor,
        routes: list[list[int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = replenishment.device
        batch_size = replenishment.shape[0]
        max_len = max((len(route) - 1 for route in routes), default=0)
        forced_actions = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
        forced_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
        for row, route in enumerate(routes):
            tail = route[1:]
            if not tail:
                continue
            forced_actions[row, : len(tail)] = torch.as_tensor(tail, dtype=torch.long, device=device)
            forced_mask[row, : len(tail)] = True
        sample = self._rollout_with_actions(
            obs,
            replenishment,
            forced_actions=forced_actions,
            forced_mask=forced_mask,
        )
        return sample.log_prob, sample.entropy
