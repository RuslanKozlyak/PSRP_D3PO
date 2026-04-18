from __future__ import annotations

import torch
from torch import nn

from src.policy.heads import BetaHead, MaskedGlimpse, PointerHead


class HierarchicalDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        n_products: int,
        n_glimpse_heads: int,
        tanh_clip: float = 10.0,
    ) -> None:
        super().__init__()
        self.vehicle_head = PointerHead(hidden_dim, tanh_clip=tanh_clip)
        self.node_head = PointerHead(hidden_dim, tanh_clip=tanh_clip)
        self.node_glimpse = MaskedGlimpse(hidden_dim, n_glimpse_heads)
        self.quantity_head = BetaHead(hidden_dim * 4 + n_products, hidden_dim)
        self.decision_stage_embedding = nn.Embedding(3, hidden_dim)
        self.context_projection = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.depot_projection = nn.Linear(hidden_dim, hidden_dim)

    def build_context(
        self,
        station_embeddings: torch.Tensor,
        vehicle_embeddings: torch.Tensor,
        global_embedding: torch.Tensor,
        selected_vehicle: torch.Tensor | None = None,
        selected_node: torch.Tensor | None = None,
        decision_stage: int = 0,
    ) -> torch.Tensor:
        batch_size, _, hidden_dim = station_embeddings.shape
        pooled_station = station_embeddings.mean(dim=1)

        if selected_vehicle is None:
            vehicle_context = torch.zeros(batch_size, hidden_dim, device=station_embeddings.device)
        else:
            vehicle_context = _gather_sequence(vehicle_embeddings, selected_vehicle)

        if selected_node is None:
            node_context = torch.zeros(batch_size, hidden_dim, device=station_embeddings.device)
        else:
            station_indices = torch.clamp(selected_node - 1, min=0)
            node_context = _gather_sequence(station_embeddings, station_indices)
            node_context = torch.where(
                (selected_node > 0).unsqueeze(-1),
                node_context,
                torch.zeros_like(node_context),
            )

        stage = torch.full(
            (batch_size,),
            decision_stage,
            dtype=torch.long,
            device=station_embeddings.device,
        )
        stage_embedding = self.decision_stage_embedding(stage)
        return self.context_projection(
            torch.cat(
                [
                    pooled_station,
                    vehicle_context,
                    node_context,
                    global_embedding,
                    stage_embedding,
                ],
                dim=-1,
            )
        )

    def build_node_candidates(
        self,
        station_embeddings: torch.Tensor,
        global_embedding: torch.Tensor,
    ) -> torch.Tensor:
        depot_token = self.depot_projection(global_embedding).unsqueeze(1)
        return torch.cat([depot_token, station_embeddings], dim=1)

    def vehicle_logits(
        self,
        station_embeddings: torch.Tensor,
        vehicle_embeddings: torch.Tensor,
        global_embedding: torch.Tensor,
        mask_vehicle: torch.Tensor,
    ) -> torch.Tensor:
        context = self.build_context(
            station_embeddings=station_embeddings,
            vehicle_embeddings=vehicle_embeddings,
            global_embedding=global_embedding,
            decision_stage=0,
        )
        return self.vehicle_head(context, vehicle_embeddings, mask_vehicle)

    def node_logits(
        self,
        station_embeddings: torch.Tensor,
        vehicle_embeddings: torch.Tensor,
        global_embedding: torch.Tensor,
        selected_vehicle: torch.Tensor,
        mask_node: torch.Tensor,
    ) -> torch.Tensor:
        node_candidates = self.build_node_candidates(station_embeddings, global_embedding)
        context = self.build_context(
            station_embeddings=station_embeddings,
            vehicle_embeddings=vehicle_embeddings,
            global_embedding=global_embedding,
            selected_vehicle=selected_vehicle,
            decision_stage=1,
        )
        glimpse_context = self.node_glimpse(context, node_candidates, mask_node)
        return self.node_head(glimpse_context, node_candidates, mask_node)

    def quantity_params(
        self,
        station_embeddings: torch.Tensor,
        vehicle_embeddings: torch.Tensor,
        compartment_embeddings: torch.Tensor,
        global_embedding: torch.Tensor,
        quantity_mask: torch.Tensor,
        selected_vehicle: torch.Tensor,
        selected_node: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, _, hidden_dim = compartment_embeddings.shape
        vehicle_context = _gather_sequence(vehicle_embeddings, selected_vehicle)
        station_indices = torch.clamp(selected_node - 1, min=0)
        node_context = _gather_sequence(station_embeddings, station_indices)
        node_context = torch.where(
            (selected_node > 0).unsqueeze(-1),
            node_context,
            torch.zeros_like(node_context),
        )

        compartment_context = _gather_vehicle_compartments(compartment_embeddings, selected_vehicle)
        repeated_global = global_embedding.unsqueeze(1).expand(batch_size, compartment_context.size(1), hidden_dim)
        repeated_vehicle = vehicle_context.unsqueeze(1).expand_as(compartment_context)
        repeated_node = node_context.unsqueeze(1).expand_as(compartment_context)
        quantity_mask_vehicle = _gather_vehicle_masks(quantity_mask, selected_vehicle).float()
        quantity_features = torch.cat(
            [
                repeated_global,
                repeated_vehicle,
                repeated_node,
                compartment_context,
                quantity_mask_vehicle,
            ],
            dim=-1,
        )
        alpha, beta = self.quantity_head(quantity_features)
        alpha = alpha.unsqueeze(-1).expand(-1, -1, quantity_mask_vehicle.size(-1))
        beta = beta.unsqueeze(-1).expand(-1, -1, quantity_mask_vehicle.size(-1))
        alpha = alpha.masked_fill(~quantity_mask_vehicle.bool(), 1.0)
        beta = beta.masked_fill(~quantity_mask_vehicle.bool(), 1.0)
        return alpha, beta


def _gather_sequence(sequence: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    gather_index = index.view(-1, 1, 1).expand(-1, 1, sequence.size(-1))
    return sequence.gather(dim=1, index=gather_index).squeeze(1)


def _gather_vehicle_compartments(
    compartment_embeddings: torch.Tensor,
    vehicle_index: torch.Tensor,
) -> torch.Tensor:
    gather_index = vehicle_index.view(-1, 1, 1, 1).expand(
        -1,
        1,
        compartment_embeddings.size(2),
        compartment_embeddings.size(3),
    )
    return compartment_embeddings.gather(dim=1, index=gather_index).squeeze(1)


def _gather_vehicle_masks(quantity_mask: torch.Tensor, vehicle_index: torch.Tensor) -> torch.Tensor:
    gather_index = vehicle_index.view(-1, 1, 1, 1).expand(
        -1,
        1,
        quantity_mask.size(2),
        quantity_mask.size(3),
    )
    return quantity_mask.gather(dim=1, index=gather_index).squeeze(1)
