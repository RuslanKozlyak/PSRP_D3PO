from __future__ import annotations

import argparse
import json

from src.classes.data_generators.GraphGenerator import GraphGenerator


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True, help="JSON graph-generator configuration")
    args = parser.parse_args()

    payload = json.loads(args.payload)
    graph_generator = GraphGenerator()

    graph = graph_generator.build_graph(
        int(payload.get("n_nodes", 15)),
        graph_type=str(payload.get("graph_type", "wheel")),
        layout=str(payload.get("layout", "spring")),
        random_seed=int(payload.get("random_seed", 45)),
    )
    graph = graph_generator.alter_graph(
        graph,
        shift=tuple(payload.get("shift", (1.3, 1.3))),
        scale=tuple(payload.get("scale", (100.0, 100.0))),
        rotation_angle=float(payload.get("rotation_angle", 0.0)),
        noised_nodes_part=float(payload.get("noised_nodes_part", 0.5)),
        node_noise_strength=float(payload.get("node_noise_strength", 0.15)),
        convert_to_int=bool(payload.get("convert_to_int", True)),
        random_seed=int(payload.get("random_seed", 45)),
    )
    graph = graph_generator.init_edge_weights(
        graph,
        round_to=int(payload.get("round_to", 1)),
    )
    weight_matrix = graph_generator.build_edge_weight_matrix(
        graph,
        fill_empty_policy=str(payload.get("fill_empty_policy", "shortest")),
        make_int=bool(payload.get("make_int", True)),
    )

    print(json.dumps({"weight_matrix": weight_matrix}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
