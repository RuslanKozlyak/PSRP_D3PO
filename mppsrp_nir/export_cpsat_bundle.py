from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np

from src.classes.data_generators.DataBuilder_MPPSRP_Simple import DataBuilder_MPPSRP_Simple


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True, help="JSON payload describing builder and solve options")
    args = parser.parse_args()

    payload = json.loads(args.payload)
    weight_matrix = np.asarray(payload["weight_matrix"], dtype=np.float64)
    builder_config = dict(payload.get("builder_config", {}))
    solve_requested = bool(payload.get("solve", False))
    solve_config = dict(payload.get("solve_config", {}))

    builder = DataBuilder_MPPSRP_Simple(weight_matrix=weight_matrix, **builder_config)
    data_model = builder.build_data_model()
    bundle: dict[str, Any] = {
        "data_model": _json_ready(data_model),
        "solve_requested": solve_requested,
        "solution_found": False,
        "kpis": None,
        "routes_schedule": None,
    }

    if solve_requested:
        from src.classes.problem_solvers.mppsrp.cpsat.TaskBuilder_MPPSRP_FullMILP_CPSAT import (
            TaskBuilder_MPPSRP_FullMILP_CPSAT,
        )
        from src.classes.problem_solvers.mppsrp.cpsat.TaskSolver_MPPSRP_CPSAT import (
            TaskSolver_MPPSRP_CPSAT,
        )

        task_builder = TaskBuilder_MPPSRP_FullMILP_CPSAT(
            max_trips_per_day=int(solve_config.get("max_trips_per_day", 2)),
            verbose=bool(solve_config.get("verbose", False)),
        )
        task = task_builder.build_task(data_model)
        task_solver = TaskSolver_MPPSRP_CPSAT(
            cache_dir=None,
            cache_all_feasible_solutions=False,
            solution_prefix=None,
            time_limit_milliseconds=solve_config.get("time_limit_milliseconds"),
        )
        solution = task_solver.solve(task)
        if solution is not None:
            bundle["solution_found"] = True
            bundle["kpis"] = _json_ready(solution.get_kpis())
            bundle["routes_schedule"] = _json_ready(_format_routes_schedule(solution.get_routes_schedule()))

    print(json.dumps(bundle))
    return 0


def _format_routes_schedule(routes_schedule: list[list[list[tuple[tuple[int, ...], dict[int, dict[int, int]]]]]]) -> list[Any]:
    serialized = []
    for day_schedule in routes_schedule:
        day_payload = []
        for vehicle_schedule in day_schedule:
            vehicle_payload = []
            for route, delivery_amount in vehicle_schedule:
                vehicle_payload.append(
                    {
                        "route": list(route),
                        "delivery_amount": {
                            str(node_id): {str(product_id): int(amount) for product_id, amount in product_dict.items()}
                            for node_id, product_dict in delivery_amount.items()
                        },
                    }
                )
            day_payload.append(vehicle_payload)
        serialized.append(day_payload)
    return serialized


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
