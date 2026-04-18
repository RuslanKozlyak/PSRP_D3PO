
import numpy as np
from datetime import datetime

from ortools.sat.python import cp_model
from ortools.sat import cp_model_pb2

from src.classes.utils import *
from pathlib import Path

from src.classes.problem_solvers.mppsrp.cpsat.Solution_MPPSRP_CPSAT import Solution_MPPSRP_CPSAT
from src.classes.problem_solvers.mppsrp.cpsat.utils import statuses_dict

class TaskSolver_MPPSRP_CPSAT():
    def __init__(self, cache_dir=None,
                 cache_all_feasible_solutions=False,
                 solution_prefix=None,
                 time_limit_milliseconds = None):

        self.solver = cp_model.CpSolver()
        self.cache_dir = cache_dir
        self.solution_prefix = solution_prefix
        self.cache_all_feasible_solutions = cache_all_feasible_solutions
        self.time_limit_milliseconds = time_limit_milliseconds

        pass

    def solve(self, task):

        if self.time_limit_milliseconds is not None:
            self.solver.parameters.max_time_in_seconds = self.time_limit_milliseconds / 1000.0

        status = self.solver.Solve( task.model, self.SolutionLogger( task, self.cache_dir,
                                                                     self.cache_all_feasible_solutions,
                                                                     self.solution_prefix) )
        status = statuses_dict[ status ]

        solution = None
        if status in ["OPTIMAL", "FEASIBLE"]:
            solution = Solution_MPPSRP_CPSAT(task, self.solver)
        else:
            print("Status: {}".format( status ))

        return solution

    class SolutionLogger(cp_model.CpSolverSolutionCallback):
        """Display the objective value and time of intermediate solutions."""

        def __init__(self, task, cache_dir = None,
                     cache_all_feasible_solutions = True,
                     solution_prefix=None):
            cp_model.CpSolverSolutionCallback.__init__(self)

            self.task = task
            self.cache_dir = cache_dir
            self.solution_prefix = solution_prefix
            self.cache_all_feasible_solutions = cache_all_feasible_solutions

            self.solutions_count = 0
            self.best_objective = 10**9
            self.start_time = datetime.now()

        def on_solution_callback(self) -> None:
            """Called on each new solution."""
            current_time = datetime.now()
            objective_value = self.ObjectiveValue()
            print( "Solution: {} | Objective value: {} | "
                   "Solving time: {} | Datetime: {}".format( self.solutions_count,
                                                            objective_value,
                                                            current_time - self.start_time,
                                                            current_time) )
            if self.cache_dir is not None:
                solution_to_cache = Solution_MPPSRP_CPSAT(self.task, self)

                if self.solution_prefix is None:
                    self.solution_prefix = "solution"

                if self.cache_all_feasible_solutions:
                    save( solution_to_cache,
                          Path( self.cache_dir, self.solution_prefix + "_" + "{}.pkl".format(self.solutions_count) ),
                          verbose=False )

                if objective_value < self.best_objective:
                    self.best_objective = objective_value

                    save( solution_to_cache,
                          Path(self.cache_dir, self.solution_prefix + "_" + "best.pkl"),
                          verbose=False )

            self.solutions_count += 1

        def solution_count(self) -> int:

            return self.solutions_count