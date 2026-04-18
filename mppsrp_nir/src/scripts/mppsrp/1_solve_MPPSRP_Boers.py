
from pprint import pprint

from src.classes.utils import *
from src.classes.paths_config import *
from src.classes.data_generators.DataBuilder_MPPSRP_Boyers import DataBuilder_MPPSRP_Boyers
from src.classes.problem_solvers.mppsrp.cpsat.TaskBuilder_MPPSRP_FullMILP_CPSAT import *
from src.classes.problem_solvers.mppsrp.cpsat.TaskSolver_MPPSRP_CPSAT import *


data_builder = DataBuilder_MPPSRP_Boyers( k_vehicles=1 )
data_model = data_builder.build_data_model()

task_builder = TaskBuilder_MPPSRP_FullMILP_CPSAT( max_trips_per_day=10, verbose=True )
task = task_builder.build_task( data_model )

task_solver = TaskSolver_MPPSRP_CPSAT( cache_dir=interim_dir,
                                       cache_all_feasible_solutions=False,
                                       solution_prefix="boers",
                                       time_limit_milliseconds=30_000 )
solution = task_solver.solve( task )

kpi_dict = solution.get_kpis()
pprint( kpi_dict )
routes_schedule = solution.get_routes_schedule()
pprint( routes_schedule )

print()
routes_schedule = solution.get_routes_schedule()
pprint( routes_schedule )

print("done")