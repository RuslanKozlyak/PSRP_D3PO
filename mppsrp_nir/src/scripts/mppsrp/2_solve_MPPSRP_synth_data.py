
from pprint import pprint
from pathlib import Path
from datetime import datetime
from src.classes.utils import *
from src.classes.paths_config import *

from src.classes.data_generators.DataBuilder_MPPSRP_Simple import *
from src.classes.problem_solvers.mppsrp.cpsat.TaskBuilder_MPPSRP_FullMILP_CPSAT import *
from src.classes.problem_solvers.mppsrp.cpsat.TaskSolver_MPPSRP_CPSAT import *

#dataset_name = "wheel_prime_10"
#dataset_name = "wheel_noised_10"
#dataset_name = "wheel_prime_21"
#dataset_name = "wheel_noised_21"
#dataset_name = "wheel_10_ring_20"
#dataset_name = "grid_noised_49"
dataset_name = "grid_noised_64"
#dataset_name = "tree_prime_4"
#dataset_name = "tree_prime_5"
#dataset_name = "tree_noised_5"
#dataset_name = "mix_prime_78"
#dataset_name = "mix_noised_78"
#dataset_name = "moskow_prime_248"
#dataset_name = "moskow_noised_248"
#dataset_name = "moskow_prime_380"
#dataset_name = "moskow_noised_380"
synth_data = load( Path( graphs_dir, "{}.pkl".format( dataset_name ) ) )

data_builder = DataBuilder_MPPSRP_Simple(synth_data["weight_matrix"],
                                         distance_multiplier=1, travel_time_multiplier=60*60,
                                         planning_horizon=7,
                                         safety_level=0.05, max_level=0.95,
                                         initial_inventory_level=0.5, tank_capacity=100,
                                         depot_service_time = 15*60, station_service_time=10*60,
                                         demand=10, products_count=5,
                                         k_vehicles=13, compartments=[5 * [50]],
                                         mean_vehicle_speed=60, vehicle_time_windows=[[9*60*60, 18*60*60]],
                                         noise_initial_inventory=0.2, noise_tank_capacity=0.2,
                                         noise_compartments=0.2, noise_demand=0.2,
                                         noise_vehicle_time_windows=0.0,
                                         noise_restrictions = 0.0,
                                         random_seed=45)
data_model = data_builder.build_data_model()

task_builder = TaskBuilder_MPPSRP_FullMILP_CPSAT( max_trips_per_day=2, verbose=True )
task = task_builder.build_task( data_model )

print("Dataset name: {}".format( dataset_name ))
print("Solving start time: {}".format( datetime.now() ))
task_solver = TaskSolver_MPPSRP_CPSAT( cache_dir=interim_dir,
                                       cache_all_feasible_solutions=False,
                                       solution_prefix=dataset_name,
                                       time_limit_milliseconds=120_000 )
solution = task_solver.solve( task )

print()
kpi_dict = solution.get_kpis()
pprint( kpi_dict )

print()
routes_schedule = solution.get_routes_schedule()
pprint( routes_schedule )

print("done")