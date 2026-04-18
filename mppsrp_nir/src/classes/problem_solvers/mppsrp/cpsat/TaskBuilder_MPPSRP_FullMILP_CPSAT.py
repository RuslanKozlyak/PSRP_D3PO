
import numpy as np
from ortools.sat.python import cp_model
from src.classes.problem_solvers.mppsrp.cpsat.Task_MPPSRP_CPSAT import Task_MPPSRP_CPSAT

class TaskBuilder_MPPSRP_FullMILP_CPSAT():
    def __init__(self, max_trips_per_day=10, verbose=True):
        """
        Builder for building an MP-PSRP task in the Boers Full MILP formulation (p.34).
        """

        self.max_trips_per_day = max_trips_per_day
        self.verbose = verbose

        pass

    def build_task(self, data_model, task_id=0):

        task = Task_MPPSRP_CPSAT( data_model=data_model,
                                  task_id=task_id )

        self.define_variables_( task )
        self.add_station_inventory_constraints_(task)
        self.add_vehicle_constraints_(task)
        self.add_route_constraints_(task)
        self.add_speed_up_constraints_(task)
        self.add_min_sum_distance_objective_( task )

        return task

    def define_variables_(self, task):

        # Boyers, p.22

        n_nodes = len(task.data_model["distance_matrix"])
        k_vehicles = task.data_model["k_vehicles"]

        task.other_parameters["planning_horizon"] = len( task.data_model["station_data"][0][5:] )
        planning_horizon = task.other_parameters["planning_horizon"]

        # trip ids
        task.other_parameters["max_trips_per_day"] = self.max_trips_per_day
        task.other_parameters["R"] = np.array( [i for i in range( self.max_trips_per_day )], dtype=np.int64 )

        # existing products in station data
        task.other_parameters["P"] = np.unique( task.data_model["station_data"][:, 1] )
        task.other_parameters["P_virtual_ids"] = {}
        task.other_parameters["virtual_p_to_true"] = {}
        products = task.other_parameters["P"]
        for i in range( len(products) ):
            task.other_parameters["P_virtual_ids"][ products[i] ] = i
            task.other_parameters["virtual_p_to_true"][i] = products[i]

        # contains information about station: station_id, real product_id, product levels
        task.station_data_dict = self.build_station_data_dict_( task )

        # 1 - if vehicle k drives from i to j in trip r during day t; 0 - otherwise
        task.variables["x"] = {}
        x = task.variables["x"]
        for t in range(planning_horizon):
            for k in range(k_vehicles):
                for r in range(self.max_trips_per_day):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            x[(t, k, r, i, j)] = task.model.NewIntVar(0, 1, "x_{}_{}_{}_{}_{}".format(t, k, r, i, j))


        # 1 - if vehicle k driving trip r during day t; 0 - otherwise
        task.variables["y"] = {}
        y = task.variables["y"]
        for k in range( k_vehicles ):
            for r in range( self.max_trips_per_day ):
                for t in range( planning_horizon ):
                    y[(k, r, t)] = task.model.NewIntVar( 0, 1, "y_{}_{}_{}".format( k, r, t ) )

        # 1 - if vehicle k visits station i during trip r on day t; 0 - otherwise
        task.variables["z"] = {}
        z = task.variables["z"]
        for k in range( k_vehicles ):
            for i in range(1, n_nodes ):
                for r in range( self.max_trips_per_day ):
                    for t in range( planning_horizon ):
                        z[(k, i, r, t)] = task.model.NewIntVar(0, 1, "z_{}_{}_{}_{}".format(k, i, r, t))

        # 1 - if product p is loaded into compartment m of vehicle k during trip r on day t
        task.variables["w"] = {}
        w = task.variables["w"]
        for p in range( len(products) ):
            for k in range( k_vehicles ):
                for m in range( len(task.data_model["vehicle_compartments"][k]) ):
                    for r in range( self.max_trips_per_day ):
                        for t in range( planning_horizon ):
                            w[(p, k, m, r, t)] = task.model.NewIntVar(0, 1, "w_{}_{}_{}_{}".format(p, k, m, r, t))

        # 1 - if trip s is driven after trip r by vehicle k on day t; 0 - otherwise
        task.variables["u"] = {}
        u = task.variables["u"]
        for t in range(planning_horizon):
            for k in range(k_vehicles):
                for r in range( self.max_trips_per_day - 1 ):
                    s = r + 1
                    u[(t, k, r, s)] = task.model.NewIntVar(0, 1, "u_{}_{}_{}_{}".format(t, k, r, s))

        # quantity of product p in compartment m of vehicle k delivered to location i during trip r on day t
        task.variables["q"] = {}
        q = task.variables["q"]
        for t in range( planning_horizon ):
            for k in range( k_vehicles ):
                for r in range(self.max_trips_per_day):
                    for i in range( n_nodes ):
                        for p in range( len(products) ):
                            for m in range( len(task.data_model["vehicle_compartments"][k]) ):
                                q[(t, k, r, i, p, m)] = task.model.NewIntVar(0,
                                                                             task.data_model["vehicle_compartments"][k][m],
                                                                             "q_{}_{}_{}_{}_{}_{}".format(t, k, r, i, p, m))

        # the time vehicle k can start the service at station i during trip r on day t
        vehicle_time_windows = task.data_model["vehicle_time_windows"]
        task.variables["S"] = {}
        S = task.variables["S"]
        for t in range(planning_horizon):
            for k in range( k_vehicles ):
                for r in range( self.max_trips_per_day ):
                    for i in range( n_nodes ):
                        working_start_time = vehicle_time_windows[k][0]
                        working_end_time = vehicle_time_windows[k][1]
                        S[(t, k, r, i)] = task.model.NewIntVar(working_start_time,
                                                               working_end_time,
                                                               "S_{}_{}_{}_{}".format(t, k, r, i))

        # inventory level of product p at station i at the end of day t
        max_inventory_level = np.max( task.data_model["station_data"][:, 3] )
        task.variables["I"] = {}
        I = task.variables["I"]
        for i in range(1, n_nodes):
            for p in range(len(products)):
                for t in range(planning_horizon):
                    I[(i, p, t)] = task.model.NewIntVar(0, 100 * max_inventory_level, "I_{}_{}_{}".format(i, p, t))
        # for constraint (3)
        for i in range(1, n_nodes):
            for p in range(len(products)):
                I[(i, p, -1)] = task.model.NewIntVar(0, 100 * max_inventory_level, "I_{}_{}_{}".format(i, p, -1))


        if self.verbose:
            print("Variables count:")
            total_count = 0
            for variable_name in task.variables.keys():
                current_count = len( task.variables[variable_name].keys() )
                print("{}: {}".format( variable_name, current_count) )
                total_count += current_count
            print("Total: {}".format( total_count ))

        return task

    def build_station_data_dict_(self, task):
        station_data = task.data_model["station_data"]
        station_data_dict = {}
        for i in range(len(station_data)):
            station_id = station_data[i][0]
            product_id = station_data[i][1]
            safety_level = station_data[i][2]
            tank_capacity = station_data[i][3]
            inventory_level = station_data[i][4]
            demands = station_data[i][5:]
            station_data_dict[(station_id, product_id)] = {
                "safety_level": safety_level,
                "tank_capacity": tank_capacity,
                "inventory_level": inventory_level,
                "demands": demands
            }

        return station_data_dict


    def add_min_sum_distance_objective_(self, task):

        n_nodes = len( task.data_model["distance_matrix"] )
        max_trips = len( task.other_parameters["R"] )
        planning_horizon = task.other_parameters["planning_horizon"]
        k_vehicles = task.data_model["k_vehicles"]

        objective_expression = [
            task.data_model["distance_matrix"][i][j] * task.variables["x"][t, k, r, i, j]
            for k in range( k_vehicles )
            for i in range( n_nodes )
            for j in range( n_nodes )
            for r in range( max_trips )
            for t in range( planning_horizon )
        ]

        task.model.Minimize(sum(objective_expression))

        return task

    def add_station_inventory_constraints_(self, task):
        # Boers, p.23, constraints (2)-(5)

        I = task.variables["I"]
        planning_horizon = task.other_parameters["planning_horizon"]
        station_data_dict = task.station_data_dict

        # constraint (2)
        for t in range( planning_horizon ):
            for station_data_key in station_data_dict.keys():
                i = station_data_key[0]
                p = station_data_key[1]
                safety_level_i = station_data_dict[ (i, p) ]["safety_level"]
                p_v = task.other_parameters["P_virtual_ids"][p]
                # Boers doesn't define L in >= L * safety_level_i
                task.model.Add( I[(i, p_v, t)] >= safety_level_i )

        # constraint (3)
        q = task.variables["q"]
        for t in range( planning_horizon ):
            for station_data_key in station_data_dict.keys():
                i = station_data_key[0]
                p = station_data_key[1]
                p_v = task.other_parameters["P_virtual_ids"][p]
                tank_capacity_i = station_data_dict[(i, p)]["tank_capacity"]
                delivered_q = [ q[(t, k, r, i, p_v, m)]
                                for k in range( task.data_model["k_vehicles"] )
                                for m in range( len(task.data_model["vehicle_compartments"][k]) )
                                for r in task.other_parameters["R"]
                            ]
                # Boers doesn't define L in <= L * tank_capacity_i
                task.model.Add( I[(i, p_v, t-1)] + sum( delivered_q ) <= tank_capacity_i )

        # constraint (4)
        for station_data_key in station_data_dict.keys():
            i = station_data_key[0]
            p = station_data_key[1]
            p_v = task.other_parameters["P_virtual_ids"][p]
            inventory_level_i_p = station_data_dict[(i, p)]["inventory_level"]
            # Boers doesn't define L in == L * inventory_level_i_p
            task.model.Add(I[(i, p_v, -1)] == inventory_level_i_p)

        # constraint (5)
        q = task.variables["q"]
        for t in range( planning_horizon ):
            for station_data_key in station_data_dict.keys():
                i = station_data_key[0]
                p = station_data_key[1]
                p_v = task.other_parameters["P_virtual_ids"][p]
                demand_i_p_t = station_data_dict[(i, p)]["demands"][t]
                delivered_q = [q[(t, k, r, i, p_v, m)]
                               for k in range(task.data_model["k_vehicles"])
                               for m in range(len(task.data_model["vehicle_compartments"][k]))
                               for r in task.other_parameters["R"]
                               ]
                # Boers doesn't define L in L * demand_i_p_t
                task.model.Add( I[(i, p_v, t)] == I[(i, p_v, t-1)] + sum( delivered_q ) - demand_i_p_t )

        return task

    def add_vehicle_constraints_(self, task):
        # Boers, p.23, constraints (6)-(12)

        q = task.variables["q"]
        x = task.variables["x"]
        w = task.variables["w"]
        z = task.variables["z"]

        k_vehicles = task.data_model["k_vehicles"]
        n_stations = len(task.data_model["distance_matrix"])
        restriction_matrix = task.data_model["restriction_matrix"]
        planning_horizon = task.other_parameters["planning_horizon"]
        trips = task.other_parameters["R"]
        products = task.other_parameters["P"]

        # constraint (6)
        for k in range( k_vehicles ):
            for i in range(1, n_stations ):
                for t in range( planning_horizon ):
                    for r in trips:
                        task.model.Add( z[(k, i, r, t)] <= restriction_matrix[k][i] )

        # constraint (7)
        for k in range( k_vehicles ):
            for t in range( planning_horizon ):
                for r in trips:
                    task.model.Add( sum(x[(t, k, r, 0, j)] for j in range( n_stations )) == 1 )

        # constraint (8)
        for k in range( k_vehicles ):
            for t in range( planning_horizon ):
                for r in trips:
                    for j in range( n_stations ):
                        in_flow = sum([x[(t, k, r, i, j)] for i in range(n_stations)])
                        out_flow = sum([x[(t, k, r, j, i)] for i in range(n_stations)])
                        task.model.Add(in_flow - out_flow == 0)

        # constraint (9)
        for k in range( k_vehicles ):
            for i in range(1, n_stations ):
                for t in range( planning_horizon ):
                    for r in trips:
                        x_link_to_z = [ x[(t, k, r, i, j)] for j in range( n_stations ) if i != j ]
                        task.model.Add( z[(k, i, r, t)] == sum( x_link_to_z ))

        # constraint (10)
        for i in range(1, n_stations ):
            for k in range( k_vehicles ):
                for m in range( len(task.data_model["vehicle_compartments"][k]) ):
                    for t in range( planning_horizon ):
                        for r in task.other_parameters["R"]:
                            for p in range( len(products) ):
                                # Boers doesn't specify L in <= L * z[(k, i, r, t)]
                                compartment_capacity = task.data_model["vehicle_compartments"][k][m]
                                task.model.Add(q[(t, k, r, i, p, m)] <= compartment_capacity * z[(k, i, r, t)])

        # constraint (11)
        for k in range( k_vehicles ):
            for m in range( len(task.data_model["vehicle_compartments"][k]) ):
                for t in range( planning_horizon ):
                    for r in task.other_parameters["R"]:
                        for p in range( len(products) ):
                            delivered_amount = [q[(t, k, r, i, p, m)] for i in range(1, n_stations )]
                            compartment_capacity = task.data_model["vehicle_compartments"][k][m]
                            task.model.Add(sum( delivered_amount ) <= w[(p, k, m, r, t)] * compartment_capacity)

        # constraint (12)
        for k in range( k_vehicles ):
            for m in range( len(task.data_model["vehicle_compartments"][k]) ):
                for t in range( planning_horizon ):
                    for r in task.other_parameters["R"]:
                        w_expression = [ w[(p, k, m, r, t)] for p in range( len(products) ) ]
                        task.model.Add( sum(w_expression) <= 1)

        return task

    def add_route_constraints_(self, task):

        # Boers, p.23, constraints (13)-(18)

        S = task.variables["S"]
        x = task.variables["x"]
        y = task.variables["y"]
        u = task.variables["u"]
        z = task.variables["z"]

        k_vehicles = task.data_model["k_vehicles"]
        n_stations = len(task.data_model["distance_matrix"])
        planning_horizon = task.other_parameters["planning_horizon"]
        vehicle_time_windows = task.data_model["vehicle_time_windows"]
        travel_time_matrix = task.data_model["travel_time_matrix"]
        service_times = task.data_model["service_times"]

        # constraints (13)-(14)
        # realized into S definition

        # constraint (15)
        sum_travel_time = np.sum( travel_time_matrix )
        for k in range(k_vehicles):
            for r in task.other_parameters["R"]:
                for t in range( planning_horizon ):
                    for i in range(1, n_stations ):
                        for j in range(1, n_stations ):
                            # Boers doesn't define L in L * (1 - x[(t, k, r, i, j)])
                            task.model.Add( S[(t, k, r, i)] + travel_time_matrix[i][j] +
                                            service_times[i] - sum_travel_time * (1 - x[(t, k, r, i, j)]) <= S[(t, k, r, j)])

        # constraint (16)
        sum_travel_time = np.sum(travel_time_matrix)
        for t in range(planning_horizon):
            for k in range(k_vehicles):
                for r in range(len(task.other_parameters["R"]) - 1):
                    s = r + 1
                    # Boers doesn't define L in L * (1 - u[(t, k, r, s)])
                    task.model.Add( S[(t, k, s, 0)] + sum_travel_time * (1 - u[(t, k, r, s)]) >= S[(t, k, r, 0)] )

        # !!!!! My custom constraint !!!!!!!
        # Vehicle can't start a trip at the last second of a working day
        # Vehicle should end a trip before a work day end
        """
        sum_travel_time = np.sum(travel_time_matrix)
        for k in range(k_vehicles):
            for r in task.other_parameters["R"]:
                for t in range(planning_horizon):
                    for i in range(1, n_stations):
                        for j in range(1, n_stations):
                            task.model.Add( S[(t, k, r, i)] + travel_time_matrix[i][j] + travel_time_matrix[j][i] +
                                            service_times[i] - sum_travel_time * (1 - x[(t, k, r, i, j)])
                                            <= vehicle_time_windows[k][1])"""

        # constraint (17)
        for k in range(k_vehicles):
            for t in range(planning_horizon):
                left_part = [ u[(t, k, r, r+1)]
                              for r in range(len(task.other_parameters["R"]) - 1)]
                right_part = [ y[(k, r, t)] for r in range(len(task.other_parameters["R"]))]
                task.model.Add( sum( left_part ) >= sum( right_part ) - 1 )

        # constraint (18)
        for k in range(k_vehicles):
            for t in range(planning_horizon):
                for r in task.other_parameters["R"]:
                    for i in range(1, n_stations ):
                        task.model.Add( z[(k, i, r, t)] <= y[(k, r, t)] )

        return task

    def add_speed_up_constraints_(self, task):
        # Boers, p.25, constraints (27)-(29)

        q = task.variables["q"]
        x = task.variables["x"]
        y = task.variables["y"]
        w = task.variables["w"]

        k_vehicles = task.data_model["k_vehicles"]
        n_stations = len(task.data_model["distance_matrix"])
        planning_horizon = task.other_parameters["planning_horizon"]
        trips = task.other_parameters["R"]
        products = task.other_parameters["P"]

        # constraint (27)
        for t in range( planning_horizon ):
            for k in range( k_vehicles ):
                for r in trips:
                    task.model.Add( sum([x[t, k, r, i, 0] for i in range(n_stations)]) == 1 )

        # constraint (28)
        for t in range( planning_horizon ):
            for k in range( k_vehicles ):
                for r in trips:
                    for i in range(n_stations):
                        for j in range(n_stations):
                            task.model.Add( x[t, k, r, i, j] <= y[k, r, t] )

        # constraint (29)
        for t in range( planning_horizon ):
            for k in range( k_vehicles ):
                for r in trips:
                    for m in range(len(task.data_model["vehicle_compartments"][k])):
                        for p in range(len(products)):
                            for i in range(n_stations):
                                # Boers doesn't define L in <= L * w[(p, k, m, r, t)]
                                compartment_size = task.data_model["vehicle_compartments"][k][m]
                                task.model.Add(q[(t, k, r, i, p, m)] <= compartment_size * w[(p, k, m, r, t)])

        return task