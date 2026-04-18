
import numpy as np
from tqdm import tqdm

class Solution_MPPSRP_CPSAT():
    def __init__(self, task, solver):

        self.n_nodes = len(task.data_model["distance_matrix"])
        self.k_vehicles = task.data_model["k_vehicles"]
        self.vehicle_compartments = task.data_model["vehicle_compartments"]
        self.max_trips_per_day = task.other_parameters["max_trips_per_day"]
        self.planning_horizon = task.other_parameters["planning_horizon"]
        self.products = task.other_parameters["P"]
        self.safety_levels = task.data_model["station_data"][:, 2]
        self.capacities = task.data_model["station_data"][:, 3]
        self.station_data_dict = task.station_data_dict

        self.task_id = task.task_id
        self.nodes_count = task.nodes_count
        self.solution_values = self.convert_solution_to_dict_( task, solver )
        self.solution_values = self.postprocess_variables_(self.solution_values)
        self.kpi_dict = self.build_kpi_dict_( task )

        pass

    def get_routes_schedule(self):
        routes_schedule = []

        for t in range( self.planning_horizon ):
            routes_schedule.append( [] )
            for k in range( self.k_vehicles ):
                routes_schedule[t].append( [] )

        x = self.solution_values["x"]
        y = self.solution_values["y"]
        q = self.solution_values["q"]
        products = self.products
        for t in range( self.planning_horizon ):
            for k in range( self.k_vehicles ):
                for r in range( self.max_trips_per_day ):
                    if y[k, r, t] == 1:
                        current_route = []
                        delivery_amount = {}
                        depot_id = 0
                        current_node_id = depot_id
                        current_route.append( depot_id )
                        current_node_id = np.argmax(x[t, k, r, current_node_id, :])
                        while current_node_id != depot_id:
                            current_route.append( current_node_id )
                            delivery_amount[ current_node_id ] = {}

                            for p in range(len(products)):
                                true_product_id = products[p]
                                if true_product_id not in delivery_amount[current_node_id].keys():
                                    delivery_amount[current_node_id][true_product_id] = 0

                                for m in range(len(self.vehicle_compartments[k])):
                                    delivery_amount[current_node_id][true_product_id] += q[t, k, r, current_node_id, p, m]

                            current_node_id = np.argmax(x[t, k, r, current_node_id, :])

                        current_route.append( depot_id )
                        current_route = tuple( current_route )
                        routes_schedule[t][k].append( (current_route, delivery_amount) )

        return routes_schedule

    def get_kpis(self):
        return self.kpi_dict

    def build_kpi_dict_(self, task):

        kpi_dict = {}

        kpi_dict["total_travel_distance"] = 0
        kpi_dict["total_travel_time"] = 0
        x = self.solution_values["x"]
        for t in range( self.planning_horizon ):
            for k in range( self.k_vehicles ):
                for r in range( self.max_trips_per_day ):
                    for i in range( self.n_nodes ):
                        for j in range( self.n_nodes ):
                            kpi_dict["total_travel_distance"] += task.data_model["distance_matrix"][i][j] * x[t, k, r, i, j]
                            kpi_dict["total_travel_time"] += task.data_model["travel_time_matrix"][i][j] * x[t, k, r, i, j]

        kpi_dict["average_stock_levels"] = np.zeros( (self.planning_horizon, ), dtype=np.int64 )
        I = self.solution_values["I"]
        for p in range( len(self.products) ):
            for t in range( self.planning_horizon ):
                current_levels = I[:, p, t+1]
                # if product type is absent on a station, then it is always zero. Don't count them.
                avg_nonzero_levels = np.sum( current_levels[ current_levels != 0 ] )
                kpi_dict["average_stock_levels"][t] = avg_nonzero_levels

        kpi_dict["average_stock_levels_percent"] = []
        I = self.solution_values["I"]
        for t in range(self.planning_horizon):
            for station_data_key in self.station_data_dict.keys():
                i = station_data_key[0]
                p = station_data_key[1]
                p_v = task.other_parameters["P_virtual_ids"][p]
                tank_capacity_i = self.station_data_dict[(i, p)]["tank_capacity"]
                current_level = I[i-1, p_v, t+1]
                kpi_dict["average_stock_levels_percent"].append( current_level / tank_capacity_i )
        kpi_dict["average_stock_levels_percent"] = 100 * np.mean( kpi_dict["average_stock_levels_percent"] )

        kpi_dict["dry_runs"] = 0
        for t in range( self.planning_horizon ):
            for station_data_key in self.station_data_dict.keys():
                i = station_data_key[0]
                p = station_data_key[1]
                safety_level_i = self.station_data_dict[ (i, p) ]["safety_level"]
                p_v = task.other_parameters["P_virtual_ids"][p]
                if self.solution_values["I"][i-1, p_v, t] <= safety_level_i:
                    kpi_dict["dry_runs"] += 1

        kpi_dict["average_vehicle_utilization"] = 0
        delivered_amount = 0
        vehicle_capacities = 0
        q = self.solution_values["q"]
        y = self.solution_values["y"]
        for t in range( self.planning_horizon ):
            for k in range( self.k_vehicles ):
                for r in range( self.max_trips_per_day ):
                    if y[k, r, t] == 1:
                        vehicle_capacities += np.sum( task.data_model["vehicle_compartments"][k] )
                        delivered_amount += np.sum( q[t, k, r, :, :, :] )
        kpi_dict["average_vehicle_utilization"] = round( 100 * (delivered_amount / vehicle_capacities), 1 )

        kpi_dict["average_stops_per_trip"] = 0
        y = self.solution_values["y"]
        z = self.solution_values["z"]
        trips_count = 0
        stops_count = 0
        for t in range( self.planning_horizon ):
            for k in range( self.k_vehicles ):
                for r in range( self.max_trips_per_day ):
                    trips_count += y[k, r, t]
                    for i in range( self.n_nodes - 1 ):
                        stops_count += z[k, i, r, t]
        kpi_dict["average_stops_per_trip"] = stops_count / trips_count


        return kpi_dict

    def postprocess_variables_(self, solution_values):
        """
        Mathematical model is correct, but it always leaves ones into x[0][0] that
        leads to always leave ones into all y (staying at the depot looks like a trip with null distance).
        We need y to build routes. Useful trips are only between depot and stations. So clean all
        ones into x[0][0] and fix y values (leave ones in trips between depot and stations).
        :param solution_values:
        :return:
        """

        x = self.solution_values["x"]
        y = self.solution_values["y"]
        for t in range( self.planning_horizon ):
            for k in range( self.k_vehicles ):
                for r in range( self.max_trips_per_day ):
                    x[t, k, r, 0, 0] = 0

                    # if vehicle stays at a depot - set trip indicator to 0
                    sum_x = np.sum( x[t, k, r, :, :] )
                    if sum_x == 0:
                        y[k, r, t] = 0

        return solution_values

    def convert_solution_to_dict_(self, task, solver):
        """
        Extracts solution values from the solver.
        :param task:
        :param solver:
        :return:
        """

        solution_dict = {}

        n_nodes = self.n_nodes
        k_vehicles = self.k_vehicles
        max_trips_per_day = self.max_trips_per_day
        planning_horizon = self.planning_horizon
        products = self.products

        # 1 - if vehicle k drives from i to j in trip r during day t; 0 - otherwise
        solution_dict["x"] = np.zeros(shape=(planning_horizon, k_vehicles, max_trips_per_day,
                                             n_nodes, n_nodes), dtype=np.int64)
        x = task.variables["x"]
        for t in range(planning_horizon):
            for k in range(k_vehicles):
                for r in range(max_trips_per_day):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            solution_dict["x"][t, k, r, i, j] = solver.Value( x[(t, k, r, i, j)] )

        # 1 - if vehicle k driving trip r during day t; 0 - otherwise
        solution_dict["y"] = np.zeros(shape=(k_vehicles, max_trips_per_day, planning_horizon), dtype=np.int64)
        y = task.variables["y"]
        for k in range(k_vehicles):
            for r in range(max_trips_per_day):
                for t in range(planning_horizon):
                    solution_dict["y"][k, r, t] = solver.Value( y[(k, r, t)] )

        # 1 - if vehicle k visits station i during trip r on day t; 0 - otherwise
        solution_dict["z"] = np.zeros(shape=(k_vehicles, n_nodes-1, max_trips_per_day, planning_horizon), dtype=np.int64)
        z = task.variables["z"]
        for k in range( k_vehicles ):
            for i in range( 1, n_nodes ):
                for r in range( max_trips_per_day ):
                    for t in range( planning_horizon ):
                        solution_dict["z"][k, i-1, r, t] = solver.Value( z[(k, i, r, t)] )

        # 1 - if product p is loaded into compartment m of vehicle k during trip r on day t
        solution_dict["w"] = np.zeros(shape=(len(products), k_vehicles,
                                             len(task.data_model["vehicle_compartments"][0]),
                                             max_trips_per_day, planning_horizon), dtype=np.int64)
        w = task.variables["w"]
        for p in range( len(products) ):
            for k in range( k_vehicles ):
                for m in range( len(task.data_model["vehicle_compartments"][k]) ):
                    for r in range( max_trips_per_day ):
                        for t in range( planning_horizon ):
                            solution_dict["w"][p, k, m, r, t] = solver.Value( w[(p, k, m, r, t)] )

        # 1 - if trip s is driven after trip r by vehicle k on day t; 0 - otherwise
        solution_dict["u"] = np.zeros(shape=(planning_horizon, k_vehicles,
                                             max_trips_per_day-1, max_trips_per_day-1), dtype=np.int64)
        u = task.variables["u"]
        for t in range(planning_horizon):
            for k in range(k_vehicles):
                for r in range( max_trips_per_day - 1 ):
                    s = r + 1
                    solution_dict["u"][t, k, r, s-1] = solver.Value( u[(t, k, r, s)] )

        # quantity of product p in compartment m of vehicle k delivered to location i during trip r on day t
        solution_dict["q"] = np.zeros(shape=(planning_horizon, k_vehicles, max_trips_per_day,
                                             n_nodes, len(products), len(task.data_model["vehicle_compartments"][0])),
                                      dtype=np.int64)
        q = task.variables["q"]
        for t in range( planning_horizon ):
            for k in range( k_vehicles ):
                for r in range( max_trips_per_day ):
                    for i in range( n_nodes ):
                        for p in range( len(products) ):
                            for m in range(len(task.data_model["vehicle_compartments"][k])):
                                solution_dict["q"][t, k, r, i, p, m] = solver.Value( q[(t, k, r, i, p, m)] )

        # the time vehicle k can start the service at station i during trip r on day t
        solution_dict["S"] = np.zeros(shape=(planning_horizon, k_vehicles, max_trips_per_day, n_nodes), dtype=np.int64)
        S = task.variables["S"]
        for t in range( planning_horizon ):
            for k in range( k_vehicles ):
                for r in range( max_trips_per_day ):
                    for i in range( n_nodes ):
                        solution_dict["S"][t, k, r, i] = solver.Value( S[(t, k, r, i)] )

        # inventory level of product p at station i at the end of day t
        solution_dict["I"] = np.zeros(shape=(n_nodes-1, len(products), planning_horizon+1), dtype=np.int64)
        I = task.variables["I"]
        for i in range( 1, n_nodes ):
            for p in range( len(products) ):
                for t in range( planning_horizon+1 ):
                    solution_dict["I"][i-1, p, t] = solver.Value( I[(i, p, t-1)] )


        return solution_dict