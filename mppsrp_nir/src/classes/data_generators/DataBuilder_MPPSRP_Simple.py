
import numpy as np

class DataBuilder_MPPSRP_Simple():
    def __init__(self, weight_matrix,
                 distance_multiplier=1, travel_time_multiplier=60*60,
                 planning_horizon=7,
                 safety_level=0.05, max_level=0.95,
                 initial_inventory_level=0.5, tank_capacity=100,
                 depot_service_time = 15*60, station_service_time=10*60,
                 demand=10, products_count=5,
                 k_vehicles=1, compartments=[5 * [20]],
                 mean_vehicle_speed=60, vehicle_time_windows=[[9*60*60, 18*60*60]],
                 noise_initial_inventory=0.0, noise_tank_capacity=0.0,
                 noise_compartments=0.0, noise_demand=0.0,
                 noise_vehicle_time_windows=0.0,
                 noise_restrictions = 0.0,
                 random_seed=45):

        self.weight_matrix = weight_matrix
        self.distance_multiplier = distance_multiplier
        self.travel_time_multiplier = travel_time_multiplier
        self.planning_horizon = planning_horizon

        self.safety_level = safety_level
        self.max_level = max_level
        self.initial_inventory_level = initial_inventory_level
        self.tank_capacity = tank_capacity
        self.depot_service_time = depot_service_time
        self.station_service_time = station_service_time

        self.demand = demand
        self.products_count = products_count

        self.k_vehicles = k_vehicles
        self.mean_vehicle_speed = mean_vehicle_speed
        self.compartments = compartments
        self.vehicle_time_windows = vehicle_time_windows

        self.noise_initial_inventory = noise_initial_inventory
        self.noise_tank_capacity = noise_tank_capacity
        self.noise_compartments = noise_compartments
        self.noise_demand = noise_demand
        self.noise_vehicle_time_windows = noise_vehicle_time_windows
        self.noise_restrictions = noise_restrictions
        self.random_seed = random_seed

        pass

    def build_data_model(self):

        vehicle_compartments, vehicle_time_windows, vehicle_station_restrictions = self.build_vehicles_data_()
        distance_matrix, travel_time_matrix = self.build_base_matrices_()
        station_data = self.build_station_data_()
        service_times = self.build_service_times_()

        data_model = {}
        data_model["distance_matrix"] = distance_matrix
        data_model["travel_time_matrix"] = travel_time_matrix
        data_model["station_data"] = station_data
        data_model["k_vehicles"] = self.k_vehicles
        data_model["vehicle_compartments"] = vehicle_compartments
        data_model["vehicle_time_windows"] = vehicle_time_windows
        data_model["restriction_matrix"] = vehicle_station_restrictions
        data_model["service_times"] = service_times

        return data_model

    def build_service_times_(self):

        service_times = np.zeros( len(self.weight_matrix), dtype=np.int64 )
        service_times[0] = self.depot_service_time
        service_times[1:] = self.station_service_time

        return service_times

    def build_vehicles_data_(self):

        compartments = np.array( self.k_vehicles * self.compartments, dtype=np.float64)
        compartments_noise = self.noise_compartments * np.random.standard_normal(compartments.shape)
        compartments += compartments_noise * compartments
        compartments = compartments.astype( np.int64 )

        vehicle_time_windows = np.array( self.k_vehicles * self.vehicle_time_windows, dtype=np.float64)
        time_windows_noise = self.noise_vehicle_time_windows * np.random.standard_normal( vehicle_time_windows.shape )
        vehicle_time_windows += time_windows_noise * vehicle_time_windows
        vehicle_time_windows = vehicle_time_windows.astype(np.int64)

        vehicle_station_restrictions = self.k_vehicles * [[1 for i in range( len(self.weight_matrix) )]]
        vehicle_station_restrictions = np.array( vehicle_station_restrictions, dtype=np.int64 )
        for i in range( vehicle_station_restrictions.shape[0] ):
            for j in range( vehicle_station_restrictions.shape[1] ):
                if np.random.random() <= self.noise_restrictions:
                    vehicle_station_restrictions[i][j] = 0

        return compartments, vehicle_time_windows, vehicle_station_restrictions

    def build_station_data_(self):

        station_data = []
        n_stations = len( self.weight_matrix )
        for i in range( 1, n_stations ):
            for j in range( self.products_count ):
                station_data_row = [i, j,
                                    self.safety_level * self.tank_capacity,
                                    self.max_level * self.tank_capacity,
                                    self.initial_inventory_level * self.tank_capacity]
                station_data_row += self.planning_horizon * [self.demand]
                station_data.append( station_data_row )
        station_data = np.array( station_data, dtype=np.float64 )


        tank_capacities = np.zeros( len(station_data[:, 0]), dtype=np.float64 ) + self.tank_capacity
        tank_capacity_noise = self.noise_tank_capacity * np.random.standard_normal( len(station_data[:, 0]) )
        tank_capacities += tank_capacity_noise * tank_capacities
        station_data[:, 2] = self.safety_level * tank_capacities
        station_data[:, 3] = self.max_level * tank_capacities
        station_data[:, 4] = self.initial_inventory_level * tank_capacities

        demand_noise = self.noise_demand * np.random.standard_normal( station_data[:, 5:].shape )
        station_data[:, 5:] += demand_noise * station_data[:, 5:]

        station_data = np.array(station_data, dtype=np.int64)

        return station_data

    def build_base_matrices_(self):

        distance_matrix = np.array( self.weight_matrix )
        distance_matrix *= self.distance_multiplier

        travel_time_matrix = distance_matrix / self.mean_vehicle_speed
        travel_time_matrix *= self.travel_time_multiplier

        distance_matrix = distance_matrix.astype( dtype=np.int64 )
        travel_time_matrix = travel_time_matrix.astype(dtype=np.int64)

        return distance_matrix, travel_time_matrix