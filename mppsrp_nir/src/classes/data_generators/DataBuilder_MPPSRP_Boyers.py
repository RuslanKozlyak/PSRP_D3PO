
import numpy as np

class DataBuilder_MPPSRP_Boyers():

    """
    Builds data_model from verification data, presented by Luke Boyers in the
    "Optimisation methods for the multi-period petrol station replenishment problem" (2019)
    on page 99.
    """

    def __init__(self, k_vehicles=1):

        # Table 1: Travel distances (*1000m)
        self.travel_distances = [
            [0, 107, 39, 43, 82],
            [109, 0, 86, 89, 36],
            [39, 85, 0, 5, 48],
            [44, 88, 5, 0, 51],
            [83, 36, 48, 51, 0]
        ]

        # Table 2: Travel times (*300s)
        self.travel_times = [
            [0, 23, 11, 14, 18],
            [23, 0, 18, 19, 9],
            [11, 18, 0, 3, 11],
            [14, 19, 3, 0, 13],
            [18, 8, 12, 13, 0]
        ]

        # Table 3: Station data, with i = station number,
        # p = product index, levels are per 100L
        # i p Ls Lc Li Ld-t1 Ld-t2 Ld-t3 Ld-t4 Ld-t5 Ld-t6 Ld-t7
        self.station_data = [
            [1, 2, 15, 188, 131, 11, 11, 8, 8, 8, 8, 12],
            [1, 3, 28, 186, 141, 19, 18, 39, 47, 43, 42, 33],
            [2, 5, 30, 289, 213, 28, 26, 31, 31, 31, 35, 35],
            [2, 6, 68, 438, 216, 36, 40, 112, 114, 114, 111, 93],
            [3, 5, 35, 289, 202, 37, 37, 39, 37, 37, 42, 44],
            [3, 6, 66, 513, 386, 39, 43, 89, 86, 88, 91, 94],
            [3, 4, 15, 94, 52, 8, 8, 9, 9, 9, 10, 11],
            [4, 5, 28, 288, 174, 23, 23, 25, 28, 28, 29, 32],
            [4, 6, 36, 414, 215, 19, 22, 31, 34, 32, 31, 32],
            [4, 4, 15, 113, 51, 6, 6, 5, 5, 6, 6, 6]
        ]
        self.k_vehicles = k_vehicles
        # Table 4: Vehicle characteristics
        self.vehicle_compartments = k_vehicles * [[70, 50, 70, 70, 70, 50, 70, 70]] # *100L
        # 5:00 - 15:30 every day with Boers time scaling
        self.vehicle_time_windows = k_vehicles * [[int(5*60*60//300), int(15.5*60*60//300)]]
        # self.vehicle_time_windows = k_vehicles * [[0, int(24 * 60 * 60//300)]]
        self.vehicle_station_restrictions = k_vehicles * [[1, 1, 1, 1, 1]] # 1 - station available, 0 - restricted

        # service times and service rates (p.57)
        self.depot_service_time = int(15*60//300) # terminal service time is 15 minutes, Boers time scaling
        self.depot_service_rate = 1800 # litres per minute
        self.station_service_time = int(10*60//300) # 10 minutes, Boers time scaling
        self.station_service_rate = 900 # litres per minute
        self.service_times = []
        # in the Boyers study service time is fixed
        for i in range( len(self.travel_times[0]) ):
            if i == 0:
                self.service_times.append( self.depot_service_time )
            else:
                self.service_times.append( self.station_service_time )



        pass

    def build_data_model(self):

        data_model = {}
        data_model["distance_matrix"] = self.travel_distances
        data_model["travel_time_matrix"] = self.travel_times
        data_model["station_data"] = self.station_data
        data_model["k_vehicles"] = self.k_vehicles
        data_model["vehicle_compartments"] = self.vehicle_compartments
        data_model["vehicle_time_windows"] = self.vehicle_time_windows
        data_model["restriction_matrix"] = self.vehicle_station_restrictions
        data_model["service_times"] = self.service_times


        for key in data_model.keys():
            if key == "k_vehicles":
                continue
            data_model[ key ] = np.array( data_model[key], dtype=np.int64 )

        ########################
        # Boers some test cases
        #data_model["vehicle_compartments"] = 100 * data_model["vehicle_compartments"]
        #data_model["station_data"][:, 5:] = 0 * data_model["station_data"][:, 5:]
        #data_model["station_data"][:, 3] = 2 * data_model["station_data"][:, 3]
        #data_model["station_data"][:, 4] =  data_model["station_data"][:, 2]
        #######################

        #####
        # For compatibility with my service time
        #data_model["travel_time_matrix"] = 300 * data_model["travel_time_matrix"]
        #####

        return data_model