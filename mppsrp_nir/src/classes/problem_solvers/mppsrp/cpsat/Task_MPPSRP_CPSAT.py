
import numpy as np
from ortools.sat.python import cp_model

class Task_MPPSRP_CPSAT():
    def __init__(self, data_model, task_id):

        self.data_model = data_model
        self.task_id = task_id
        self.solver_name = "CP-SAT"
        self.model = cp_model.CpModel( )
        self.variables = {}
        self.other_parameters = {}
        self.station_data_dict = {}
        self.nodes_count = len(data_model["distance_matrix"])

        pass