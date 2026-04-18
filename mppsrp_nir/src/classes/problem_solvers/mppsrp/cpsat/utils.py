
from ortools.sat.python.cp_model import UNKNOWN, MODEL_INVALID, FEASIBLE, INFEASIBLE, OPTIMAL

statuses_dict = {}
statuses_dict[ UNKNOWN ] = "UNKNOWN"
statuses_dict[ MODEL_INVALID ] = "MODEL_INVALID"
statuses_dict[ FEASIBLE ] = "FEASIBLE"
statuses_dict[ INFEASIBLE ] = "INFEASIBLE"
statuses_dict[ OPTIMAL ] = "OPTIMAL"