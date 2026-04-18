
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from src.classes.data_generators.GraphGenerator import GraphGenerator
from src.classes.paths_config import *
from src.classes.utils import *

graph_generator = GraphGenerator()

g_2 = graph_generator.build_graph( 10, graph_type="wheel", layout="kamada-kawai")
g_2 = graph_generator.alter_graph( g_2, shift=(1.3, 1.3), scale=(100.0, 100.0), rotation_angle=0, convert_to_int=True )

g_3 = graph_generator.build_graph( 20, graph_type="ring", layout="kamada-kawai")
g_3 = graph_generator.alter_graph( g_3, shift=(1.0, 1.0), scale=(130.0, 130.0), rotation_angle=0, convert_to_int=True )

g = graph_generator.compose_graphs( g_2, g_3, m_routes=4, k_neighbours=2,
                                    routes_strategy="global", neighbours_strategy="nearest" )
g = graph_generator.init_edge_weights( g, round_to = 1 )
weight_matrix = graph_generator.build_edge_weight_matrix(g, fill_empty_policy="shortest")


dataset_name = "wheel_10_ring_20"
graph_generator.plot_graph( g, with_labels=True, with_edge_labels=True )
#plt.show()
plt.savefig( Path( images_dir, "{}.jpg".format( dataset_name ) ), dpi=1000 )
graph_data = {}
graph_data["graph"] = g
graph_data["weight_matrix"] = weight_matrix
save( graph_data, Path( graphs_dir, "{}.pkl".format( dataset_name ) ))

print("done")