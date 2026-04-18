
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from src.classes.data_generators.GraphGenerator import GraphGenerator
from src.classes.paths_config import *
from src.classes.utils import *

graph_generator = GraphGenerator()

g_1 = graph_generator.build_graph( 10, graph_type="wheel", layout="kamada-kawai")
g_1 = graph_generator.alter_graph( g_1, shift=(1.3, 1.3), scale=(100.0, 100.0), rotation_angle=0, convert_to_int=True )
g_2 = graph_generator.build_graph( 20, graph_type="ring", layout="kamada-kawai")
g_2 = graph_generator.alter_graph( g_2, shift=(1.0, 1.0), scale=(130.0, 130.0),
                                   noised_nodes_part=0.4,
                                   node_noise_strength=0.25,
                                   rotation_angle=0, convert_to_int=True )
composed_graph = graph_generator.compose_graphs( g_1, g_2, m_routes=1, k_neighbours=2, routes_strategy="local" )

g_3 = graph_generator.build_graph( 4, graph_type="cube", layout="kamada-kawai")
g_3 = graph_generator.alter_graph( g_3, shift=(4.0, 1.3), scale=(100.0, 100.0),
                                   noised_nodes_part=0.4,
                                   node_noise_strength=0.3,
                                   rotation_angle=45,
                                   convert_to_int=True,
                                   random_seed=45)
composed_graph = graph_generator.compose_graphs( composed_graph, g_3, m_routes=4 )

g_3 = graph_generator.build_graph( 5, graph_type="tree", layout="kamada-kawai")
g_3 = graph_generator.alter_graph( g_3, shift=(3.0, 3.0), scale=(100.0, 100.0),
                                   noised_nodes_part=0.2,
                                   node_noise_strength=0.10,
                                   rotation_angle=30,
                                   convert_to_int=True,
                                   random_seed=45)
composed_graph = graph_generator.compose_graphs( composed_graph, g_3, m_routes=4 )

composed_graph = graph_generator.init_edge_weights( composed_graph, round_to=1 )
weight_matrix = graph_generator.build_edge_weight_matrix(composed_graph, fill_empty_policy="shortest")


dataset_name = "mix_noised_78"
graph_generator.plot_graph( composed_graph, with_labels=True, with_edge_labels=True,
                            node_size=20, font_size=8)
#plt.show()
plt.savefig( Path( images_dir, "{}.jpg".format( dataset_name ) ), dpi=1000 )
graph_data = {}
graph_data["graph"] = composed_graph
graph_data["weight_matrix"] = weight_matrix
save( graph_data, Path( graphs_dir, "{}.pkl".format( dataset_name ) ))



print("done")