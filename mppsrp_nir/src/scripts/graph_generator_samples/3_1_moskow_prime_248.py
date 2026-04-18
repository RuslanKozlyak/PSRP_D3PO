
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from src.classes.data_generators.GraphGenerator import GraphGenerator
from src.classes.paths_config import *
from src.classes.utils import *

graph_generator = GraphGenerator()

g_1 = graph_generator.build_graph( 8, graph_type="wheel", layout="kamada-kawai")
g_1 = graph_generator.alter_graph( g_1, shift=(15, 15), scale=(100.0, 100.0), rotation_angle=0, convert_to_int=True )
g_2 = graph_generator.build_graph( 16, graph_type="ring", layout="kamada-kawai")
g_2 = graph_generator.alter_graph( g_2, shift=(12.5, 12.5), scale=(120.0, 120.0), rotation_angle=0, convert_to_int=True )
composed_graph = graph_generator.compose_graphs( g_1, g_2 )

g_4 = graph_generator.build_graph( 32, graph_type="ring", layout="kamada-kawai")
g_4 = graph_generator.alter_graph( g_4, shift=(10, 10), scale=(150.0, 150.0), rotation_angle=0, convert_to_int=True )
composed_graph = graph_generator.compose_graphs( composed_graph, g_4 )

g_4 = graph_generator.build_graph( 5, graph_type="tree", layout="kamada-kawai")
g_4 = graph_generator.alter_graph( g_4, shift=(7.5, 7.5), scale=(200.0, 200.0), rotation_angle=90, convert_to_int=True )
composed_graph = graph_generator.compose_graphs( composed_graph, g_4, m_routes=5 )

g_4 = graph_generator.build_graph( 5, graph_type="tree", layout="kamada-kawai")
g_4 = graph_generator.alter_graph( g_4, shift=(7.5, 7.5), scale=(200.0, 200.0), rotation_angle=0, convert_to_int=True )
composed_graph = graph_generator.compose_graphs( composed_graph, g_4, m_routes=5 )

g_4 = graph_generator.build_graph( 5, graph_type="tree", layout="kamada-kawai")
g_4 = graph_generator.alter_graph( g_4, shift=(16.5, 16.5), scale=(100.0, 100.0), rotation_angle=0, convert_to_int=True )
composed_graph = graph_generator.compose_graphs( composed_graph, g_4 )

g_4 = graph_generator.build_graph( 5, graph_type="tree", layout="kamada-kawai")
g_4 = graph_generator.alter_graph( g_4, shift=(13.5, 13.5), scale=(100.0, 100.0), rotation_angle=0, convert_to_int=True )
composed_graph = graph_generator.compose_graphs( composed_graph, g_4 )

g_4 = graph_generator.build_graph( 5, graph_type="tree", layout="kamada-kawai")
g_4 = graph_generator.alter_graph( g_4, shift=(16.5, 13.5), scale=(100.0, 100.0), rotation_angle=0, convert_to_int=True )
composed_graph = graph_generator.compose_graphs( composed_graph, g_4 )

g_4 = graph_generator.build_graph( 5, graph_type="tree", layout="kamada-kawai")
g_4 = graph_generator.alter_graph( g_4, shift=(13.5, 16.5), scale=(100.0, 100.0), rotation_angle=0, convert_to_int=True )
composed_graph = graph_generator.compose_graphs( composed_graph, g_4 )

g_4 = graph_generator.build_graph( 5, graph_type="tree", layout="kamada-kawai")
g_4 = graph_generator.alter_graph( g_4, shift=(13.5, 16.5), scale=(100.0, 100.0), rotation_angle=0, convert_to_int=True )
composed_graph = graph_generator.compose_graphs( composed_graph, g_4 )

composed_graph = graph_generator.init_edge_weights( composed_graph, round_to=1 )
weight_matrix = graph_generator.build_edge_weight_matrix(composed_graph, fill_empty_policy="shortest")

dataset_name = "moskow_prime_248"
graph_generator.plot_graph( composed_graph, with_labels=True, with_edge_labels=False,
                            node_size=20, font_size=8)
#plt.show()
plt.savefig( Path( images_dir, "{}.jpg".format( dataset_name ) ), dpi=1000 )
graph_data = {}
graph_data["graph"] = composed_graph
graph_data["weight_matrix"] = weight_matrix
save( graph_data, Path( graphs_dir, "{}.pkl".format( dataset_name ) ))

print("done")