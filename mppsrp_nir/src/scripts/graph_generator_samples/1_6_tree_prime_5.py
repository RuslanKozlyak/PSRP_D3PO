
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from src.classes.data_generators.GraphGenerator import GraphGenerator
from src.classes.paths_config import *
from src.classes.utils import *

graph_generator = GraphGenerator()

g = graph_generator.build_graph( 5, graph_type="tree", layout="kamada-kawai")
g = graph_generator.alter_graph( g, shift=(1.0, 1.0), scale=(100.0, 100.0), rotation_angle=0, convert_to_int=True )

composed_graph = graph_generator.init_edge_weights( g, round_to=1 )
weight_matrix = graph_generator.build_edge_weight_matrix(composed_graph, fill_empty_policy="shortest")

dataset_name = "tree_prime_5"
graph_generator.plot_graph( composed_graph, with_labels=True, with_edge_labels=False,
                            node_size=20, font_size=8)
#plt.show()
plt.savefig( Path( images_dir, "{}.jpg".format( dataset_name ) ), dpi=1000 )
graph_data = {}
graph_data["graph"] = composed_graph
graph_data["weight_matrix"] = weight_matrix
save( graph_data, Path( graphs_dir, "{}.pkl".format( dataset_name ) ))

print("done")