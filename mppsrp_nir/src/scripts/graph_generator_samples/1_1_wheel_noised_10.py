
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")
from src.classes.data_generators.GraphGenerator import GraphGenerator
from pprint import pprint
from src.classes.paths_config import *
from src.classes.utils import *

graph_generator = GraphGenerator()

g = graph_generator.build_graph( 10, graph_type="wheel", layout="kamada-kawai")
g = graph_generator.alter_graph( g, shift=(1.3, 1.3), scale=(100.0, 100.0), rotation_angle=0, convert_to_int=True,
                                 noised_nodes_part=0.8, node_noise_strength=0.2, random_seed=45)
g = graph_generator.init_edge_weights( g, round_to = 1 )
weight_matrix = graph_generator.build_edge_weight_matrix(g, fill_empty_policy ="shortest", make_int=True)


dataset_name = "wheel_noised_10"
graph_generator.plot_graph( g, with_labels=True, with_edge_labels=True )
#plt.show()
plt.savefig( Path( images_dir, "{}.jpg".format( dataset_name ) ), dpi=1000 )
graph_data = {}
graph_data["graph"] = g
graph_data["weight_matrix"] = weight_matrix
save( graph_data, Path( graphs_dir, "{}.pkl".format( dataset_name ) ))

print("done")