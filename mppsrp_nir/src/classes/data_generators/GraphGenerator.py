
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import numpy as np
import networkx as nx
from pprint import pprint

class GraphGenerator():
    def __init__(self):
        pass

    def build_graph(self, n, graph_type, layout="kamada-kawai", random_seed=45):

        g_init = None
        if graph_type == "ring": g_init = nx.cycle_graph( n )
        if graph_type == "wheel": g_init = nx.wheel_graph(n)
        if graph_type == "tree": g_init = nx.binomial_tree( n )
        if graph_type == "cube":
            g_init = nx.grid_2d_graph( n, n )
            g_init = self.convert_grid_ids_to_int_( g_init )

        if g_init is None: raise Exception( "Available graph types: ring, wheel, tree, cube" )

        node_positions = None
        if layout == "spring": node_positions = nx.drawing.spring_layout( g_init, seed=random_seed )
        if layout == "kamada-kawai": node_positions = nx.drawing.kamada_kawai_layout(g_init)
        if node_positions is None: raise Exception("Available layouts types: spring, kamada-kawai")

        g = nx.Graph()
        for node_id in node_positions:
            g.add_node( node_id, pos=(node_positions[node_id][0], node_positions[node_id][1]))

        edges_init = []
        for node_id in node_positions:
            current_edges = g_init.edges( node_id )
            for current_edge in current_edges:
                edges_init.append( current_edge )
        g.add_edges_from( edges_init )

        return g

    def convert_grid_ids_to_int_(self, grid_graph):
        int_ids_graph = nx.Graph()

        grid_graph_nodes = list(grid_graph.nodes)
        mapped_node_ids = {}
        for i, node_id in enumerate(grid_graph_nodes):
            mapped_node_ids[node_id] = i
        for node_id in grid_graph_nodes:
            new_node_id = mapped_node_ids[ node_id ]
            int_ids_graph.add_node( new_node_id )

        grid_graph_edges = list( grid_graph.edges.data() )
        for grid_edge in grid_graph_edges:
            new_edge = ( mapped_node_ids[grid_edge[0]], mapped_node_ids[grid_edge[1]] )
            int_ids_graph.add_edge( new_edge[0], new_edge[1] )

        return int_ids_graph


    def plot_graph(self, g, ax=None, with_edge_labels=False, **kwds):

        if ax is None:
            cf = plt.gcf()
        else:
            cf = ax.get_figure()
        cf.set_facecolor("w")
        if ax is None:
            if cf.axes:
                ax = cf.gca()
            else:
                ax = cf.add_axes((0, 0, 1, 1))

        if "with_labels" not in kwds:
            kwds["with_labels"] = "labels" in kwds


        pos = nx.get_node_attributes(g, 'pos')

        if with_edge_labels:
            labels = nx.get_edge_attributes(g, 'weight')
            nx.draw_networkx_edge_labels(g, pos, edge_labels=labels)

        nx.draw_networkx(g, pos=pos, ax=ax, **kwds)

    def alter_graph(self, g,
                    shift=(0.0, 0.0),
                    scale=(1.0, 1.0),
                    rotation_angle=0,
                    noised_nodes_part=0.0,
                    node_noise_strength=0.0,
                    convert_to_int=True,
                    random_seed=45):

        np.random.seed( random_seed )
        node_positions = nx.get_node_attributes( g, "pos" )
        old_x, old_y, new_x, new_y = [], [], [], []
        for node_id in node_positions:
            x = node_positions[node_id][0]
            y = node_positions[node_id][1]

            x = (x + shift[0]) * scale[0]
            y = (y + shift[1]) * scale[1]

            if noised_nodes_part > 0.0 and node_noise_strength > 0.0:
                if np.random.uniform() <= noised_nodes_part:
                    x = x + node_noise_strength * np.random.uniform() * x
                    y = y + node_noise_strength * np.random.uniform() * y

            if rotation_angle > 0:
                rotation_phi = (rotation_angle / 180) * np.pi
                x_rot = x * np.cos( rotation_phi ) - y * np.sin( rotation_phi )
                y_rot = x * np.sin( rotation_phi ) + y * np.cos( rotation_phi )
                old_x.append( x )
                old_y.append( y )
                new_x.append( x_rot )
                new_y.append( y_rot )
                x = x_rot
                y = y_rot

            if convert_to_int:
                x = round( x )
                y = round( y )

            node_positions[node_id] = (x, y)

        if rotation_angle > 0:
            mean_old_x = np.mean(old_x)
            mean_old_y = np.mean(old_y)
            mean_new_x = np.mean(new_x)
            mean_new_y = np.mean(new_y)

            for node_id in node_positions:
                x = node_positions[node_id][0]
                y = node_positions[node_id][1]
                x = x + (mean_old_x - mean_new_x)
                y = y + (mean_old_y - mean_new_y)
                if convert_to_int:
                    x = round(x)
                    y = round(y)
                node_positions[node_id] = (x, y)

        nx.set_node_attributes( g, node_positions, name="pos" )
        return g

    def compose_graphs(self, g_1, g_2, m_routes=1, k_neighbours=1,
                       routes_strategy="global", neighbours_strategy="nearest"):
        """
        Compose 2 graphs into one by building routes from g_1 nodes to g_2 nodes.
        :param g_1:
        :param g_2:
        :param m_routes:
        :param k_neighbours:
        :param routes_strategy: global strategy builds m_routes in total,
        local strategy builds m_routes for each node.
        :param neighbours_strategy: nearest, further, random
        :return:
        """

        g_1_nodes = nx.get_node_attributes(g_1, 'pos')
        g_2_nodes = nx.get_node_attributes(g_2, 'pos')
        g_1_edges = list( g_1.edges.data() )
        g_2_edges = list( g_2.edges.data() )

        # mapping g_2 nodes and edges to new node_ids
        max_g_1_node_id = max( list(g_1_nodes.keys()) )
        g_2_node_mapping = {}
        for i in g_2_nodes.keys():
            g_2_node_mapping[i] = max_g_1_node_id + 1 + i
        #inv_g_2_node_mapping = {v: k for k, v in g_2_node_mapping.items()}

        mapped_g_2_nodes = {}
        for i in g_2_nodes.keys():
            new_node_id = g_2_node_mapping[i]
            mapped_g_2_nodes[ new_node_id ] = g_2_nodes[ i ]

        mapped_g_2_edges = []
        for edge in g_2_edges:
            i, j = edge[0], edge[1]
            i, j = g_2_node_mapping[i], g_2_node_mapping[j]
            new_edge = ( i, j, edge[2] )
            mapped_g_2_edges.append( new_edge )

        # building connection edges between graphs
        graphs_connection_edges = None
        if neighbours_strategy == "nearest":
            pairwise_distances = self.calculate_pairwise_distances_( g_1_nodes, g_2_nodes )
            if routes_strategy == "local":
                graphs_connection_edges = self.build_connection_edges_for_each_node_( pairwise_distances, k_neighbours )
            elif routes_strategy == "global":
                graphs_connection_edges = self.build_global_connection_edges_(pairwise_distances, m_routes, k_neighbours)
            else:
                raise Exception("Available routing_strategy: local, global")
        else:
            raise Exception( "Available neighbours_strategy: nearest" )
        for i in range( len(graphs_connection_edges) ):
            node_1 = graphs_connection_edges[i][0]
            node_2 = graphs_connection_edges[i][1]
            node_2 = g_2_node_mapping[ node_2 ]
            graphs_connection_edges[i] = (node_1, node_2, {})

        # building composed graph
        composed_graph = nx.Graph()
        for node_id in g_1_nodes:
            composed_graph.add_node( node_id, pos=g_1_nodes[node_id] )
        for node_id in mapped_g_2_nodes:
            composed_graph.add_node( node_id, pos=mapped_g_2_nodes[node_id] )
        for edge in g_1_edges:
            composed_graph.add_edge( edge[0], edge[1] )
        for edge in mapped_g_2_edges:
            composed_graph.add_edge( edge[0], edge[1] )
        for edge in graphs_connection_edges:
            composed_graph.add_edge( edge[0], edge[1] )

        return composed_graph

    def build_connection_edges_for_each_node_(self, pairwise_distances, k):

        graphs_connection_edges = []
        for i in range( pairwise_distances.shape[0] ):
            current_distances = pairwise_distances[i]
            k_nearest_node_ids = np.argsort( current_distances )[:k]
            for j in range( k ):
                graphs_connection_edges.append( (i, k_nearest_node_ids[j]) )

        return graphs_connection_edges

    def build_global_connection_edges_(self, pairwise_distances, m, k):
        graphs_connection_edges = []

        mean_k_dists = np.zeros( (pairwise_distances.shape[0]), dtype=np.float64 )
        for i in range(pairwise_distances.shape[0]):
            current_distances = pairwise_distances[i]
            k_nearest_node_ids = np.argsort(current_distances)[:k]
            mean_k_nearest_distance = np.mean( current_distances[ k_nearest_node_ids ] )
            mean_k_dists[i] = mean_k_nearest_distance

        m_nearest_nodes_ids = np.argsort(mean_k_dists)[:m]
        for g_1_node_id in m_nearest_nodes_ids:
            current_distances = pairwise_distances[g_1_node_id]
            k_nearest_node_ids = np.argsort(current_distances)[:k]
            for j in range(k):
                graphs_connection_edges.append((g_1_node_id, k_nearest_node_ids[j]))

        return graphs_connection_edges


    def calculate_pairwise_distances_(self, g_1_nodes, g_2_nodes):
        g_1_nodes_coors = np.zeros((len(g_1_nodes), 2), dtype=np.float64)
        g_2_nodes_coors = np.zeros((len(g_2_nodes), 2), dtype=np.float64)
        pairwise_distances = np.zeros((g_1_nodes_coors.shape[0], g_2_nodes_coors.shape[0]), dtype=np.float64)

        for i, node_id in enumerate(g_1_nodes.keys()):
            current_node_coors = g_1_nodes[node_id]
            g_1_nodes_coors[i][0] = current_node_coors[0]
            g_1_nodes_coors[i][1] = current_node_coors[1]

        for i, node_id in enumerate(g_2_nodes.keys()):
            current_node_coors = g_2_nodes[node_id]
            g_2_nodes_coors[i][0] = current_node_coors[0]
            g_2_nodes_coors[i][1] = current_node_coors[1]

        for i in range(g_1_nodes_coors.shape[0]):
            for j in range(g_2_nodes_coors.shape[0]):
                a = g_1_nodes_coors[i]
                b = g_2_nodes_coors[j]
                pairwise_distances[i][j] = np.linalg.norm(a - b)

        return pairwise_distances

    def init_edge_weights(self, g, round_to = 3):
        """
        init by euclidean distance between nodes
        :param g:
        :return:
        """

        node_positions = nx.get_node_attributes(g, "pos")
        distances_attr = {}
        for node_i in node_positions:
            adj_edges = g.edges(node_i)
            ni_xy = node_positions[node_i]
            for edge in adj_edges:
                nj_xy = node_positions[edge[1]]
                distance = np.sqrt( (nj_xy[0] - ni_xy[0]) ** 2 + (nj_xy[1] - ni_xy[1]) ** 2)
                distance = round( distance, round_to )
                distances_attr[(edge[0], edge[1])] = {"weight": distance}
        nx.set_edge_attributes( g, distances_attr )

        return g

    def build_edge_weight_matrix(self, g, fill_empty_policy = "shortest", make_int = True):
        """
        Build, for example, distance or travel time matrix between nodes
        (depends on the sense of weights of edges).

        :param g:
        :return:
        """

        edge_weights = nx.get_edge_attributes(g, 'weight')
        n = len(g.nodes())
        weight_matrix = np.zeros( shape=(n, n), dtype=np.float64 )
        for edge in edge_weights:
            edge_weight = edge_weights[edge]
            weight_matrix[ edge[0], edge[1] ] = edge_weight
            weight_matrix[ edge[1], edge[0] ] = edge_weight

        for i in range( n ):
            for j in range( n ):
                if i == j: continue
                if weight_matrix[i][j] == 0.0:
                    weight_matrix[i][j] = np.nan

        if fill_empty_policy == "shortest":
            weight_matrix = self.fill_by_shortest_path_( g, weight_matrix )
        elif fill_empty_policy == "sum":
            weight_matrix = self.fill_by_sum_( weight_matrix )
        else:
            raise Exception("Available fill policies: shortest, sum")

        if make_int:
            weight_matrix = weight_matrix.astype( np.int64 )

        weight_matrix = weight_matrix.tolist()

        return weight_matrix

    def fill_by_shortest_path_(self, g, weight_matrix):

        n = len(g.nodes())
        for i in range(n):
            for j in range(n):
                if np.isnan( weight_matrix[i][j] ):
                    shortest_path_len = nx.shortest_path_length( g, source=i, target=j,
                                                             weight="weight", method="dijkstra" )
                    shortest_path_len = round( shortest_path_len )
                    weight_matrix[i][j] = shortest_path_len
                    weight_matrix[j][i] = shortest_path_len

        return weight_matrix

    def fill_by_sum_(self, weight_matrix):
        sum_weight = round( np.nansum( weight_matrix ), 1 )
        n = weight_matrix.shape[0]
        for i in range(n):
            for j in range(n):
                if np.isnan( weight_matrix[i][j] ):
                    weight_matrix[i][j] = sum_weight
                    weight_matrix[j][i] = sum_weight
        return weight_matrix
