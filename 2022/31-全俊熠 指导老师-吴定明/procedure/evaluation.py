import numpy as np
from littleballoffur import RandomWalkSampler, SnowBallSampler, ForestFireSampler, BreadthFirstSearchSampler
from collections import defaultdict
import networkx as nx
import igraph as ig
import sys


def seed_generation(nodes_num):
    return np.random.choice(nodes_num)


def l1_distance(sub_property, original_property):
    return np.abs(np.sum(sub_property - original_property)) / np.sum(original_property)


def l1_distance_for_distribution(sub_property, original_property):
    keys = set(sub_property.keys()) | set(original_property.keys())
    dist = 0
    norm = sum(list(original_property.values()))
    for key in keys:
        if key not in sub_property.keys():
            sub_property[key] = 0
        if key not in original_property.keys():
            original_property[key] = 0
        dist += np.fabs(original_property[key] - sub_property[key])
    return float(dist) / norm


def node_number_distance(subgraph, original_graph):
    original_graph_nodes_number = nx.number_of_nodes(original_graph)
    subgraph_nodes_number = nx.number_of_nodes(subgraph)
    return l1_distance(subgraph_nodes_number, original_graph_nodes_number)


def average_degree(original_graph):
    degree = dict(nx.degree(original_graph))
    num = original_graph.number_of_nodes()
    return sum(degree.values()) / num


def degree_distribution(original_graph):
    nodes_num = len(original_graph.nodes)
    d_distribution = defaultdict(float)
    degree_sum = 0
    for node_degree in original_graph.degree:
        d_distribution[node_degree[1]] += 1
        degree_sum += node_degree[1]
    for degree in d_distribution.keys():
        d_distribution[degree] = d_distribution[degree] / nodes_num
    return degree_sum / nodes_num, d_distribution


def degree_distribution_distance(subgraph, original_graph):
    original_ad, original_dd = degree_distribution(original_graph)
    sub_ad, sub_dd = degree_distribution(subgraph)
    return l1_distance(sub_ad, original_ad), l1_distance_for_distribution(sub_dd, original_dd)


def neighbor_connectivity(original_graph):
    degree_nodes = defaultdict(list)
    for node_v in original_graph.nodes:
        degree_v = original_graph.degree()[node_v]
        degree_nodes[degree_v].append(node_v)

    knn = defaultdict(float)
    for degree in degree_nodes:
        if degree * len(degree_nodes[degree]) == 0:
            continue
        for node_v in degree_nodes[degree]:
            for node_w in list(original_graph.neighbors(node_v)):
                knn[degree] += original_graph.degree()[node_w]
        knn[degree] = float(knn[degree]) / (degree * len(degree_nodes[degree]))

    return knn


def neighbor_connectivity_distance(subgraph, original_graph):
    original_neighbor_connectivity = neighbor_connectivity(original_graph)
    sub_neighbor_connectivity = neighbor_connectivity(subgraph)
    return l1_distance_for_distribution(sub_neighbor_connectivity, original_neighbor_connectivity)


def network_clustering_coefficient_distance(subgraph, original_graph):
    original_clustering_coefficient = nx.average_clustering(original_graph)
    sub_clustering_coefficient = nx.average_clustering(subgraph)
    return l1_distance(sub_clustering_coefficient, original_clustering_coefficient)


def degree_dependent_clustering_coefficient(original_graph):
    degree_num = defaultdict(int)
    lcc_degree_sum = defaultdict(float)
    nodes = original_graph.nodes
    node_degree = dict(original_graph.degree)
    nodes_num = len(nodes)
    sum_lcc = 0

    for node in nodes:
        degree = node_degree[node]
        degree_num[degree] += 1
        if degree == 0 or degree == 1:
            continue
        lcc = 0
        node_adj_list = list(original_graph[node].keys())
        for i in range(0, degree - 1):
            x = node_adj_list[i]
            for j in range(i + 1, degree):
                y = node_adj_list[j]
                if node != x and x != y and y != node:
                    x_adj_list = list(original_graph[x].keys())
                    lcc += 2 * x_adj_list.count(y)

        lcc = float(lcc) / (degree * (degree - 1))
        lcc_degree_sum[degree] += lcc
        sum_lcc += lcc

    ddcc = defaultdict(float)
    for degree in degree_num:
        if degree_num[degree] > 0:
            ddcc[degree] = float(lcc_degree_sum[degree]) / degree_num[degree]
    return sum_lcc / nodes_num, ddcc


def degree_dependent_clustering_coefficient_distance(subgraph, original_graph):
    original_acc, original_ddcc = degree_dependent_clustering_coefficient(original_graph)
    sub_acc, sub_ddcc = degree_dependent_clustering_coefficient(subgraph)
    return l1_distance(sub_acc, original_acc), l1_distance_for_distribution(sub_ddcc, original_ddcc)


def common_neighbor_distribution(original_graph):
    cnd = defaultdict(float)

    for node_i in original_graph.nodes:
        node_i_adj_list = list(original_graph[node_i].keys())
        for node_j in node_i_adj_list:
            if node_j <= node_i:
                continue
            m = 0
            for node_k in node_i_adj_list:
                if node_k == node_i and node_k == node_j:
                    continue
                node_j_adj_list = list(original_graph[node_j].keys())
                m += node_j_adj_list.count(node_k)
            cnd[m] += 1

    norm = sum(list(cnd.values()))
    for m in cnd:
        cnd[m] = float(cnd[m]) / norm

    return cnd


def common_neighbor_distribution_distance(subgraph, original_graph):
    original_cnd = common_neighbor_distribution(original_graph)
    sub_cnd = common_neighbor_distribution(subgraph)
    return l1_distance_for_distribution(sub_cnd, original_cnd)


def calc_shortest_path_properties(original_graph):
    original_edges = nx.to_pandas_edgelist(original_graph).values
    ig_original_graph = ig.Graph(original_edges)
    ig_original_path_length_hist = ig_original_graph.path_length_hist(directed=False)
    spld = defaultdict(float)
    num_all = ig_original_path_length_hist.n
    bins = tuple(ig_original_path_length_hist.bins())

    for (i, j, k) in bins:
        if j != i + 1:
            print("Error.")
            exit(0)
        spld[i] = float(k) / num_all

    diameter = max(list(dict(spld).keys()))
    apl = sum([l * spld[l] for l in spld])
    return spld, diameter, apl


def shortest_path_length_distance(subgraph, original_graph):
    original_spld, original_diameter, original_apl = calc_shortest_path_properties(original_graph)
    sub_spld, sub_diameter, sub_apl = calc_shortest_path_properties(subgraph)
    spld_distance = l1_distance_for_distribution(sub_spld, original_spld)
    diameter_distance = l1_distance(sub_diameter, original_diameter)
    apl_distance = l1_distance(sub_apl, original_apl)
    return spld_distance, diameter_distance, apl_distance


def degree_dependent_betweeness_centrality(original_graph):
    original_edges = nx.to_pandas_edgelist(original_graph).values
    ig_original_graph = ig.Graph(original_edges)
    degrees = ig_original_graph.degree(list(range(0, len(original_graph.nodes))))
    bc = ig_original_graph.betweenness(directed=False)
    n = int(ig_original_graph.vcount())

    ddbc = defaultdict(float)
    V_d = defaultdict(int)
    for i in range(0, len(degrees)):
        d = degrees[i]
        ddbc[d] += float(bc[i]) / ((n - 1) * (n - 2))
        V_d[d] += 1

    for d in ddbc:
        ddbc[d] = float(ddbc[d]) / V_d[d]

    return ddbc


def betweeness_centrality_distance(subgraph, original_graph):
    original_ddbc = degree_dependent_betweeness_centrality(original_graph)
    sub_ddbc = degree_dependent_betweeness_centrality(subgraph)
    return l1_distance_for_distribution(sub_ddbc, original_ddbc)


def largest_eigenvalue(original_graph):
    original_edges = nx.to_pandas_edgelist(original_graph).values
    ig_original_graph = ig.Graph(original_edges)
    L = ig_original_graph.laplacian(normalized=True)
    eigenvalues = np.linalg.eigvals(L)
    return float(max(eigenvalues))


def largest_eigenvalue_distance(subgraph, original_graph):
    original_L = largest_eigenvalue(original_graph)
    sub_L = largest_eigenvalue(subgraph)
    return l1_distance(sub_L, original_L)


def subgraph_modify(subgraph, original_graph):
    subgraph = nx.Graph(subgraph)
    subgraph_nodes_list = list(subgraph.nodes)
    for node in subgraph_nodes_list:
        node_neighbors_list = list(nx.all_neighbors(original_graph, node))
        for neighbor in node_neighbors_list:
            subgraph.add_node(neighbor)
            subgraph.add_edge(node, neighbor)
    return subgraph


def breadth_first_search(original_graph, start_node):
    breadth_first_search_sampler = BreadthFirstSearchSampler(
        number_of_nodes=int(original_graph.number_of_nodes() * 0.1))
    subgraph = breadth_first_search_sampler.sample(original_graph, start_node=start_node)
    subgraph = subgraph_modify(subgraph, original_graph)

    print("Normalized L1 distance of each property of breadth first search")

    n = node_number_distance(subgraph, original_graph)
    print("Number of nodes:", n)

    ad, dd = degree_distribution_distance(subgraph, original_graph)
    print("Average degree:", ad)
    print("Degree distribution:", dd)

    nc = neighbor_connectivity_distance(subgraph, original_graph)
    print("Neighbor connectivity:", nc)

    acc, ddcc = degree_dependent_clustering_coefficient_distance(subgraph, original_graph)
    print("Average local clustering coefficient:", acc)
    print("Degree-dependent clustering coefficient:", ddcc)

    cnd = common_neighbor_distribution_distance(subgraph, original_graph)
    print("Common neighbor distribution:", cnd)

    spld, diamter, apl = shortest_path_length_distance(subgraph, original_graph)
    print("Average shortest path length:", apl)
    print("Shortest path length distribution:", spld)
    print("Diameter:", diamter)

    bc = betweeness_centrality_distance(subgraph, original_graph)
    print("Degree-dependent betweenness centrality:", bc)

    # le = largest_eigenvalue_distance(subgraph, original_graph)
    # print("Largest eigenvalue:", le)
    print()


def snowball_sampling(original_graph, start_node):
    snow_ball_sampler = SnowBallSampler(number_of_nodes=original_graph.number_of_nodes() * 0.1, k=50)
    subgraph = snow_ball_sampler.sample(original_graph, start_node=start_node)
    subgraph = subgraph_modify(subgraph, original_graph)

    print("Normalized L1 distance of each property of snowball sampling")

    n = node_number_distance(subgraph, original_graph)
    print("Number of nodes:", n)

    ad, dd = degree_distribution_distance(subgraph, original_graph)
    print("Average degree:", ad)
    print("Degree distribution:", dd)

    nc = neighbor_connectivity_distance(subgraph, original_graph)
    print("Neighbor connectivity:", nc)

    acc, ddcc = degree_dependent_clustering_coefficient_distance(subgraph, original_graph)
    print("Average local clustering coefficient:", acc)
    print("Degree-dependent clustering coefficient:", ddcc)

    cnd = common_neighbor_distribution_distance(subgraph, original_graph)
    print("Common neighbor distribution:", cnd)

    spld, diamter, apl = shortest_path_length_distance(subgraph, original_graph)
    print("Average shortest path length:", apl)
    print("Shortest path length distribution:", spld)
    print("Diameter:", diamter)

    bc = betweeness_centrality_distance(subgraph, original_graph)
    print("Degree-dependent betweenness centrality:", bc)

    # le = largest_eigenvalue_distance(subgraph, original_graph)
    # print("Largest eigenvalue:", le)
    print()


def forest_fire_sampling(original_graph):
    forest_fire_sampler = ForestFireSampler(number_of_nodes=original_graph.number_of_nodes() * 0.1, p=0.7)
    subgraph = forest_fire_sampler.sample(original_graph)
    subgraph = subgraph_modify(subgraph, original_graph)

    print("Normalized L1 distance of each property of forest fire sampling")

    n = node_number_distance(subgraph, original_graph)
    print("Number of nodes:", n)

    ad, dd = degree_distribution_distance(subgraph, original_graph)
    print("Average degree:", ad)
    print("Degree distribution:", dd)

    nc = neighbor_connectivity_distance(subgraph, original_graph)
    print("Neighbor connectivity:", nc)

    acc, ddcc = degree_dependent_clustering_coefficient_distance(subgraph, original_graph)
    print("Average local clustering coefficient:", acc)
    print("Degree-dependent clustering coefficient:", ddcc)

    cnd = common_neighbor_distribution_distance(subgraph, original_graph)
    print("Common neighbor distribution:", cnd)

    spld, diamter, apl = shortest_path_length_distance(subgraph, original_graph)
    print("Average shortest path length:", apl)
    print("Shortest path length distribution:", spld)
    print("Diameter:", diamter)

    bc = betweeness_centrality_distance(subgraph, original_graph)
    print("Degree-dependent betweenness centrality:", bc)

    # le = largest_eigenvalue_distance(subgraph, original_graph)
    # print("Largest eigenvalue:", le)
    print()


def random_walk_sampling(original_graph, start_node):
    random_walk_sampler = RandomWalkSampler(number_of_nodes=original_graph.number_of_nodes() * 0.1)
    subgraph = random_walk_sampler.sample(original_graph, start_node=start_node)
    subgraph = subgraph_modify(subgraph, original_graph)

    print("Normalized L1 distance of each property of random walk sampling")

    n = node_number_distance(subgraph, original_graph)
    print("Number of nodes:", n)

    ad, dd = degree_distribution_distance(subgraph, original_graph)
    print("Average degree:", ad)
    print("Degree distribution:", dd)

    nc = neighbor_connectivity_distance(subgraph, original_graph)
    print("Neighbor connectivity:", nc)

    acc, ddcc = degree_dependent_clustering_coefficient_distance(subgraph, original_graph)
    print("Average local clustering coefficient:", acc)
    print("Degree-dependent clustering coefficient:", ddcc)

    cnd = common_neighbor_distribution_distance(subgraph, original_graph)
    print("Common neighbor distribution:", cnd)

    spld, diamter, apl = shortest_path_length_distance(subgraph, original_graph)
    print("Average shortest path length:", apl)
    print("Shortest path length distribution:", spld)
    print("Diameter:", diamter)

    bc = betweeness_centrality_distance(subgraph, original_graph)
    print("Degree-dependent betweenness centrality:", bc)

    # le = largest_eigenvalue_distance(subgraph, original_graph)
    # print("Largest eigenvalue:", le)
    print()


if __name__ == '__main__':
    file_name = "../data/syn10000.txt"
    graph = nx.read_edgelist(file_name, create_using=nx.Graph(), nodetype=int)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    start_node = seed_generation(graph.number_of_nodes())
    breadth_first_search(graph, start_node)
    snowball_sampling(graph, start_node)
    forest_fire_sampling(graph)
    random_walk_sampling(graph, start_node)
