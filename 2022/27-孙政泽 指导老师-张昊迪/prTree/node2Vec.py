"""
Reference implementation of node2vec. 

Author: Aditya Grover

Paper: node2vec: Scalable feature learning for networks
"""

import networkx as nx
import numpy as np
import random
from gensim.models import Word2Vec


class Graph:
    def __init__(self, nx_G, is_directed, p, q):
        self.G = nx_G
        self.is_directed = is_directed
        self.p = p
        self.q = q

    def node2vec_walk(self, walk_length, start_node):
        """
		Simulate a random walk starting from start node.
		"""
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = sorted(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0], alias_edges[(prev, cur)][1])]
                    walk.append(next)
            else:
                break

        return walk

    def simulate_walks(self, num_walks, walk_length):
        """
		Repeatedly simulate random walks from each node.
		"""
        G = self.G
        walks = []
        nodes = list(G.nodes())
        print("Walk iteration:")
        for walk_iter in range(num_walks):
            print(str(walk_iter + 1), "/", str(num_walks))
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

        return walks

    def get_alias_edge(self, src, dst):
        """
		Get the alias edge setup lists for a given edge.
		"""
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]["weight"] / p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]["weight"])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]["weight"] / q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        """
		Preprocessing of transition probabilities for guiding the random walks.
		"""
        G = self.G
        is_directed = self.is_directed

        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr]["weight"] for nbr in sorted(G.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}

        if is_directed:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        else:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return


def alias_setup(probs):
    """
	Compute utility lists for non-uniform sampling from discrete distributions.
	Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
	for details
	"""
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    """
	Draw sample from a non-uniform discrete distribution using alias sampling.
	"""
    K = len(J)

    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]


class Parameter:
    def __init__(self, d_model) -> None:
        # self.output = "qtree.emb"
        self.dimensions = d_model
        self.walk_length = 80
        self.num_walks = 10
        self.window_size = 10
        self.iter = 1
        self.workers = 8
        self.p = 1
        self.q = 1
        self.weighted = False
        self.unweighted = True
        self.directed = False
        self.undirected = True


def read_graph(id_edge_list, node2vec_args):
    """
	Reads the input network in networkx.
	"""
    if node2vec_args.weighted:
        # G = nx.read_edgelist(input_path, nodetype=int, data=(("weight", float),), create_using=nx.DiGraph())
        G = nx.DiGraph(id_edge_list)  # 这里的 id_edge_list 需要带有权重
    else:
        G = nx.DiGraph(id_edge_list)

        for edge in G.edges():
            G[edge[0]][edge[1]]["weight"] = 1

    if not node2vec_args.directed:
        G = G.to_undirected()

    return G


def learn_embeddings(walks, node2vec_args):
    """
	Learn embeddings by optimizing the Skipgram objective using SGD.
	"""
    walks = [[str(i) for i in walk] for walk in walks]
    model = Word2Vec(walks, vector_size=node2vec_args.dimensions, window=node2vec_args.window_size, min_count=0, sg=1, workers=node2vec_args.workers, epochs=node2vec_args.iter)
    # model.wv.save_word2vec_format(node2vec_args.output)

    # 获得训练好的词向量
    node_num = len(model.wv.key_to_index)

    all_vectors = model.wv[[str(i) for i in range(node_num)]]
    # print(all_vectors[0])
    # print(all_vectors[1])
    # print(all_vectors[-1])
    return all_vectors


def node2vec_embed(id_edge_list, d_model):
    """
	Pipeline for representational learning for all nodes in a graph.
	"""
    node2vec_args = Parameter(d_model)
    nx_G = read_graph(id_edge_list, node2vec_args)
    G = Graph(nx_G, node2vec_args.directed, node2vec_args.p, node2vec_args.q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(node2vec_args.num_walks, node2vec_args.walk_length)

    all_vectors = learn_embeddings(walks, node2vec_args)

    return all_vectors
