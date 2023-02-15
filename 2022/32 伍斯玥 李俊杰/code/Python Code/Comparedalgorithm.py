import random
import networkx as nx

def forestfire(G, size):
    G1 = nx.Graph()
    list_nodes = list(G.nodes())
    # print(len(G))
    dictt = set()
    random_node = random.sample(list_nodes, 1)[0]
    # print(random_node)
    q = set()  # q = set contains the distinct values
    q.add(random_node)
    while (len(G1.nodes()) < size):
        if (len(q) > 0):
            initial_node = q.pop()
            if (initial_node not in dictt):
                # print(initial_node)
                dictt.add(initial_node)
                neighbours = list(G.neighbors(initial_node))
                # print(list(G.neighbors(initial_node)))
                np = random.randint(1, len(neighbours))
                # print(np)
                # print(neighbours[:np])
                for x in neighbours[:np]:
                    if (len(G1.nodes()) < size):
                        G1.add_edge(initial_node, x)
                        q.add(x)
                    else:
                        break
            else:
                continue
        else:
            random_node = random.sample(list(list_nodes) and list(dictt), 1)[0]
            q.add(random_node)
    q.clear()
    return G1

class Queue():
    # Constructor creates a list
    def __init__(self):
        self.queue = list()

    # Adding elements to queue
    def enqueue(self, data):
        # Checking to avoid duplicate entry (not mandatory)
        if data not in self.queue:
            self.queue.insert(0, data)
            return True
        return False

    # Removing the last element from the queue
    def dequeue(self):
        if len(self.queue) > 0:
            return self.queue.pop()
        else:
            # plt.show()
            exit()

    # Getting the size of the queue
    def size(self):
        return len(self.queue)

    # printing the elements of the queue
    def printQueue(self):
        return self.queue

def snowball( G, size, k):
    G1 = nx.Graph()
    q = Queue()
    list_nodes = list(G.nodes())
    m = k
    dictt = set()
    while(m):
        id = random.sample(list(G.nodes()), 1)[0]
        q.enqueue(id)
        m = m - 1
    # print(q.printQueue())
    while(len(G1.nodes()) <= size):
        if(q.size() > 0):
            id = q.dequeue()
            G1.add_node(id)
            if(id not in dictt):
                dictt.add(id)
                list_neighbors = list(G.neighbors(id))
                if(len(list_neighbors) > k):
                    for x in list_neighbors[:k]:
                        q.enqueue(x)
                        G1.add_edge(id, x)
                elif(len(list_neighbors) <= k and len(list_neighbors) > 0):
                    for x in list_neighbors:
                        q.enqueue(x)
                        G1.add_edge(id, x)
            else:
                continue
        else:
            initial_nodes = random.sample(list(G.nodes()) and list(dictt), k)
            no_of_nodes = len(initial_nodes)
            for id in initial_nodes:
                q.enqueue(id)
    return G1

def random_walk_sampling_simple(complete_graph, nodes_to_sample,growth_size = 2,T = 100):
    complete_graph = nx.convert_node_labels_to_integers(complete_graph, 0, 'default', True)
    # giving unique id to every node same as built-in function id
    for n, data in complete_graph.nodes(data=True):
        complete_graph.nodes[n]['id'] = n

    nr_nodes = len(complete_graph.nodes())
    upper_bound_nr_nodes_to_sample = nodes_to_sample
    index_of_first_random_node = random.randint(0, nr_nodes - 1)
    sampled_graph = nx.Graph()

    sampled_graph.add_node(complete_graph.nodes[index_of_first_random_node]['id'])

    iteration = 1
    edges_before_t_iter = 0
    curr_node = index_of_first_random_node
    while sampled_graph.number_of_nodes() != upper_bound_nr_nodes_to_sample and iteration<1000:
        edges = [n for n in complete_graph.neighbors(curr_node)]
        index_of_edge = random.randint(0, len(edges) - 1)
        chosen_node = edges[index_of_edge]
        sampled_graph.add_node(chosen_node)
        sampled_graph.add_edge(curr_node, chosen_node)
        curr_node = chosen_node
        iteration = iteration + 1

        if iteration % T == 0:
            if ((sampled_graph.number_of_edges() - edges_before_t_iter) < growth_size):
                curr_node = random.randint(0, nr_nodes - 1)
            edges_before_t_iter = sampled_graph.number_of_edges()
    return sampled_graph