from celluloid import Camera
import networkx as nx
import numpy
from networkx import random_layout

SEPARATOR = '/'
DEFAULT_COLOR = 'blue'
MARKED_COLOR = 'green'
EXTENSION = '.gif'
FILENAME_TEMPLATE = "{storage}{algorithm}_animation_(...{source}){extension}"
FRAME_INTERVAL = 600
NODE_SIZE = 500


def get_filename(directory: str) -> str:

    return directory.rsplit(SEPARATOR)[-1]


def graph_builder(file_path: str) -> nx.Graph:

    graph = nx.Graph()
    #with open('../topo.txt', 'r') as file:
    with open('topo.txt', 'r') as file:
        for line in file.readlines():
            line = line.strip().split()
            u, v = int(line[0]), int(line[1])
            graph.add_edge(u, v)
    return graph


def create_gif(G: nx.Graph, camera: Camera, start: object, storage: str,
               func: str = 'dfs', source: str = '') -> bool:
    file_name = FILENAME_TEMPLATE.format(storage=storage, algorithm=func,
                                         source=get_filename(source),
                                         extension=EXTENSION)
    # nodes_color = [graph.nodes[node].get('color', DEFAULT_COLOR) for node in graph.nodes()]
    # nx.draw_planar(graph, with_labels=True, node_size=NODE_SIZE, node_color=nodes_color)
    
    # 把随机数种子确定下来，保证绘制的拓扑都长一样
    pos = random_layout(G, seed=numpy.random)

    # 绘制边的颜色
    edges_color = [G.edges[edge].get('color', DEFAULT_COLOR) for edge in G.edges()]
    nx.draw(G, pos, with_labels=True, node_size=NODE_SIZE, edge_color=edges_color)
    # 抓取快照
    camera.snap() 

    #with open('../path.txt', 'r') as f:
    with open('path.txt', 'r') as f:
        for path in f.readlines():
            path = path.strip().split()
            n = len(path)
            array = []
            # 每次读取路径的一条边就绘制一次
            for i in range(0, n, 2):
                u, v = int(path[i]), int(path[i+1])
                array.append([u, v])
                G[u][v]['color'] = 'red'

                edges_color = [G.edges[edge].get('color', DEFAULT_COLOR) for edge in G.edges()]
                nx.draw(G, pos, with_labels=True, node_size=NODE_SIZE, edge_color=edges_color)
                camera.snap()

            # 整个path读取完毕，颜色修正一次
            for u, v in array:
                G[u][v]['color'] = 'green'
            edges_color = [G.edges[edge].get('color', DEFAULT_COLOR) for edge in G.edges()]
            nx.draw(G, pos, with_labels=True, node_size=NODE_SIZE, edge_color=edges_color)
            camera.snap()

    animation = camera.animate(FRAME_INTERVAL)
    animation.save(file_name, writer='imagemagick')
    return True
