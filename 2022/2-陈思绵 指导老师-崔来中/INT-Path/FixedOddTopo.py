import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
#顶点数  奇数顶点数
def FixedoddTopo(NUM_NODES,NUM_ODD_DEGREE):
    while(True):
        WS = nx.random_graphs.connected_watts_strogatz_graph(NUM_NODES,2,NUM_ODD_DEGREE/NUM_NODES)
    # spring layout
        count = 0
        for node in range(NUM_NODES):
            if WS.degree(node)%2!=0:
                count+=1
        if count==NUM_ODD_DEGREE:

            network_matrix = np.zeros([NUM_NODES,NUM_NODES])

            for edge in WS.edges():
                # print(edge)
                network_matrix[edge[0]][edge[1]] = 1
                network_matrix[edge[1]][edge[0]] = 1

            #print(network_matrix)
            #显示图
            #nx.draw(WS)
           # plt.show()
            break
    return network_matrix
if __name__ == '__main__':
    for i in range (0,250):
        if(i%2==0):
            network_matrix=FixedoddTopo(500,i)
            print(network_matrix)