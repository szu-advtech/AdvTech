import random
import sys

import networkx as nx
import numpy as np
import copy
#无向图边数与顶点数的关系：0<= k <= n(n-1)/2
def Oddedge(NUM_NODES,edge_incre):
	select = []
	network_matrix=np.zeros([NUM_NODES, NUM_NODES])
	for j in range(NUM_NODES):
		for k in range(j + 1, NUM_NODES):
			if (network_matrix[j][k] == 0):
				select.append([j, k])
	for l in range(edge_incre):
		temp = select.pop(random.randint(0, len(select) - 1))
		network_matrix[temp[0]][temp[1]] = 1
		network_matrix[temp[1]][temp[0]] = 1

	return network_matrix

if __name__ == '__main__':
	for i in range(8):
		if(i<4*3/2):
			matrix = Oddedge(4,i)
			print(matrix)