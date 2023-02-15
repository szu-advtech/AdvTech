#!/usr/bin/python
import copy
import random
import sys
import numpy as np

def createRandomTopoWithFixedOdds(oddNum, maxSNum, step):
	sys.setrecursionlimit(1000000)
	sNum = 2*oddNum
	topoLists = []

	while sNum <= maxSNum:
		flag = 0
		while flag != 1:
			#create adjaMatrix
			topoMatrix = g_generator_edge(10000, sNum)

			for i in topoMatrix:
				oddCount = calOddNum(i, sNum)
				if oddCount == oddNum:
					topoLists.append(i)
					flag = 1
					continue
					# print topoMatrix
		sNum += step
	return topoMatrix
	#return topoLists

def calOddNum(topoMatrix, sNum):
	count = 0
	for i in range(sNum):
		degreeSum = 0
		for j in range(sNum):
			degreeSum += topoMatrix[i][j]
		if degreeSum%2 == 1:
			count += 1
	return count

def g_generator_edge(NUM_GRAPHS,NUM_NODES,edge_incre=1):
	op=[]
	NUM_GRAPHS=min(NUM_GRAPHS,int(NUM_NODES*(NUM_NODES-1)/2-NUM_NODES))
	for i in range(NUM_GRAPHS):
		if(i==0):
			network_matrix=np.zeros([NUM_NODES, NUM_NODES])
			for j in range(NUM_NODES-1):
				network_matrix[j][j+1] = 1
				network_matrix[j+1][j]=1
			network_matrix[0][NUM_NODES-1]=1
			network_matrix[NUM_NODES-1][0]=1
			op.append(network_matrix)
		else:
			network_matrix=copy.deepcopy(op[i-1])
			select=[]
			for j in range(NUM_NODES):
				for k in range(j+1,NUM_NODES):
					if(network_matrix[j][k]==0):
						select.append([j,k])
			for l in range(edge_incre):
				temp=select.pop(random.randint(0,len(select)-1))
				network_matrix[temp[0]][temp[1]]=1
				network_matrix[temp[1]][temp[0]]=1
			op.append(network_matrix)
	return op


if __name__ == '__main__':
	# sNum = 5
	# topo, oddCount = createRandomTopo(sNum)
	# print(topo)
	# print(oddCount)
	topoList = createRandomTopoWithFixedOdds(4, 10, 2) #each sNum has five graphs
	print(topoList)