#随机创建拓扑
import random
import sys

def createRandomTopo(sNum):
	sys.setrecursionlimit(1000000)
	#create adjaMatrix
	topoMatrix = [[0 for i in range(sNum)] for i in range(sNum)]
	visited = [0 for i in range(sNum)]
	#create topo randomly
	for i in range(sNum):
		for j in range(i+1, sNum):
			link = random.randint(0,1)
			topoMatrix[i][j] = link
			topoMatrix[j][i] = link
	#DFS
	def DFS(v):
		visited[v] = 1
		for j in range(sNum):
			if topoMatrix[v][j] == 1 and visited[j] == 0:
				DFS(j)
	#check the network connectivity using DFS
	disconNode = []
	for i in range(sNum):
		if visited[i] == 0:
			DFS(i)
			disconNode.append(i)
	#if the network is unconnected, connect each disconNode
	for i in range(len(disconNode)-1):
		topoMatrix[disconNode[i]][disconNode[i+1]] = 1
		topoMatrix[disconNode[i+1]][disconNode[i]] = 1

	oddCount = calOddNum(topoMatrix, sNum)

	return topoMatrix, oddCount

def calOddNum(topoMatrix, sNum):
	count = 0
	for i in range(sNum):
		degreeSum = 0
		for j in range(sNum):
			degreeSum += topoMatrix[i][j]
		#这是奇数
		if degreeSum%2 == 1:
			count += 1
	return count

if __name__ == '__main__':
	sNum = 500
	topo, oddCount = createRandomTopo(sNum)
	print(topo)
	print(oddCount)
