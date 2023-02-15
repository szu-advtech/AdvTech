#!/usr/bin/python
import random
import sys
import copy
import time
import randomTopo
import FixedOddTopo
import  OddEdge
import specialTopo
def DFLSPathPlan(topoMatrix, sNum):
	sys.setrecursionlimit(1000000)#设置递归调用的深度
	linkState = copy.deepcopy(topoMatrix)
	pathCount = [0]
	#DFLS
	def DFLS(v, isNewPath):
		#计算出非重叠路径的数目
		if isNewPath == 1:
			pathCount[0] += 1
			isNewPath = 0

		for j in range(sNum):
			if linkState[v][j] == 1:
				linkState[v][j] = 0
				linkState[j][v] = 0
				isNewPath = DFLS(j, isNewPath)
		return 1
	DFLS(0,1)
	return pathCount[0]
'''
	sNum = 0
	#修改这里，只需要传入拓扑图即可
	topoMatrix,oddCount= randomTopo.createRandomTopo(sNum)
	#论文中拓扑结构进行验证
	#topoMatrix=([[0, 1, 0, 0, 0], [1, 0, 1, 1, 0], [0, 1, 0, 1, 1], [0, 1, 1, 0, 1], [0, 0, 1, 1, 0]])
	print(topoMatrix)
	pathNum = DFLSPathPlan(topoMatrix, sNum)
	print("DFS计算出的非重叠路径有：",pathNum,"条")
'''
'''
	#第二个图采集
	for i in range(40,150):
		topoMatrix=FixedOddTopo.FixedoddTopo(i,20)
		pathNum = DFLSPathPlan(topoMatrix, i)
		print(pathNum)
'''
'''
	#第三个图测试750组数据
	for i in range(750):
		if(i<750*749/2):
			#第一哥参数是顶点
			topoMatrix = OddEdge.Oddedge(100,i)
			pathNum = DFLSPathPlan(topoMatrix, 100)
			print(pathNum)

'''
'''
	#第六个图采集 固定10个奇数点
	for i in range(20,100):
		start = time.clock()
		topoMatrix=FixedOddTopo.FixedoddTopo(i,10)
		pathNum = DFLSPathPlan(topoMatrix, i)
		end = time.clock()
		print(end-start)
'''
'''
	#最后一部分工作 Fattree  sNum必须是大于10  5的倍数
	for i in range(2,20):
		topoMatrix = specialTopo.genFatTree(i*5)
		#print(topoMatrix)
		pathNum = DFLSPathPlan(topoMatrix,i*5)
		print(pathNum)
'''
''''
	#最后一部分工作 LeafSpine  sNum必须是大于3  3的倍数
	for i in range(1,34):
		topoMatrix = specialTopo.genSpineLeaf(i*3)
		#print(topoMatrix)
		pathNum = DFLSPathPlan(topoMatrix,i*3)
		print(pathNum)

'''
if __name__ == '__main__':
	#最后一部分工作 LeafSpine  sNum必须是大于3  3的倍数
	for i in range(1,34):
		topoMatrix = specialTopo.genSpineLeaf(i*3)
		#print(topoMatrix)
		pathNum = DFLSPathPlan(topoMatrix,i*3)
		print(pathNum)
