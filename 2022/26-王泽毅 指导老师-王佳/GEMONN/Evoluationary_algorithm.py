import torch
import torch.nn as nn
import numpy as np
import cupy as cp
import random

# 初始化种群 #
def Initialization_Population(PopulationSize, Dimension, HiddenNum):
    Dimension += 1
    Dimension = int(Dimension)
    Population = (cp.random.random((PopulationSize, Dimension * HiddenNum)) - 0.5) * 2 * ((6 / cp.power((Dimension + HiddenNum), 1 / 2)))
    for i in range(PopulationSize):
        Population[i] = Population[i] * (cp.random.rand(Dimension * HiddenNum, ) < ((i + 1) / PopulationSize))
    Boundary = cp.hstack((cp.tile([[10], [-10]], [1, (Dimension - 1) * HiddenNum]), cp.tile([[20], [-20]], [1, HiddenNum])))
    Coding = 'Real'
    return Population, Boundary, Coding

# 生成初始种群 #
def Evaluation(Population, Dimension, HiddenNum, Data, Model):
    pop_size = Population.shape[0]  # 种群数量
    Weight_Grad = cp.zeros(Population.shape)
    FunctionValue = cp.zeros((pop_size, 2))
    FunctionValue[:, 0] = cp.sum(Population != 0, axis=1) / ((Dimension + 1) * HiddenNum)  # 计算稀疏性（存入0）
    FunctionValue = cp.asnumpy(FunctionValue)
    data_iter = iter(Data)
    images, labels = data_iter.next()
    images = images.view(-1, Dimension).cuda()
    criterion = nn.MSELoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(Model.parameters(), lr=learning_rate)
    for i in range(pop_size):
        weights = torch.Tensor(cp.reshape(Population[i, :], (Dimension + 1, HiddenNum))).t()
        weights.requires_grad_(requires_grad=True)
        Model.Initialization(weights.cuda())
        outputs = Model(images)
        loss = criterion(outputs[0], images)
        FunctionValue[i, 1] = loss.cpu().detach().numpy()  # 计算loss（存入1）
        optimizer.zero_grad()
        loss.backward()
        Weight_Grad_Temp = cp.reshape(Model.fc1.weight.grad.t().cpu().numpy(), (Dimension * HiddenNum,))
        Weight_Grad_T_ = cp.reshape(Model.fc2.weight.grad.cpu().numpy(), (Dimension * HiddenNum,))
        Weight_Grad[i, :] = cp.hstack(
            (Coperate_Weight_Grad(Weight_Grad_Temp, Weight_Grad_T_), Model.fc1.bias.grad.cpu().numpy()))
        optimizer.step()
    return FunctionValue, Weight_Grad

# 交配池 #
def Mating(Population, FrontValue, CrowdDistance):
    N, D = Population.shape
    MatingPool = cp.zeros((N, D))
    Rank = np.random.permutation(N)
    Pointer = 0
    index = []
    for i in range(0, N, 2):
        k = [0, 0]
        for j in range(2):
            if Pointer >= N:
                Rank = np.random.permutation(N)
                Pointer = 0
            p = Rank[Pointer]
            q = Rank[Pointer + 1]
            if FrontValue[0, p] < FrontValue[0, q]:
                k[j] = p
            elif FrontValue[0, p] > FrontValue[0, q]:
                k[j] = q
            elif CrowdDistance[0, p] > CrowdDistance[0, q]:
                k[j] = p
            else:
                k[j] = q
            Pointer += 2
        MatingPool[i:i + 2, :] = Population[k[0:2], :]
        index = np.hstack((index, k[0:2]))
    return cp.array(MatingPool), np.int64(index)

# gSBX算子 #
def gSBX(MatingPool, Pop_Gradient, Boundary, Coding, MaxOffspring):
    N, D = MatingPool.shape
    if MaxOffspring < 1 or MaxOffspring > N:
        MaxOffspring = N
    if Coding == "Real":
        ProC = 1
        ProM = 1 / D
        DisC = 20
        Out = Pop_Gradient
        Offspring = cp.zeros((N, D))
        for i in range(0, N, 2):
            flag = cp.random.rand(1) > 0.5
            miu1 = cp.random.rand(D, ) / 2
            miu2 = cp.random.rand(D, ) / 2 + 0.5
            miu_temp = cp.random.random((D,))
            dictor = MatingPool[i, :] > MatingPool[i + 1, :]
            MatingPool[i][dictor], MatingPool[i + 1][dictor] = MatingPool[i + 1][dictor], MatingPool[i][dictor]
            Out[i][dictor], Out[i + 1][dictor] = Out[i + 1][dictor], Out[i][dictor]
            G_temp = Out[i:i + 2, :].copy()
            L = G_temp[0, :].copy()
            P = miu1.copy()
            P[L > 0] = miu2[L > 0].copy()
            P[L == 0] = miu_temp[L == 0].copy()
            miu = P.copy()
            beta = cp.zeros((D,))
            beta[miu <= 0.5] = (2 * miu[miu <= 0.5]) ** (1 / (DisC + 1))
            beta[miu > 0.5] = (2 - 2 * miu[miu > 0.5]) ** (-1 / (DisC + 1))
            beta[cp.random.random((D,)) > ProC] = 1
            if flag == True:
                beta[MatingPool[i] == 0] = 1
            Offspring[i, :] = ((MatingPool[i, :] + MatingPool[i + 1, :]) / 2) + (
                cp.multiply(beta, (MatingPool[i, :] - MatingPool[i + 1, :]) / 2))
            L = -G_temp[0, :].copy()
            P = miu1.copy()
            P[L > 0] = miu2[L > 0].copy()
            P[L == 0] = miu_temp[L == 0].copy()
            miu = P.copy()
            beta = cp.zeros((D,))
            beta[miu <= 0.5] = (2 * miu[miu <= 0.5]) ** (1 / (DisC + 1))
            beta[miu > 0.5] = (2 - 2 * miu[miu > 0.5]) ** (-1 / (DisC + 1))
            beta[cp.random.random((D,)) > ProC] = 1
            if flag == True:
                beta[MatingPool[i + 1] == 0] = 1
            Offspring[i + 1, :] = ((MatingPool[i, :] + MatingPool[i + 1, :]) / 2) - (
                cp.multiply(beta, (MatingPool[i, :] - MatingPool[i + 1, :]) / 2))
            Out[i][dictor], Out[i + 1][dictor] = Out[i + 1][dictor], Out[i][dictor]
            k1 = cp.random.rand(D, ) > 0.5
            L = G_temp[0, :].copy()
            kl1 = cp.bitwise_and(k1, L < 0)
            L = -G_temp[1, :].copy()
            kl2 = cp.bitwise_and(k1, L < 0)
            Offspring[i][kl1], Offspring[i + 1][kl2] = Offspring[i + 1][kl1], Offspring[i][kl2]
            Out[i][kl1], Out[i + 1][kl2] = Out[i + 1][kl1], Out[i][kl2]
            Offspring[i][dictor], Offspring[i + 1][dictor] = Offspring[i + 1][dictor], Offspring[i][dictor]
        Offspring_temp = Offspring[:MaxOffspring, :].copy()
        Offspring = Offspring_temp
        if MaxOffspring == 1:
            MaxValue = Boundary[0, :]
            MinValue = Boundary[1, :]
        else:
            MaxValue = cp.tile(Boundary[0, :], (MaxOffspring, 1))
            MinValue = cp.tile(Boundary[1, :], (MaxOffspring, 1))
        k = cp.random.random((MaxOffspring, D))
        miu = cp.random.random((MaxOffspring, D))
        Temp = cp.bitwise_and(k <= ProM, miu < 0.5)
        Offspring[Temp] = 0
        Temp = cp.bitwise_and(k <= ProM, miu >= 0.5)
        Offspring[Temp] = 0
        Offspring[Offspring > MaxValue] = MaxValue[Offspring > MaxValue]
        Offspring[Offspring < MinValue] = MinValue[Offspring < MinValue]
    return Offspring

# 环境选择 #
def Environment_Select(Population, Weight_Grad_Temp, FunctionValue, N):
    FrontValue, MaxFront = Non_Dominant_Sorting(FunctionValue, N)  # 非主导排序
    CrowdDistance = Crowded_Distance(FunctionValue, FrontValue)  # 拥挤距离计算
    Next = np.zeros((1, N), dtype="int64")  # 下一代种群
    NoN = np.sum(FrontValue < MaxFront)  # 得到前MaxFNO-1层的个体
    Next[0, :NoN] = np.where(FrontValue < MaxFront)[1]  # 满足条件的索引是个列向量，但可以赋值给行向量
    # 对第MaxFNO层，根据拥挤距离进行选取
    Last = np.where(FrontValue == MaxFront)[1]  # 得到第MaxFNO层个体
    Rank = np.argsort(-(CrowdDistance[0, Last]))  # 根据拥挤距离进行排序
    Next[0, NoN:] = Last[Rank[:N - NoN]]  # 选择拥挤距离大的N-NoN个个体加入
    FrontValue_temp = np.array([FrontValue[0, Next[0, :]]])  # 得到选择个体的序列
    CrowdDistance_temp = np.array([CrowdDistance[0, Next[0, :]]])  # 得到选择个体的拥挤距离
    Population_temp = Population[Next[0, :], :]  # 得到下一代个体
    FunctionValue_temp = FunctionValue[Next[0, :], :]  # 得到选择个体的收敛度
    Weight_Grad = Weight_Grad_Temp[Next[0, :], :]  # 得到选择个体的梯度
    return Population_temp, FunctionValue_temp, Weight_Grad, FrontValue_temp, CrowdDistance_temp

# 拥挤距离 #
def Crowded_Distance(FunctionValue, FrontValue):
    N, M = FunctionValue.shape
    CrowdDistance = np.zeros((1, N))
    temp = np.unique(FrontValue)  # 保留1,2,3,...,MaxFNO,+∞
    Fronts = temp[temp != np.inf]  # 将1,2,3,...,MaxFNO留下
    for f in range(len(Fronts)):
        Front = np.where(FrontValue == Fronts[f])[1]  # array返回两个值，前面的一维的坐标，后面是二维的坐标
        Fmax = np.max(FunctionValue[Front, :], axis=0)
        Fmin = np.min(FunctionValue[Front, :], axis=0)
        for i in range(M):  # 计算拥挤距离
            Rank = FunctionValue[Front, i].argsort()
            CrowdDistance[0, Front[Rank[0]]] = np.inf
            CrowdDistance[0, Front[Rank[-1]]] = np.inf
            for j in range(1, len(Front) - 1, 1):
                CrowdDistance[0, Front[Rank[j]]] = CrowdDistance[0, Front[Rank[j]]] + \
                                                   (FunctionValue[Front[Rank[j + 1]], i] - FunctionValue[
                                                       Front[Rank[j - 1]], i]) / (Fmax[i] - Fmin[i])
    return CrowdDistance

# 预排序 #
def Pre_Sort(Matrix, order="ascend"):
    Matrix_temp = Matrix[:, ::-1]
    Matrix_row = Matrix_temp.T
    if order == "ascend":
        rank = np.lexsort(Matrix_row)
    elif order == "descend":
        rank = np.lexsort(-Matrix_row)
    Sorted_Matrix = Matrix[rank, :]
    return Sorted_Matrix, rank

# 非主导排序 #
def Non_Dominant_Sorting(PopObj, Remain_Num):
    N, M = PopObj.shape  # N种群数量，M目标数（网络维数）
    FrontNO = np.inf * np.ones((1, N))  # +∞
    MaxFNO = 0
    PopObj, rank = Pre_Sort(PopObj)  # 从小到大进行排序
    while (np.sum(FrontNO < np.inf) < Remain_Num):  # 进行划分
        MaxFNO += 1  # 层数加1
        for i in range(N):
            if FrontNO[0, i] == np.inf:
                Dominated = False
                for j in range(i - 1, -1, -1):  # 将第i个个体和前面的个体一一比较，如果没有能支配它的个体，则在第MaxFNO层
                    if FrontNO[0, j] == MaxFNO:
                        m = 2
                        while (m <= M) and (PopObj[i, m - 1] >= PopObj[j, m - 1]):
                            m += 1
                        Dominated = m > M
                        if Dominated or (M == 2):
                            break
                if not Dominated:
                    FrontNO[0, i] = MaxFNO
    front_temp = np.zeros((1, N))
    front_temp[0, rank] = FrontNO  # 非主导排序完成后的序列
    return front_temp, MaxFNO  # 排序后的序列，排序到第MaxFNO层

def Coperate_Weight_Grad(Weight_Grad_Temp, Weight_Grad_T_):
    Result_Grad = Weight_Grad_Temp + Weight_Grad_T_
    Temp_1 = Weight_Grad_Temp
    Temp_1[Temp_1 > 0] = 1
    Temp_1[Temp_1 < 0] = -1
    Temp_2 = Weight_Grad_T_
    Temp_2[Temp_2 > 0] = 1
    Temp_2[Temp_2 < 0] = -1
    zeroIndex = (Temp_1 + Temp_2) == 0
    Result_Grad[zeroIndex] = 0
    return Result_Grad

class Population:
    def __init__(self, min_range, max_range, dim, factor, rounds, size, object_func, CR=0.75):
        self.min_range = min_range
        self.max_range = max_range
        self.dimension = dim
        self.factor = factor
        self.rounds = rounds
        self.size = size
        self.cur_round = 1
        self.CR = CR
        self.get_object_function_value = object_func
        # 初始化种群
        self.individuality = [np.array([random.uniform(self.min_range, self.max_range) for s in range(self.dimension)])
                              for tmp in range(size)]
        self.object_function_values = [self.get_object_function_value(v) for v in self.individuality]
        self.mutant = None

    def mutate(self):
        self.mutant = []
        for i in range(self.size):
            r0, r1, r2 = 0, 0, 0
            while r0 == r1 or r1 == r2 or r0 == r2 or r0 == i:
                r0 = random.randint(0, self.size - 1)
                r1 = random.randint(0, self.size - 1)
                r2 = random.randint(0, self.size - 1)
            tmp = self.individuality[r0] + (self.individuality[r1] - self.individuality[r2]) * self.factor
            for t in range(self.dimension):
                if tmp[t] > self.max_range or tmp[t] < self.min_range:
                    tmp[t] = random.uniform(self.min_range, self.max_range)
            self.mutant.append(tmp)

    def crossover_and_select(self):
        for i in range(self.size):
            Jrand = random.randint(0, self.dimension)
            for j in range(self.dimension):
                if random.random() > self.CR and j != Jrand:
                    self.mutant[i][j] = self.individuality[i][j]
                tmp = self.get_object_function_value(self.mutant[i])
                if tmp < self.object_function_values[i]:
                    self.individuality[i] = self.mutant[i]
                    self.object_function_values[i] = tmp

def cross(population, alfa, numRangeList, n=2):
    N = population.shape[0]
    V = population.shape[1]
    populationList = range(N)
    for _ in range(N):
        r = random.random()
        if r < alfa:
            p1, p2 = random.sample(populationList, 2)
            beta = np.array([0] * V)
            randList = np.random.random(V)
            for j in range(V):
                if randList.any() <= 0.5:
                    beta[j] = (2.0 * randList[j]) ** (1.0 / (n + 1))
                else:
                    beta[j] = (1.0 / (2.0 * (1 - randList[j]))) ** (1.0 / (n + 1))
                # 随机选取两个个体
                old_p1 = population[p1,]
                old_p2 = population[p2,]
                # 交叉
                new_p1 = 0.5 * ((1 + beta) * old_p1 + (1 - beta) * old_p2)
                new_p2 = 0.5 * ((1 - beta) * old_p1 + (1 + beta) * old_p2)
                # 上下界判断
                new_p1 = np.max(np.vstack((new_p1, np.array([0] * V))), 0)
                new_p1 = np.min(np.vstack((new_p1, numRangeList)), 0)
                new_p2 = np.max(np.vstack((new_p2, np.array([0] * V))), 0)
                new_p2 = np.min(np.vstack((new_p2, numRangeList)), 0)
                # 将交叉后的个体返回给种群
                population[p1,] = new_p1
                population[p2,] = new_p2