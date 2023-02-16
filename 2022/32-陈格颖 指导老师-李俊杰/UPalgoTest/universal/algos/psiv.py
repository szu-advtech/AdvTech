from ..algo import Algo
# from MyLogger import MyLogger
from .GeneticAlgorithm import Input
# from .GeneticAlgorithm2 import Genetic
from .Gen_Data import var_mean
import numpy as np
from .itrv import Genetic, IntervalTypeModel
import pandas as pd
from universal.result import ListResult
from universal import tools


class PSIV(Algo):
    PRICE_TYPE = 'ratio'
    REPLACE_MISSING = True

    def __init__(self, batch=1):
        super(PSIV, self).__init__()
        self.batch = batch
        self.filename = "/home/aze/Documents/szufintech-portfolio-risk-master/portfolio-risk/UPalgoTest/universal/data/djia.pkl"
        #self.filename = filename


    def initweight(self, m):
        return np.ones(m) / m

    def step(self, x, last_b, history):
        if history.shape[0] % self.batch != 0:
            b = last_b
        else:
            b = self.updateWeight(history)
        return b

    def updateWeight(self, history):
        ndays = history.shape[0]
        # input = Input()
        # one_generation = input.pop_size_chromosomes(self.filename, ndays)
        calculateMean = IntervalTypeModel()
        # rl, ru = calculateMean.create_r(self.filename, ndays)
        pop = calculateMean.creatRandomPop()
        # meanlists = var_mean(self.filename, ndays)[1]
        # meanlists = np.array(meanlists)
        pop = np.array(pop)
        gen = Genetic()
        for i in range(gen.N_GENERATIONS):
            pop = np.array(gen.crossover_and_mutation(pop))
            pop = gen.Normalized(pop)
            fitness = gen.get_fitness(self.filename, pop, ndays)
            pop = gen.select(pop, fitness)  # 选择生成新的种

        fitness = gen.get_fitness(self.filename, pop, ndays)
        max_fitness_index = np.argmax(fitness)
        input = Input()
        max_portation = input.Normalized(pop[max_fitness_index])
        max_portation = np.array(max_portation)
        return max_portation


#
# if __name__ == "__main__":
#     result = tools.quickrun(PSIV())
#     res = ListResult([result], ['PSIV'])
#     df = res.to_dataframe()
#     df.to_csv('PSIV_profit.csv')
#
#     result.B.to_csv('PSIV_balances.csv')