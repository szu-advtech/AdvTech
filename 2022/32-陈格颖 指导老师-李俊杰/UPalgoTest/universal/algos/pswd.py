import os
import csv
import datetime
from ..algo import Algo
# from MyLogger import MyLogger
from .GeneticAlgorithm import Input
from .GeneticAlgorithm2 import Genetic
from .Gen_Data import var_mean
import numpy as np
import pandas as pd
from universal.result import ListResult
from universal import tools
import heapq



class PSWD(Algo):
    """pswd algo"""
    PRICE_TYPE = 'ratio'
    REPLACE_MISSING = True
    def __init__(self, filename="msci", batch=1):
        super(PSWD, self).__init__()
        self.batch = batch
        self.top_low = TopLowStocksSelector()

        # self.logger = MyLogger('pswd_log')
        self.filename = "/home/aze/project/UPalgoTest/universal/data/" + filename + ".pkl"
        # self.filename = filename
    def init_weights(self, m):
        return np.ones(m) / m

    def step(self, x, last_b, history):
        if len(history) % self.batch != 0:
            b = last_b
        else:
            b = self.updateWeight(history)
            # self.top_low.TopLowStocksSave(b, history)

        return b

    def updateWeight(self, history):
        ndays = history.shape[0]
        input = Input()
        one_generation = input.pop_size_chromosomes(self.filename, ndays)
        meanlists = var_mean(self.filename, ndays)[1]
        # meanlists = np.array(meanlists)
        pop = np.array(one_generation)
        gens = Genetic()
        for i in range(gens.N_GENERATIONS):
            pop = np.array(gens.crossover_and_mutation(pop))
            pop = gens.Normalized(pop)
            fitness = gens.get_fitness(pop, meanlists)
            pop = gens.select(pop, fitness)
        fitness = gens.get_fitness(pop, meanlists)
        epsilon = np.array(meanlists)
        max_fitness_index = np.argmax(fitness)
        max_portation = input.Normalized(pop[max_fitness_index])

        b = np.array(max_portation)
        return b


class TopLowStocksSelector():
    def __init__(self):
        self.nTopStocks = 3
        self.nLowStocks = 3
        # self.savefile = os.getcwd() + '/resultSave/' + str(datetime.datetime.now()) + 'PSWD' + '.csv'
        # self.file = open(self.savefile, 'w')
        # self.csv_writer = csv.writer(self.file)

        # the file to save final result
        # self.resfile = os.getcwd() + '/buySave/' + str(datetime.datetime.now()) + 'PSWD' + '.csv'
        # self.refile = open(self.resfile, 'w')
        # self.csv_writer2 = csv.writer(self.refile)

    def selector(self, b):
        b = list(b)
        top_index = list(map(b.index, heapq.nlargest(self.nTopStocks, b)))
        last_index = list(map(b.index, heapq.nsmallest(self.nLowStocks, b)))
        # b = np.array(b)

        return top_index, last_index
        # top_value = b[top_index]
        # low_value = b[last_index]

    def TopLowStocksSave(self, b, history):
        b = np.array(b)
        top_index, low_index = self.selector(b)
        top_value = b[top_index]
        low_value = b[low_index]
        # self.csv_writer2.writerow(
        #     ["day", "index of top", "weights of top", "index of low", "weights of low"])
        # self.csv_writer2.writerow([[str(0) + '--' + str(history.shape[0])],
        #                            top_index, list(top_value),
        #                            low_index, list(low_value)])









if __name__ == "__main__":
    result = tools.quickrun(PSWD())
    res = ListResult([result], ['PSWD'])
    df = res.to_dataframe()
    df.to_csv('PSWD_profit.csv')

    result.B.to_csv('PSWD_balances.csv')