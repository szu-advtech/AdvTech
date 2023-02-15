import shutil

import numpy as np
import logging
import sys
import argparse
import time
import os

import torch.cuda
from pymoo.algorithms.nsga2 import NSGA2, nsga2, binary_tournament
from pymoo.operators.crossover.point_crossover import PointCrossover
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.random_sampling import RandomSampling
from pymoo.operators.selection.tournament_selection import TournamentSelection
from pymoo.optimize import minimize
from pymop.problem import Problem

import model_genotype
import train_operation


# =================================================================================
# set arguments for Terminal input
parser = argparse.ArgumentParser("Multi-objective Genetic Algorithm for NAS")
parser.add_argument('--save', type=str, default='NSGANetV1', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')

parser.add_argument('--n_blocks', type=int, default=5, help='number of blocks in a cell')
parser.add_argument('--n_ops', type=int, default=12, help='number of operations considered')
parser.add_argument('--n_cells', type=int, default=2, help='number of cells to search')

parser.add_argument('--n_nodes', type=int, default=5, help='number of nodes per phases')

parser.add_argument('--pop_size', type=int, default=40, help='population size for search')
parser.add_argument('--n_gens', type=int, default=30, help='number of total generations')
parser.add_argument('--n_offspring', type=int, default=20, help='number of offsprings')

parser.add_argument('--init_channels', type=int, default=24, help='#channels of filters for first cell')
parser.add_argument('--layers', type=int, default=11, help='number of layers of the networks')
parser.add_argument('--epochs', type=int, default=25, help='training epochs for each individual')

parser.add_argument('--device', type=str, default='cuda', help='GPU/CPU device selected')

args = parser.parse_args()
args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))

device = args.device if torch.cuda.is_available() else 'cpu'

# =================================================================================
def mkdir_save(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    # print(f'Experiment directory: {path}')

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path,'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


# make save directory
mkdir_save(args.save)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log_search.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


# ------------------
# define your NAS problem
# -----------------
class NAS(Problem):

    def __init__(self, n_var=40, n_obj=2, n_constr=0, lb=None, ub=None,
                 init_channels=24, layers=8, epochs=25, save_dir=None):
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, type_var=np.int32)
        self.xl = lb    # lb = lower bound
        self.xu = ub    # ub = upper bound
        self._init_channels = init_channels
        self._layers = layers
        self._epochs = epochs
        self._save_dir = save_dir
        self._n_evaluate = 0        # record the architecture ID

    def _evaluate(self, x, out, *args, **kwargs):

        objs = np.full((x.shape[0], self.n_obj), np.nan)    # x.shape[0]: nums of individuals

        for i in range(x.shape[0]):     # for each individual
            arch_id = self._n_evaluate + 1
            print('\n')
            logging.info(f'Network id = {arch_id}')

            genome = x[i, :]    # extract the genotype of the individual (in index type)
            architecture = model_genotype.decode(genome)
            logging.info(f'Architecture = {architecture}')
            performance = train_operation.evaluate(genome, init_channels=self._init_channels, layers=self._layers,
                                                   epochs=self._epochs, device=device)

            objs[i, 0] = 100 - performance['acc']
            objs[i, 1] = performance['flops']
            logging.info(f"FLOPs: {performance['flops']/1e6} M, parameters: {performance['params']/1e6} MB, "
                         f"Acc: {performance['acc']}")
            self._n_evaluate += 1

            if self._save_dir is not None:
                arch_save_path = os.path.join(self._save_dir, f'arch_{arch_id}')
                mkdir_save(arch_save_path)
                with open(os.path.join(arch_save_path, 'arch.txt'), 'w') as file:
                    file.write(f"Genome = {genome}\n")
                    file.write(f"Architecture = {architecture}\n")
                    file.write(f"param size = {performance['params']/1e6}MB\n")
                    file.write(f"FLOPs = {performance['flops']/1e6}M\n")
                    file.write(f"valid_acc = {performance['acc']}\n")

                with open(os.path.join(arch_save_path, 'genome'), "w") as file:
                    for element in genome:
                        file.write(f"{element} ")

        out["F"] = objs     # Fitness


# ---------------------------------------------------------------------------------------------------------
# Define what statistics to print or save for each generation
# ---------------------------------------------------------------------------------------------------------
def do_every_generations(algorithm):
    # this function will be call every generation
    # it has access to the whole algorithm class
    gen = algorithm.n_gen
    pop_var = algorithm.pop.get("X")
    pop_obj = algorithm.pop.get("F")

    # report generation info to files
    logging.info(f"generation = {gen}")
    logging.info(f"population error: best = {np.min(pop_obj[:, 0])}, mean = {np.mean(pop_obj[:, 0])}, "
                 f"median = {np.median(pop_obj[:, 0])}, worst = {np.max(pop_obj[:, 0])}")
    logging.info(f"population complexity: best = {np.min(pop_obj[:, 1])}, mean = {np.mean(pop_obj[:, 1])}, "
                 f"median = {np.median(pop_obj[:, 1])}, worst = {np.max(pop_obj[:, 1])}")

    # save result
    mkdir_save(os.path.join(args.save, f'gen_{gen}'))
    with open(os.path.join(args.save, f'gen_{gen}', 'individuals'), 'w') as file:
        for x in pop_var:
            for element in x:
                file.write(f"{element} ")
            file.write("\n")

    with open(os.path.join(args.save, f'gen_{gen}', 'fitness'), 'w') as file:
        file.write(f"Accuracy\tFLOPs(M)\n")
        for f in pop_obj:
            file.write(f"{100 - f[0]:.6f}\t{f[1] / 1e6:4f}\n")


def main():

    # set Problem
    n_blocks = args.n_blocks
    n_var = 4 * n_blocks * 2
    lower_bounds, upper_bounds = get_lb_hb(n_blocks)
    problem = NAS(n_var=40, n_obj=2, n_constr=0, lb=lower_bounds, ub=upper_bounds,
                  init_channels=args.init_channels, layers=args.layers, epochs=args.epochs,
                  save_dir=args.save)

    # set NSGA2
    method = nsga2(pop_size=args.pop_size,
                   sampling=RandomSampling(var_type=np.int),
                   selection=TournamentSelection(func_comp=binary_tournament),
                   crossover=PointCrossover(n_points=4),
                   mutation=PolynomialMutation(eta=3, var_type=np.int),
                   n_offsprings=args.n_offspring,
                   elimiate_duplicates=True)

    # Run search: minimize
    res = minimize(problem=problem,
                   method=method,
                   termination=('n_gen', args.n_gens),
                   callback=do_every_generations)

    # save result
    mkdir_save(os.path.join(args.save, 'result'))
    with open(os.path.join(args.save, 'result', 'individuals'), 'w') as file:
        for x in res.X:
            for element in x:
                file.write(f"{element} ")
            file.write("\n")

    with open(os.path.join(args.save, 'result', 'fitness'), 'w') as file:
        file.write(f"Accuarcy\tFLOPs(M)\n")
        for f in res.F:
            file.write(f"{100-f[0]:.6f}\t{f[1]/1e6:4f}\n")


def get_lb_hb(n_blocks=5):
    n_var = 4 * n_blocks * 2
    lb = np.zeros(n_var)
    ub = np.ones(n_var)
    h = 1
    for i in range(0, n_var//2, 4):
        ub[i] = args.n_ops - 1
        ub[i+1] = h
        ub[i+2] = args.n_ops - 1
        ub[i+3] = h
        h += 1
    ub[n_var//2 : ] = ub[:n_var//2]
    return lb, ub


if __name__ == '__main__':
    main()

