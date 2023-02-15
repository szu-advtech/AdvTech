import os
import pickle
import argparse
from local_path import data_dir


def evaluate(dataset):

    with open(os.path.join(data_dir, dataset + '.pkl'), 'rb') as f:
        data = pickle.load(f)

    obfuscate_keys = ['user_visit'] + [e for e in list(data.keys()) if 'obfuscate' in e]
    # Only one iteration
    for key in obfuscate_keys:
        os.system('python FedMF.py --dataset {} --visiting {} --iteration {}'.format(dataset, key, 1))


params_parser = argparse.ArgumentParser()
params_parser.add_argument('--dataset', type=str, default='movie_lens')
args = params_parser.parse_args()

evaluate(args.dataset)
