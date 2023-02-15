import numpy as np
import pickle
from scipy import stats
import sys
import argparse
import os
join=os.path.join

parser = argparse.ArgumentParser()

parser.add_argument('--n', type=int, default=1000)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--log_dir', type=str, default='logs')
parser.add_argument('--runs', type=int, default=2)

args = parser.parse_args()

np.random.seed(args.seed)

def get_kde_estimates(bandwidth, data) :
	# kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
	data = data.T
	kernel = stats.gaussian_kde(data, bandwidth)
	return kernel.evaluate(data)

max_data_size = args.n
log_dir = args.log_dir

if not os.path.exists(log_dir):
	os.makedirs(log_dir)

bandwidth = 'scott'

data = pickle.load(open('./higgs/HIGGS_TST.pckl', 'rb'))
dataX = data[0]
dataY = data[1]

del data

dataX = dataX[:,:4]
dataY = dataY[:,:4]

fname = join(log_dir, 'result_' + str(max_data_size) + '.txt')

def in_top_k(scores, k):
	indices = np.argsort(scores)[::-1]
	pos = np.where(indices==0)[0][0]
	return pos <= k

def evaluate():
	scores = []
	idx1 = np.random.choice(len(dataX), max_data_size, replace=False)
	idx2 = np.random.choice(len(dataY), max_data_size, replace=False)

	data1 = dataX[idx1]
	data2 = dataY[idx2]
	for rep in range(100):

		if rep != 0:
			data_all = np.concatenate([data1, data2], axis=0)
			data_all = data_all[np.random.permutation(range(data_all.shape[0]))]
			data1 = data_all[:max_data_size]
			data2 = data_all[max_data_size:]
		datam = np.concatenate([data1[:max_data_size//2], data2[:max_data_size//2]], axis=0)
		
		logprob_1 = get_kde_estimates(bandwidth, data1)
		logprob_2 = get_kde_estimates(bandwidth, data2)
		logprob_m = get_kde_estimates(bandwidth, datam)

		vdiv = np.mean(-logprob_m) - min(np.mean(-logprob_1), np.mean(-logprob_2))
		with open(fname, 'a') as f:
			f.write("Scores " + str(np.mean(logprob_1)) + " " + str(np.mean(logprob_2)) + " " + str(np.mean(logprob_m)) + "\n")
		scores.append(vdiv)

	return in_top_k(scores, k=5)


def get_scores():	
	evaluations = 100
	print('total runs - ', args.runs)
	across_runs_test_power = []
	for run in range(args.runs):
		print('Starting run - ', run)
		test_power_val = []
		np.random.shuffle(dataX)
		np.random.shuffle(dataY)
		for evaluation in range(evaluations):
			test_power_val.append(evaluate()*1.0)
			with open(fname, 'a') as f:
				f.write(str(evaluation) + " " + str(np.mean(test_power_val)) + '\n')
			print('Iteration ', evaluation, '/', evaluations, ' Average Test power - ', np.mean(test_power_val))

		with open(fname, 'a') as f:
			f.write(str(np.mean(test_power_val)) + " " + str(np.std(test_power_val)) + '\n')
		print(np.mean(test_power_val), np.std(test_power_val))
		across_runs_test_power.append(np.mean(test_power_val))
		print(across_runs_test_power)
	print('across ', args.runs, ' runs the mean test power is ', np.mean(across_runs_test_power))

if __name__ == "__main__":
	get_scores()








