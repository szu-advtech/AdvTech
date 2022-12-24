# nr_steps = 2000000
nr_steps = 10000
params = {}
params["test_adversary_ratios"] = [-1, 0, 0.25, 0.5, 0.75]
#params["test_adversary_ratios"] = [0.75]
params["test_algorithms"] = ["RADAR_X"]
#params["test_algorithms"] = ["RAT_DQN"]
params["test_directory"] = "tests"
params["test_interval"] = 10
params["nr_test_episodes"] = 50
params["use_global_reward"] = True
params["save_summaries"] = True
params["alpha"] = 0.001

# These hyperparameters are only required for DQN, VDN, QMIX
params["warmup_phase"] = 10000
params["target_update_period"] = 4000
params["memory_capacity"] = 20000
params["epsilon_decay"] = 1.0/50000

# Uncomment to manually set random seed
"""
GLOBAL_SEED = 42
import torch
import numpy
import random
torch.manual_seed(GLOBAL_SEED)
numpy.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
"""