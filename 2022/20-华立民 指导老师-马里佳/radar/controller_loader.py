import radar.algorithm as algorithm
import radar.data as data
from os.path import join
import radar.utils as utils
import random

def get_paths(basepath, algorithm_name, params):
    adversary_ratio = params["adversary_ratio"]
    if adversary_ratio is not None:
        adversary_ratio = float(adversary_ratio)
    data_prefix_pattern = "{}-agents_domain-{}_adversaryratio-{}_{}_".format(params["nr_agents"],\
        params["domain_name"], adversary_ratio, algorithm_name)
    directories = data.list_directories(basepath, lambda x: x.startswith(data_prefix_pattern))
    result = []
    predicate = lambda a,x: x.startswith("protagonist_model") or x.startswith("adversary_model")
    for directory in directories:
        if len(data.list_files_with_predicate(directory, predicate)) == 2:
            result.append(directory)
    assert len(result) > 0, "{} not found".format(data_prefix_pattern) 
    return result

def load_agents(path, algorithm_name, params):
    agents = algorithm.make(algorithm_name, params)
    agents.load_weights(path)
    return agents

def combine_agents(protagonist_controller, adversary_controller, algorithm_name, params):
    controller = algorithm.make(algorithm_name, params)
    adversary_ratio = params["adversary_ratio"]
    controller.adversary_ratio = adversary_ratio
    controller.policy_net.protagonist_net = protagonist_controller.policy_net.protagonist_net
    controller.policy_net.adversary_net = adversary_controller.policy_net.adversary_net
    if adversary_ratio == 0:
        # Play 50:50 mixture of protagonists if adversary controller is purely cooperative
        controller.adversary_ratio = 0.5
        controller.policy_net.adversary_net = adversary_controller.policy_net.protagonist_net
    return controller