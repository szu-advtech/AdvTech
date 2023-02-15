import radar.domain as domain
import radar.algorithm as algorithm
import radar.experiments as experiments
import radar.data as data
import radar.utils as utils
import sys
from settings import params, nr_steps

params["adversary_ratio"] = utils.get_float_argument(sys.argv, 3, None)
params["test_suite"] = experiments.run_test_suite
params["nr_test_episodes"] = 50
params["domain_name"] = utils.get_argument(sys.argv, 2, None)
assert params["domain_name"] is not None, "domain_name is required"

params["algorithm_name"] = sys.argv[1]
env = domain.make(params["domain_name"], params)
nr_episodes = int(nr_steps/env.time_limit)        #time_limit默认50
params["directory"] = "output/{}-agents_domain-{}_adversaryratio-{}_{}".\
    format(params["nr_agents"], params["domain_name"],\
        params["adversary_ratio"], params["algorithm_name"])
params["directory"] = data.mkdir_with_timestap(params["directory"]) #在  字符串后加上当前时间，并创建/返回该文件夹夹
params["global_observation_shape"] = env.global_observation_space.shape  #返回矩阵格式
params["local_observation_shape"] = env.local_observation_space.shape
params["nr_actions"] = env.action_space.n
params["gamma"] = env.gamma
params["env"] = env
controller = algorithm.make(params["algorithm_name"], params)
result = experiments.run(controller, nr_episodes, params, log_level=0)
