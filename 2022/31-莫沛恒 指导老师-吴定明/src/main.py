import parameter
import configs
import cluster
import REINFORCE_tfagent
import DQN_tfagent
import workload
import PPO_tfagent
import DDPG_tfagent


def main():
    parameter.load_config()
    workload.read_workload()
    cluster.init_cluster()

    if configs.algo == 'reinforce':
        print("Running Reinforce Algorithm with iteration: {}, workload: {}, beta: {}"
              .format(configs.iteration, configs.workload, configs.beta))
        REINFORCE_tfagent.train_reinforce(num_iterations=configs.iteration)
    elif configs.algo == 'dqn':
        print("Running DQN Algorithm with iteration: {}, workload: {}, beta: {}"
              .format(configs.iteration, configs.workload, configs.beta))
        DQN_tfagent.train_dqn(num_iterations=configs.iteration)
    elif configs.algo == 'ppo':
        print("Running PPO Algorithm with iteration: {}, workload: {}, beta: {}"
              .format(configs.iteration, configs.workload, configs.beta))
        PPO_tfagent.train_ppo(num_iterations=configs.iteration)
    elif configs.algo == 'ddpg':
        print("Running DDPG Algorithm with iteration: {}, workload: {}, beta: {}"
              .format(configs.iteration, configs.workload, configs.beta))
        DDPG_tfagent.train_ddpg(num_iterations=configs.iteration)
    else:
        print('Please specify valid algo option in config.ini file\n')


if __name__ == '__main__':
    main()
