from tensorboard_logger import Logger as TbLogger
from parameters import getParameters
from problem_type.pdtsp import PDTSP
from problem_type.pdtsplifo import PDTSPLIFO
from training_algorithm.ppo import PPO
import os
import torch
import numpy as np
import json
import pprint


def load_rl_model(name):
    rl_model = {
        'ppo': PPO,
    }.get(name, None)
    assert rl_model is not None, "目前不支持的训练算法: {}!".format(name)
    return rl_model

def load_instance(str):
    problem = {
        'pdtsp': PDTSP,
        'pdtspl': PDTSPLIFO,
    }.get(str, None)
    assert problem is not None, "Currently unsupported problem: {}!".format(str)
    return problem


def execute(model_parameters):

    # 打印参数
    pprint.pprint(vars(model_parameters))

    # 设置随机种子
    torch.manual_seed(model_parameters.seed)
    np.random.seed(model_parameters.seed)

    # 可选配置tensorboard
    tb_logger = None
    if not model_parameters.no_tb and not model_parameters.distributed:
        tb_logger = TbLogger(os.path.join(model_parameters.log_dir, "{}_{}".format(model_parameters.problem, 
                                                          model_parameters.graph_size), model_parameters.run_name))
    if not model_parameters.no_saving and not os.path.exists(model_parameters.save_dir):
        os.makedirs(model_parameters.save_dir)

    # 保存参数，以便始终能找到准确的配置
    if not model_parameters.no_saving:
        with open(os.path.join(model_parameters.save_dir, "args.json"), 'w') as f:
            json.dump(vars(model_parameters), f, indent=True)

    # 对设备进行设置
    model_parameters.device = torch.device("cuda" if model_parameters.use_cuda else "cpu")

    print("--------------加载问题-------------")
    # 指定问题
    problem = load_instance(model_parameters.problem)(
                            p_size = model_parameters.graph_size,
                            init_val_met = model_parameters.init_val_met,
                            with_assert = model_parameters.use_assert)

    print("--------------加载模型-------------")
    # RL算法模型
    rl_model = load_rl_model(model_parameters.RL_agent)(problem.NAME, problem.size,  model_parameters)

    path = model_parameters.load_path if model_parameters.load_path is not None else model_parameters.resume
    if path is not None:
        rl_model.load(path)

    # 验证
    if model_parameters.eval_only:
        # 加载验证测试问题
        rl_model.inference(problem, model_parameters.val_dataset, tb_logger)
        print("--------------开始验证-------------")
    else:
        if model_parameters.resume:
            epoch_resume = int(os.path.splitext(os.path.split(model_parameters.resume)[-1])[0].split("-")[1])
            print("Resuming after {}".format(epoch_resume))
            rl_model.parameters.epoch_start = epoch_resume + 1
        print("--------------开始训练-------------")
        # 开始训练循环
        rl_model.training(problem, model_parameters.val_dataset, tb_logger)
            


if __name__ == "__main__":

    # 设置算法的确定式属性
    torch.backends.cudnn.deterministic = True
    # 执行算法
    execute(getParameters())
