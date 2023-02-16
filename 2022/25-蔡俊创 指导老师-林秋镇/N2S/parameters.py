import os
import time
import argparse
import torch


def getParameters(parameters=None):
    parser = argparse.ArgumentParser(description="神经邻域搜索")

    # 日志输出设置
    parser.add_argument('--no_progress_bar', action='store_true', help='禁用进度条')
    parser.add_argument('--log_dir', default='logs', help='要写入TensorBoard信息的目录')
    parser.add_argument('--log_step', type=int, default=50, help='每个log_step梯度步骤的日志信息')
    parser.add_argument('--output_dir', default='outputs', help='要写入输出模型的目录')
    parser.add_argument('--run_name', default='run_name', help='用于识别运行进程的名称')
    parser.add_argument('--checkpoint_epochs', type=int, default=1,
                        help='保存检查点每n个epoch(默认为1)，0表示不保存检查点')
    # 设置
    parser.add_argument('--problem', default='pdtsp', choices = ['pdtsp','pdtspl'], help="待解决的目标问题, 默认 'pdp'")
    parser.add_argument('--show_figs', action='store_true', help='是否启用图记录')
    parser.add_argument('--no_saving', action='store_true', help='禁止保存断点')
    parser.add_argument('--use_assert', action='store_true', help='启用assertion判断')
    parser.add_argument('--no_DDP', action='store_true', help='禁用分布式并行')
    parser.add_argument('--seed', type=int, default=1234, help='使用的随机种子')
    parser.add_argument('--graph_size', type=int, default=100,
                        help="目标问题中的客户数量")
    parser.add_argument('--init_val_met', choices=['greedy', 'random'], default='random',
                        help='生成推理初始解的方法')
    parser.add_argument('--no_cuda', action='store_true', help='禁用 GPUs')
    parser.add_argument('--no_tb', action='store_true', help='关闭Tensorboard日志记录')

    # 训练参数的设置
    parser.add_argument('--RL_agent', default='ppo', choices = ['ppo'], help='RL的训练算法使用ppo')
    parser.add_argument('--gamma', type=float, default=0.999, help='未来奖励折扣因子')
    parser.add_argument('--K_epochs', type=int, default=3, help='最小的 PPO epoch')
    parser.add_argument('--eps_clip', type=float, default=0.1, help='PPO的clip比例')
    parser.add_argument('--T_train', type=int, default=250, help='训练的迭代次数')
    parser.add_argument('--n_step', type=int, default=5, help='第N步估算收益')
    parser.add_argument('--warm_up', type=float, default=2, help='课程学习CL标量的超参数')
    parser.add_argument('--batch_size', type=int, default=600,help='训练期间每个batch的问题problem_type个数')
    parser.add_argument('--epoch_end', type=int, default=200, help='最大训练epoch')
    parser.add_argument('--epoch_size', type=int, default=12000, help='训练期间每个epoch的问题（problem_type）数')
    parser.add_argument('--lr_model', type=float, default=8e-5, help="动作网络的学习率")
    parser.add_argument('--lr_critic', type=float, default=2e-5, help="评价网络的学习率")
    parser.add_argument('--lr_decay', type=float, default=0.985, help='每轮学习率衰减')
    parser.add_argument('--max_grad_norm', type=float, default=0.05, help='最大梯度裁剪的L2范数')
    # 算法参数设置
    parser.add_argument('--v_range', type=float, default=6., help='控制熵')
    parser.add_argument('--actor_head_num', type=int, default=4, help='N2S的动作网络的头数')
    parser.add_argument('--critic_head_num', type=int, default=4, help='N2S的评价网络的头数')
    parser.add_argument('--embedding_dim', type=int, default=128, help='输入d嵌入维度(NEF和PFE)')
    parser.add_argument('--hidden_dim', type=int, default=128, help='编码器和解码器中隐藏层的尺寸')
    parser.add_argument('--n_encode_layers', type=int, default=3, help='编码器中堆叠的层数')
    parser.add_argument('--normalization', default='layer', help="规范化类型，` layer `(默认)或` batch `")
    #  恢复和加载模型
    parser.add_argument('--load_path', default = None, help='加载模型参数和优化器状态的路径')
    parser.add_argument('--resume', default = None, help='从先前的检查点文件恢复')
    parser.add_argument('--epoch_start', type=int, default=0, help='从epoch开始(与学习率衰减相关)')
    # 推断和验证的参数
    parser.add_argument('--T_max', type=int, default=1500, help='推断的步骤数')
    parser.add_argument('--eval_only', action='store_true', help='切换到推理模式')
    parser.add_argument('--val_size', type=int, default=1000, help='验证/推断的问题数量')
    parser.add_argument('--val_batch_size', type=int, default=1000, help='每个batch用于验证/推理的问题数量')
    parser.add_argument('--val_dataset', type=str, default='./instances/pdp_100.pkl', help='数据集文件存储路径')
    parser.add_argument('--val_m', type=int, default=1, help='算法2的数据强化数')

    parameterList = parser.parse_args(parameters)
    
    # 确定是否使用分布式训练
    parameterList.world_size = torch.cuda.device_count()
    parameterList.distributed = (torch.cuda.device_count() > 1) and (not parameterList.no_DDP)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '4879'
    assert parameterList.val_m <= parameterList.graph_size // 2
    parameterList.use_cuda = torch.cuda.is_available() and not parameterList.no_cuda
    parameterList.run_name = "{}_{}".format(parameterList.run_name, time.strftime("%Y%m%dT%H%M%S")) \
        if not parameterList.resume else parameterList.resume.split('/')[-2]
    parameterList.save_dir = os.path.join(
        parameterList.output_dir,
        "{}_{}".format(parameterList.problem, parameterList.graph_size),
        parameterList.run_name
    ) if not parameterList.no_saving else None
    
    return parameterList
