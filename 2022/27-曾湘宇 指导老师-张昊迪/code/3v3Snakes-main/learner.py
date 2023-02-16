import datetime
import multiprocessing
import pickle
import subprocess
import time
from argparse import ArgumentParser
from multiprocessing import Process
from pathlib import Path

import horovod.tensorflow.keras as hvd
import tensorflow as tf
import zmq
from pyarrow import deserialize
from tensorflow.keras.backend import set_session

from core.mem_pool import MemPoolManager, MultiprocessingMemPool
from custom_model import ACCNNModel
from ppo_agent import PPOAgent
from utils import logger
from utils.cmdline import parse_cmdline_kwargs

# Horovod: initialize Horovod.
'''
Horovod是Uber于2017年开源的分布式训练框架，它的优势就是兼容主流框架Tensorflow，PyTorch，MXNet等
'''
hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
'''
tf.ConfigProto用于配置tensorflow的Session
# 注意里面的参数要自己配置
1. 记录设备指派情况： tf.ConfigProto(log_device_placement=True)
        设置tf.ConfigProto()中参数log_device_placement=True,可以获取到operations和Tensor被指派到哪个设备（几号CPU或几号GPU）上运行，会在终端打印出各项操作是在那个设备上运行的。
2. 自动选择运行设备：tf.ConfigProto(allow_soft_placement=True)
        在tf中，通过命令 “with tf.device("/cpu:0"):”，允许手动设置操作运行的设备。如果手动设置不存在或者不可用，为了防止这种情况，可以设置tf.ConfigProto()中参数 allow_soft_placement=True,允许tf自动选择一个存在并且可用的设备来运行操作。
3. 限制GPU资源使用
        为了加快运行效率，tf 在初始化时会尝试分配所用可用的GPU显存资源给自己，这在多人使用的服务器上工作就会导致GPU占用，别人无法使用GPU工作的情况。
        tf提供了两种控制GPU资源使用的方法，一是让tf在运行过程中动态申请显存，需要多少就申请多少，第二种方法就是吸纳之GPU的使用率。
1）动态申请显存
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    session=tf.Session(config=config)
2）限制GPU使用率
    config=tf.ConfigProto()
    config.gpu_options.per_process_gpu_momory_fraction=0.4   #占用40%显存
    session=tf.Session(config=config)
    或者
    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
    config=tf.ConfigProto(gpu_options=gpu_options)
    session=tf.Session(config=config)
3)设置使用哪块GPU
1. 在python程序中设置
    os.environ["CUDA_VISIBLE_DEVICES"]="0"   #使用GPU 0
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1"   # 使用GPU 0，1
2. 在执行python程序的时候
    CUDA_VISIBLE_DEVICES=0,1 python yourcode.py
'''
config = tf.ConfigProto()

config.gpu_options.allow_growth = True
#机器数量
config.gpu_options.visible_device_list = str(hvd.local_rank())
set_session(tf.Session(config=config))

parser = ArgumentParser()
parser.add_argument('--alg', type=str, default='ppo', help='The RL algorithm')
parser.add_argument('--env', type=str, default='PongNoFrameskip-v4', help='The game environment')
parser.add_argument('--num_steps', type=int, default=10000000, help='The number of total training steps')
parser.add_argument('--data_port', type=int, default=5000, help='Learner server port to receive training data')
parser.add_argument('--param_port', type=int, default=5001, help='Learner server to publish model parameters')
parser.add_argument('--model', type=str, default='accnn', help='Training model')
parser.add_argument('--pool_size', type=int, default=1280, help='The max length of data pool')
parser.add_argument('--training_freq', type=int, default=1,
                    help='How many receptions of new data are between each training, '
                         'which can be fractional to represent more than one training per reception')
parser.add_argument('--keep_training', type=bool, default=False,
                    help="No matter whether new data is received recently, keep training as long as the data is enough "
                         "and ignore `--training_freq`")
parser.add_argument('--batch_size', type=int, default=1280, help='The batch size for training')
parser.add_argument('--exp_path', type=str, default=None, help='Directory to save logging data and config file')
parser.add_argument('--config', type=str, default=None, help='The YAML configuration file')
parser.add_argument('--record_throughput_interval', type=int, default=10,
                    help='The time interval between each throughput record')
parser.add_argument('--num_envs', type=int, default=1, help='The number of environment copies')
parser.add_argument('--ckpt_save_freq', type=int, default=10, help='The number of updates between each weights saving')


def create_experiment_dir(args, prefix: str) -> None:
    if args.exp_path is None:
        args.exp_path = prefix + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
    args.exp_path = Path(args.exp_path)

    if args.exp_path.exists():
        raise FileExistsError(f'Experiment directory {str(args.exp_path)!r} already exists')

    args.exp_path.mkdir()

def main():
    # Parse input parameters
    args, unknown_args = parser.parse_known_args()
    args.num_steps = int(args.num_steps)
    unknown_args = parse_cmdline_kwargs(unknown_args)

    # Expose socket to actor(s)
    context = zmq.Context()
    weights_socket = context.socket(zmq.PUB)
    weights_socket.bind(f'tcp://*:{args.param_port}')
    #new 一个 agent
    agent = PPOAgent(ACCNNModel, [10, 20, 11], 4)

    # Configure experiment directory
    create_experiment_dir(args, 'LEARNER-')
    args.log_path = args.exp_path / 'log'
    args.ckpt_path = args.exp_path / 'ckpt'
    args.ckpt_path.mkdir()
    args.log_path.mkdir()

    logger.configure(str(args.log_path))

    # Record commit hash
    with open(args.exp_path / 'hash', 'w') as f:
        f.write(str(subprocess.run('git rev-parse HEAD'.split(), stdout=subprocess.PIPE).stdout.decode('utf-8')))

    # Variables to control the frequency of training
    '''
    multiprocessing模块
    multiprocessing模块是最常用的多进程模块。
    1、创建子进程
        （1）最基本的方法是通过函数：multiprocessing.Process(group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None)
        或者multiprocessing.Process子类化也可以。
        
        group为预留参数。
        target为可调用对象（函数对象），为子进程对应的活动；相当于multiprocessing.Process子类化中重写的run()方法。
        name为线程的名称，默认（None）为"Process-N"。
        args、kwargs为进程活动（target）的非关键字参数、关键字参数。
        deamon为bool值，表示是否为守护进程。
    另外还有几个子进程通用的函数：
        XXX.start() #启动进程活动（run())。XXX为进程实例。
        XXX.join(timeout = None) #使主调进程（包含XXX.join()语句的进程）阻塞，直至被调用进程XXX运行结束或超时（如指定timeout）。XXX为进程实例。
        def f(a, b = value):
            pass
        p = multiprocessing.Process(target = f, args = (a,), kwargs = {b : value}) 
        p.start()
        p.join()
    '''
    '''
    multiprocessing.Condition(lock = None) #条件锁，当条件触发时释放。
    其通过wait_for来条件阻塞，当条件满足时自动释放；也可用作类事件锁，通过wait阻塞，notify或notify_all释放。
    '''
    receiving_condition = multiprocessing.Condition()
    '''
    在使用tornado的多进程时，需要多个进程共享一个状态变量，于是考虑使用multiprocessing.Value
    对于共享整数或者单个字符，初始化比较简单，参照下图映射关系即可。如i = Value('i', 1), c = Value('c', '0')。
    第一个参数去的是首字母 如int->'i'
    '''
    num_receptions = multiprocessing.Value('i', 0)

    # Start memory pool in another process
    manager = MemPoolManager()
    manager.start()
    #内存池
    '''
    经典的内存池（MemPool）技术，是一种用于分配大量大小相同的小对象的技术。通过该技术可以极大加快内存分配/释放过程
    '''
    mem_pool = manager.MemPool(capacity=args.pool_size)
    '''
    创建进程
    '''
    Process(target=recv_data,
            args=(args.data_port, mem_pool, receiving_condition, num_receptions, args.keep_training)).start()

    # Print throughput statistics 打印吞吐量统计 也用一个进程记录吞吐量
    Process(target=MultiprocessingMemPool.record_throughput, args=(mem_pool, args.record_throughput_interval)).start()

    update = 0
    '''
    python中的【//】是算术运算符号，表示取整除，它会返回结果的整数部分，例如【print(7//2)】，输出结果为3。
    '''
    nupdates = args.num_steps // args.batch_size

    while True:
        # Do some updates
        agent.update_training(update, nupdates)
        #在内存池不大于batch_size的时候 如果参数keep_training为true 就在内存池中用batch_size来sample采样learn
        if len(mem_pool) >= args.batch_size:
            # keep_training是什么意思
            if args.keep_training:
                '''
                mem_pool的sample返回的不知道是什么 就是作为training_data 给agent训练
                learn返回值：每次都计算mean值 作为dict返回
                '''
                agent.learn(mem_pool.sample(size=args.batch_size))
            else:
                # 这一个小代码块不知道是什么意思
                with receiving_condition:
                    # num_receptions.value不够减去training_freq的话 就一直等 这里这俩参数不知道是什么意思
                    while num_receptions.value < args.training_freq:
                        receiving_condition.wait()
                    data = mem_pool.sample(size=args.batch_size)
                    num_receptions.value -= args.training_freq
                # Training
                # keep_training为false的时候就这样learn
                stat = agent.learn(data)
                #不为空的话就 记录state 为什么之前不用记录state再logger？
                if stat is not None:
                    for k, v in stat.items():
                        logger.record_tabular(k, v)
                logger.dump_tabular()

            # Sync weights to actor
            # 获取权重 权重应该是动态更新的
            weights = agent.get_weights()
            '''
            只在worker 0上发送
            
            Python提供了 pickle（泡菜） 模块来实现序列化。pickle模块常用的方法有：dumps、loads、dump、load
            pickle.dumps(obj) — 把 obj 对象序列化后以 bytes 对象返回，不写入文件
            pickle.loads(bytes_object) — 从 bytes 对象中读取一个反序列化对象，并返回其重组后的对象
            pickle.dump(obj , file) — 序列化对象，并将结果数据流写入到文件对象中
            '''
            if hvd.rank() == 0:
                weights_socket.send(pickle.dumps(weights))
            update += 1
            #ckpt_save_freq即需要update次数到了需要保存的时候的时候
            if update % args.ckpt_save_freq == 0:
                with open(args.ckpt_path / f'{args.alg}.{args.env}.ckpt',
                          'wb') as f:
                    pickle.dump(weights, f)

# 接收数据 再process中 作为 target方法重写 run方法
def recv_data(data_port, mem_pool, receiving_condition, num_receptions, keep_training):
    context = zmq.Context()
    data_socket = context.socket(zmq.REP)
    # 绑定到某端口
    data_socket.bind(f'tcp://*:{data_port}')
    #持续接收？
    while True:
        # noinspection PyTypeChecker
        # 端口中接收到的数据 反序列化(deserialize)
        data: dict = deserialize(data_socket.recv())
        data_socket.send(b'200')
        '''
        如果 keep_training==true 应该就是把接收的数据放到内存池
        否则 with receiving_condition(不知道是什么意思，应该是在锁中的意思 处理一些接收数据)：
            放入数据
            num_receptions变量+1
            释放锁
        '''
        if keep_training:
            mem_pool.push(data)
        else:
            with receiving_condition:
                mem_pool.push(data)
                #应
                num_receptions.value += 1
                # 这里notify好像是释放锁的意思
                receiving_condition.notify()


if __name__ == '__main__':
    main()
