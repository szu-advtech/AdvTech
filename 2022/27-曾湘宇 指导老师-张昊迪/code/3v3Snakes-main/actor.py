import datetime
import os
import pickle
import subprocess
import time
from argparse import ArgumentParser
from itertools import count
from multiprocessing import Array, Process
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import zmq
from pyarrow import serialize

from env import SnakeEnv
from utils import logger
from utils.cmdline import parse_cmdline_kwargs

gamma = 0.99
lam = 0.95

from custom_model import ACCNNModel
'''
通过命令行运行Python脚本时，可以通过ArgumentParser来高效地接受并解析命令行参数。
流程
新建一个ArgumentParser类对象，然后来添加若干个参数选项，最后通过parse_args()方法解析并获得命令行传来的参数。
    import argparser
    parser = argparser.ArgumentParser()
    # 此处省略添加若干个参数选项的详细步骤
    # ...
    parser.parse_args()
最后通过parser.<argument_name>来获取传递过来的参数。
添加参数选项
使用add_argument()来添加参数选项
    # 添加位置参数
    parser.add_argument("echo", help="echo the string you use here")
    parser.add_argument("square", help="display ...", type=int)
    # 添加可选参数
    parser.add_argument("-v", "--verbosity", help="...", type=int, choices=[0, 1, 2]， default=0)
对以上代码做出如下解释：
在使用add_argument来添加参数选项的时候，首先要指定参数的名字argument_name这个属性，可选参数有长短两个名称；
在命令行指定位置参数时直接传值，指定可选参数时，先注明长短名称，然后在后面接值；
help提示参数的作用，type规定了参数的取值类型，choices以列表的形式规定了值域，default规定了参数的默认值

参数选项组
使用add_mutually_exclusive_group()来添加相互对立的参数选项组
一个对立的可选参数组在指定参数时，只能任选其一或都不选
    # 导入模块和新建ArgumentParser类的过程省略
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-d", "--down")
    group.add_argument("-t", "--top")
    # ...
最后获取参数时，仍旧是通过parser.down和parser.top.

额外的小插曲
对于可选参数还有一个action属性，常见的有store_true和count两种
    # 指定-v可选参数时，-v等于True，否则为False
    parser.add_argument("-v", action="store_true")
    # 指定-v可选参数时，-v等于v出现的次数
    parser.add_argument("-v", action="count")
'''
parser = ArgumentParser()
parser.add_argument('--ip', type=str, default='127.0.0.1', help='IP address of learner server')
parser.add_argument('--data_port', type=int, default=5000, help='Learner server port to send training data')
parser.add_argument('--param_port', type=int, default=5001, help='Learner server port to subscribe model parameters')
parser.add_argument('--num_replicas', type=int, default=1, help='The number of actors')
parser.add_argument('--max_steps_per_update', type=int, default=128,
                    help='The maximum number of steps between each update')
parser.add_argument('--exp_path', type=str, default=None,
                    help='Directory to save logging data, model parameters and config file')
parser.add_argument('--num_saved_ckpt', type=int, default=10, help='Number of recent checkpoint files to be saved')


def create_experiment_dir(args, prefix: str) -> None:
    if args.exp_path is None:
        args.exp_path = prefix + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
    '''
    pathlib提供path对象来操作，包括目录和文件。
        from pathlib import Path
        p =Path()     #输出格式。PosixPath('.')
        p =Path('a','b','c/d')  #输出格式PosixPath('a/b/c/d')
        p =Path('/etc')    #PosixPath('/etc')
    '''
    args.exp_path = Path(args.exp_path)

    if args.exp_path.exists():
        raise FileExistsError(f'Experiment directory {str(args.exp_path)!r} already exists')
    # 如果已经存在就报错 不存在就创建
    args.exp_path.mkdir()

def run_one_agent(index, args, unknown_args, actor_status):
    import tensorflow.compat.v1 as tf
    from tensorflow.keras.backend import set_session

    # Set 'allow_growth'
    '''
    tf.ConfigProto一般用在创建session的时候用来对session进行参数配置。
    tf.ConfigProto()的参数
    log_device_placement = True : 是否打印设备分配日志
    allow_soft_placement = True : 如果你指定的设备不存在，允许TF自动分配设备
    示例代码：
       tf.ConfigProto(log_device_placement = True, allow_soft_placement = True)
    记录设备指派情况 :  tf.ConfigProto(log_device_placement=True)
        设置tf.ConfigProto()中参数log_device_placement = True ,可以获取到 operations 和 Tensor 被指派到哪个设备(几号CPU或几号GPU)上运行,会在终端打印出各项操作是在哪个设备上运行的。
    自动选择运行设备 ： tf.ConfigProto(allow_soft_placement=True)
        在tf中，通过命令 "with tf.device('/cpu:0'):",允许手动设置操作运行的设备。如果手动设置的设备不存在或者不可用，就会导致tf程序等待或异常，为了防止这种情况，可以设置tf.ConfigProto()中参数allow_soft_placement=True，允许tf自动选择一个存在并且可用的设备来运行操作。
    限制GPU资源使用：
        为了加快运行效率，TensorFlow在初始化时会尝试分配所有可用的GPU显存资源给自己，这在多人使用的服务器上工作就会导致GPU占用，别人无法使用GPU工作的情况。
        tf提供了两种控制GPU资源使用的方法，一是让TensorFlow在运行过程中动态申请显存，需要多少就申请多少;第二种方式就是限制GPU的使用率。
    1.动态申请显存:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
    2.限制GPU使用率:
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.4  #占用40%GPU显存
        session = tf.Session(config=config)
    或者：
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
        config=tf.ConfigProto(gpu_options=gpu_options)
        session = tf.Session(config=config)
    指定GPU：
    1.在python程序中设置：
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '0' #使用 GPU 0
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' # 使用 GPU 0，1
    2.在执行python程序时指定：
    这样在跑你的网络之前，告诉程序只能看到0,1号GPU，其他的GPU它不可见
        CUDA_VISIBLE_DEVICES=0,1 python python_filename.py
    '''
    '''
    tf.Session：创建一个新的TensorFlow会话。
    tf.Session(self, target='', graph=None, config=None)
    如果在构造会话时未指定`graph`参数，则默认图将在会话中启动。 如果在同一过程中使用多个图（通过tf.Graph（）创建），则每个图必须使用不同的会话，但是每个图可以在多个会话中使用。 在这种情况下，将要显式启动的图传递给会话构造函数通常更为清晰。
    Args:
    target：（可选）要连接的执行引擎。 默认为使用进程内引擎。 
    graph：（可选）要启动的“图”。
    config ：（可选）一个[ConfigProto`]协议缓冲区，带有会话的配置选项。
    '''
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    # Connect to learner
    '''
    zmq(ZeroMQ)是一个可嵌入的网络库，作用就像是一个并发框架。 是因为这个框架想体现自己的“零代理” “零延时（尽可能零延时）”
        应答服务端：有的博主把这一端比喻成老师，负责回应
        简易版的服务器
        请求-应答模式，它对应适用于RPC(远程过程调用)和传统的客户端/服务器模型
        在这个情况下，如果终止服务器并且重新启动它，客户端是无法正常恢复的
            import zmq
            import time
            context=zmq.Context()
            # REP是应答方
            socket=context.socket(zmq.REP)
            socket.bind('tcp://127.0.0.1:5559')
            while True:
                message=socket.recv()
                print("Received request: ", message)
                time.sleep(1)
                socket.send("World".encode())
        请求-客户端：对应教师，这一端被比喻为学生，负责提问
            import zmq
            import time
            context=zmq.Context(
            print("Connecting to hello world server..."  )
            # REQ 是提问方
            socket=context.socket(zmq.REQ)
            socket.connect("tcp://127.0.0.1:5559")
            for request in range(10):
                print("Sending request ", request, "...")
                socket.send('Hello'.encode())
                message=socket.recv()
                print("Received reply ", request, "[", message, "]")
    '''
    context = zmq.Context() #这句话暂时不知是什么意思
    context.linger = 0  # For removing linger behavior
    socket = context.socket(zmq.REQ)
    socket.connect(f'tcp://{args.ip}:{args.data_port}')

    # Configure logging only in one process
    if index == 0:
        logger.configure(str(args.log_path))

    # Initialize values
    model_id = -1
    num_episode = 0

    model = ACCNNModel(observation_space = [10,20,11], action_space = 4)
    env = SnakeEnv('snakes-3v3')

    while True:
        num_episode += 1

        mb_states, mb_actions, mb_rewards, mb_dones, mb_values, mb_neglogp, mb_legalac = [], [], [], [], [], [], []
        done = False
        # 得到观测矩阵和info（包括legal action）
        next_state, next_info = env.reset()
        while not done:
            # Sample action
            state = next_state
            info = next_info
            #v p是什么？
            action, v, p = model.forward(state, info["legal_action"])
            # 得到更新后的结果
            next_state, reward, done, next_info = env.step(action)
            # 如果done(结束)
            if done:
                # 输出done的结果 长度等
                print("length:",info['length']," ave:",np.mean(info["length"]))
                # 记录结果
                logger.record_tabular("ave_rl", np.mean(info["length"]))
            # 应该是累计结果
            mb_states.append(state)
            mb_actions.append(action)
            mb_rewards.append(reward)
            mb_dones.append(done)
            mb_values.append(v)
            mb_neglogp.append(p)
            mb_legalac.append(info["legal_action"])

        # 将list转为array形式
        mb_states = np.asarray(mb_states, dtype=state.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogp = np.asarray(mb_neglogp, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        mb_legalac = np.asarray(mb_legalac, dtype=np.bool)
        #记录每个蛇的结果？
        mb_s, mb_a, mb_r, mb_d, mb_ex = [], [], [], [], []
        for j in range(6):
            for i in range(len(mb_states)):
                mb_s.append([mb_states[i][j]])
                mb_a.append([mb_actions[i][j]])
                mb_r.append([mb_rewards[i][j]])
                mb_d.append([mb_dones[i]])
                d = {}
                d['value'] = [mb_values[i][j]]
                d['neglogp'] = [mb_neglogp[i][j]]
                d['legal_action'] = [mb_legalac[i][j]]
                mb_ex.append(d)
        #转换为array
        mb_s = np.asarray(mb_s, dtype=state.dtype)
        mb_r = np.asarray(mb_r, dtype=np.float32)
        mb_a = np.asarray(mb_a)
        mb_d = np.asarray(mb_d, dtype=np.bool)
        # dict{}形式的data
        data = prepare_training_data([mb_s, mb_a, mb_r, mb_d, state, mb_ex])
        '''
        序列化是将对象状态转换为可保持或传输的格式的过程。与序列化相对的是反序列化，
        它将流转换为对象。这两个过程结合起来，可以轻松地存储和传输数据。
        序列化
        由于存在于内存中的对象都是暂时的，无法长期驻存，为了把对象的状态保持下来，这时需要把对象写入到磁盘或者其他介质中，这个过程就叫做序列化。
        Python中,socket用来实现网络通信,它默认的recv是一个阻塞的函数,也就是说,当运行到recv时,会在这个位置一直等待直到有数据传输过来,我在网上一篇文章看到:
        '''
        socket.send(serialize(data).to_buffer())
        socket.recv()

        # Log information 不知道方法是什么意思 应该就是保存日志
        logger.record_tabular("episodes", num_episode)
        logger.dump_tabular()

        # Update weights
        new_weights, model_id = find_new_weights(model_id, args.ckpt_path)
        if new_weights is not None:
            model.set_weights(new_weights)



def sf01(arr):
    """
    swap and then flatten axes 0 and 1
        import numpy as np
        arr = np.arange(20).reshape(4,5)
        ##建立了一个4行5列的矩阵

        ##索引取其中的第4、3、1、2行
        arr[[3,2,0,1]]

        ##获取其转置矩阵
        arr.T
        ##变成了5行4列的矩阵

    当我们开始遇到多维数组时，题目中提到的transpose和swapaxes就开始发挥作用了。
        言归正传，对于transpose来说，会将多维数组的轴编号，也就是给各个轴建立索引，我们自设一个三维数组，维数为（2，2，4），有三个轴：
        arr = np.arange(16).reshape(2,2,4)
        当我们进行变换时，有 arr.transpose(2,1,0),这里就是让索引2变换到了索引0的位置，索引0变到了索引2的位置，索引1保持不变，根据索引的变动，形状也发生相关位置的变化，如下：[4,2,2]
    swapaxes 对轴进行两两置换
        理解了上面的transpose，应该再理解swapaxes就不难了。swapaxes实际上也是针对轴索引进行变化，区别就在于transpose可以一次搞三个，但是swapaxes只能两两置换。
        对于swapaxes来说，括号内的两参数，交换位置和不交换，实际结果相同。
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


def prepare_training_data(trajectory):
    #将数据作为轨迹
    mb_states, mb_actions, mb_rewards, mb_dones, next_state, mb_extras = trajectory
    #前面不是已经asarray了嘛 这里为什么还要？？
    mb_values = np.asarray([extra_data['value'] for extra_data in mb_extras])
    mb_neglogp = np.asarray([extra_data['neglogp'] for extra_data in mb_extras])
    mb_legalac = np.asarray([extra_data['legal_action'] for extra_data in mb_extras])
    #这是什么意思,应该是二维数组的意思
    last_values = [[0]]
    mb_values = np.concatenate([mb_values, last_values])
    #广义优势估计( Gae ) 的 一部分
    mb_deltas = mb_rewards + gamma * mb_values[1:] * (1.0 - mb_dones) - mb_values[:-1]

    nsteps = len(mb_states)
    mb_advs = np.zeros_like(mb_rewards)
    lastgaelam = 0
    # 完整的 广义优势估计( Gae )
    for t in reversed(range(nsteps)):
        nextnonterminal = 1.0 - mb_dones[t]
        mb_advs[t] = lastgaelam = mb_deltas[t] + gamma * lam * nextnonterminal * lastgaelam

    mb_returns = mb_advs + mb_values[:-1]
    #数据处理 不太看得懂
    data = [sf01(arr) for arr in [mb_states, mb_returns, mb_actions, mb_values[:-1], mb_neglogp, mb_legalac]]
    name = ['state', 'return', 'action', 'value', 'neglogp', 'legal_action']
    '''
    zip函数：接受任意多个（包括0个和1个）序列作为参数，返回一个tuple列表。
        1.
        x = [1, 2, 3]
        y = [4, 5, 6]
        z = [7, 8, 9]
        xyz = zip(x, y, z)
        print xyz
        运行的结果是：
        [(1, 4, 7), (2, 5, 8), (3, 6, 9)]
        2.
        x = [1, 2, 3]
        x = zip(x)
        print x
        运行的结果是：
        [(1,), (2,), (3,)]
        从这个结果可以看出zip函数在只有一个参数时运作的方式
        3.
        x = [1, 2, 3]
        y = [4, 5, 6]
        z = [7, 8, 9]
        xyz = zip(x, y, z)
        u = zip(*xyz)
        print u
        运行的结果是：
        [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
        4.
        x = [1, 2, 3]
        r = zip(* [x] * 3)
        print r
        运行的结果是：
        [(1, 1, 1), (2, 2, 2), (3, 3, 3)]
        5.
        key = 'abcde'
        value = range(1, 6)
        res = dict(zip(key, value))
        print res
        运行的结果是：
        {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
    '''
    # 变成一一对应字典的形式 返回
    return dict(zip(name, data))


def run_weights_subscriber(args, actor_status):
    """Subscribe weights from Learner and save them locally"""
    '''
    context不确定是什么意思
    https://blog.csdn.net/biheyu828/article/details/88932826
    '''
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(f'tcp://{args.ip}:{args.param_port}')
    socket.setsockopt_string(zmq.SUBSCRIBE, '')  # Subscribe everything

    for model_id in count(1):  # Starts from 1
        while True:
            try:
                # 从learner接收信息
                weights = socket.recv(flags=zmq.NOBLOCK)

                # Weights received
                with open(args.ckpt_path / f'{model_id}.{args.alg}.{args.env}.ckpt', 'wb') as f:
                    f.write(weights)
                # 为何要移除？、
                if model_id > args.num_saved_ckpt:
                    os.remove(args.ckpt_path / f'{model_id - args.num_saved_ckpt}.{args.alg}.{args.env}.ckpt')
                break
            except zmq.Again:
                pass

            if all(actor_status):
                # All actors finished works
                return

            # For not cpu-intensive
            time.sleep(1)


def find_new_weights(current_model_id: int, ckpt_path: Path) -> Tuple[Any, int]:
    try:
        '''
        os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
        它不包括 . 和 .. 即使它在文件夹中。
        只支持在 Unix, Windows 下使用。
                import os, sys
                # 打开文件
                path = "/var/www/html/"
                dirs = os.listdir( path )
                # 输出所有文件和文件夹
                for file in dirs:
                   print (file)
        '''
        '''
        sorted(iterable[, key][, reverse]
            从 iterable 中的项目返回新的排序列表。
            有两个可选参数，必须指定为关键字参数。
            key 指定一个参数的函数，用于从每个列表元素中提取比较键：key=str.lower。默认值为 None （直接比较元素）。
            reverse 是一个布尔值。如果设置为 True，那么列表元素将按照每个比较反转进行排序。
        示例：创建由元组构成的列表：a = [('b',3), ('a',2), ('d',4), ('c',1)]
            按照第一个元素排序
              sorted(a, key=lambda x:x[0])  
              >>> [('a',2),('b',3),('c',1),('d',4)]
            按照第二个元素排序
              sorted(a, key=lambda x:x[1]) 
              >>> [('c',1),('a',2),('b',3),('d',4)]
        '''
        ckpt_files = sorted(os.listdir(ckpt_path), key=lambda p: int(p.split('.')[0]))
        latest_file = ckpt_files[-1]
    except IndexError:
        # No checkpoint file
        return None, -1
    new_model_id = int(latest_file.split('.')[0])

    if int(new_model_id) > current_model_id:
        loaded = False
        while not loaded:
            try:
                with open(ckpt_path / latest_file, 'rb') as f:
                    '''
                    通过pickle模块的序列化操作pickle.dump(obj, file, [,protocol])，我们能够将程序中运行的对象信息保存到文件中去，永久存储。
                    通过pickle模块的反序列化操作pickle.load(file)，我们能从文件中创建上一次程序保存的对象
                    
                    pickle模块只能在python中使用，python中几乎所有的数据类型（列表，字典，集合，类等）都可以用pickle来序列化，
                    pickle序列化后的数据，可读性差，人一般无法识别。
                    '''
                    new_weights = pickle.load(f)
                    if int(new_model_id) % 100 == 0:
                        with open( '/home/wangjt/snakes/rl-framework-baseline-zhaoj2/base_model_backup_1_2000_data1223/' f'{int(new_model_id)}.pth',
                                    'wb') as f:
                                pickle.dump(new_weights, f)
                loaded = True
            except (EOFError, pickle.UnpicklingError):
                # The file of weights does not finish writing
                pass

        return new_weights, new_model_id
    else:
        return None, current_model_id


def main():
    # Parse input parameters
    '''
    有时间一个脚本只需要解析所有命令行参数中的一小部分，剩下的命令行参数给两一个脚本或者程序。
    在这种情况下，parse_known_args()就很有用。
    它很像parse_args()，但是它在接受到多余的命令行参数时不报错。
    相反的，返回一个tuple类型的命名空间和一个保存着余下的命令行字符的list。
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--flag_int',
            type=float,
            default=0.01,
            help='flag_int.'
        )
        FLAGS, unparsed = parser.parse_known_args()
        print(FLAGS)
        print(unparsed)
    结果：
    $ python prog.py --flag_int 0.02 --double 0.03 a 1
    Namespace(flag_int=0.02)
    ['--double', '0.03', 'a', '1']
    '''
    args, unknown_args = parser.parse_known_args()
    args.num_steps = int(args.num_steps)
    # 返回键值对形式 即 其他未使用的输入参数
    unknown_args = parse_cmdline_kwargs(unknown_args)

    # Create experiment directory
    create_experiment_dir(args, 'ACTOR-')
    # 构建更多存储信息的dir_path
    args.ckpt_path = args.exp_path / 'ckpt'
    args.log_path = args.exp_path / 'log'
    args.ckpt_path.mkdir()
    args.log_path.mkdir()

    # Record commit hash 记录提交信息
    '''
    subprocess可以帮我们执行命令，获取执行结果及返回内容。
    1、subprocess.run()
    此方法为python3.5版本后的推荐方法，可以获取执行结果、返回内容等一些常用的信息， 满足大部分开发需要。
    subprocess.run(args, *, stdin=None, input=None, stdout=None, stderr=None, capture_output=False, shell=False, cwd=None, timeout=None, check=False, encoding=None, errors=None, text=None, env=None, universal_newlines=None)
        args： 要执行的命令。类型为str（如 “ls -l”）或包含str的list，tuple等（如 [“ls”, “-l”]）, 推荐使用list形式，如果传入的args为str且包含参数，则 必须shell=True，默认为False。
        **stdin、stdout、stderr： ** 子进程的标准输入、输出、错误，常用的为stdout，我们可以获取命令执行后的输出内容。
        **shell：**如果该参数为 True，将通过操作系统的 shell 执行指定的命令，默认为False。
        **timeout：**设置命令超时时间。如果命令执行时间超时，子进程将被杀死，并弹出 TimeoutExpired 异常。
        **check：**如果该参数设置为 True，并且进程退出状态码不是 0，则弹 出 CalledProcessError 异常。
        encoding: 如果指定了该参数，则 stdin、stdout 和 stderr 可以接收字符串数据，并以该编码方式编码。否则只接收 bytes 类型的数据。
        capture_output： 设置为True，将捕获stdout和stderr，从而获执行命令后取返回的内容。
    subprocess.getoutput()
        该方法可以直接获取命令执行后的输出内容，返回值为str
    '''
    with open(args.exp_path / 'hash', 'w') as f:
        f.write(str(subprocess.run('git rev-parse HEAD'.split(), stdout=subprocess.PIPE).stdout.decode('utf-8')))

    # Running status of actors actors的运行状态
    '''
    Python [0] * n 描述的意思:list * int 意思是将数组重复 int 次并依次连接形成一个新数组
    Array不知道是什么意思
    '''
    actor_status = Array('i', [0] * args.num_replicas)

    # Run weights subscriber
    '''
    虽然使用 os.fork() 方法可以启动多个进程，但这种方式显然不适合 Windows，而 Python 是跨平台的语言，所以 Python 绝不能仅仅局限于 Windows 系统，因此 Python 也提供了其他方式在 Windows 下创建新进程。
    Python 在 multiprocessing 模块下提供了 Process 来创建新进程。与 Thread 类似的是，使用 Process 创建新进程也有两种方式：
    以指定函数作为 target，创建 Process 对象即可创建新进程。
     定义一个普通的action函数（target），该函数准备作为进程执行体
     和使用 thread 类创建子线程的方式非常类似，使用 Process 类创建实例化对象，其本质是调用该类的构造方法创建新进程。Process 类的构造方法格式如下：
        def __init__(self,group=None,target=None,name=None,args=(),kwargs={})
        其中，各个参数的含义为：
        group：该参数未进行实现，不需要传参；
        target：为新建进程指定执行任务，也就是指定一个函数；
        name：为新建进程设置名称；
        args：为 target 参数指定的参数传递非关键字参数；
        kwargs：为 target 参数指定的参数传递关键字参数。
            from multiprocessing import Process
                import os
                print("当前进程ID：",os.getpid())
                
                # 定义一个函数，准备作为新进程的 target 参数
                def action(name,*add):
                    print(name)
                    for arc in add:
                        print("%s --当前进程%d" % (arc,os.getpid()))
                if __name__=='__main__':
                    #定义为进程方法传入的参数
                    my_tuple = ("http://c.biancheng.net/python/",\
                                "http://c.biancheng.net/shell/",\
                                "http://c.biancheng.net/java/")
                    #创建子进程，执行 action() 函数
                    my_process = Process(target = action, args = ("my_process进程",*my_tuple))
                    #启动子进程
                    my_process.start()
                    #主进程执行该函数
                    action("主进程",*my_tuple)
    '''
    # subscriber是主进程？
    subscriber = Process(target=run_weights_subscriber, args=(args, actor_status))
    subscriber.start()

    def exit_wrapper(index, *x, **kw):
        """Exit all agents on KeyboardInterrupt (Ctrl-C)"""
        try:
            run_one_agent(index, *x, **kw)
        except KeyboardInterrupt:
            if index == 0:
                for _i, _p in enumerate(agents):
                    if _i != index:
                        _p.terminate()
                    actor_status[_i] = 1

    agents = []
    # 为什么这里要直接执行exit？
    for i in range(args.num_replicas):
        p = Process(target=exit_wrapper, args=(i, args, unknown_args, actor_status))
        p.start()
        os.system(f'taskset -p -c {(i+0) % os.cpu_count()} {p.pid}')  # For CPU affinity

        agents.append(p)
    '''
    一 Process对象的join方法
        在主进程运行过程中如果想并发地执行其他的任务，我们可以开启子进程，此时主进程的任务与子进程的任务分两种情况
        情况一：
        在主进程的任务与子进程的任务彼此独立的情况下，主进程的任务先执行完毕后，主进程还需要等待子进程执行完毕，然后统一回收资源。 这种是没有join方法
        情况二：
        如果主进程的任务在执行到某一个阶段时，需要等待子进程执行完毕后才能继续执行，
        就需要有一种机制能够让主进程检测子进程是否运行完毕，在子进程执行完毕后才继续执行，否则一直在原地阻塞，这就是join方法的作用
        让主进程等着，所有子进程执行完毕后，主进程才继续执行
        p.join()是让谁等？
        很明显p.join()是让主线程等待p 子进程的结束，卡住的是主进程而绝非 子进程p，
    '''
    for agent in agents:
        agent.join()

    subscriber.join()


if __name__ == '__main__':
    main()
