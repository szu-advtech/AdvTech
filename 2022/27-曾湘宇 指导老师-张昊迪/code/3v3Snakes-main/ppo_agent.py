import inspect
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow.train import AdamOptimizer


class PPOAgent():
    def __init__(self, model_cls, observation_space, action_space, config=None,
                 gamma=0.99, lam=0.95, lr=2.5e-4, clip_range=0.1, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
                 epochs=4, nminibatches=4, *args, **kwargs):

        # Define parameters
        self.gamma = gamma
        self.lam = lam
        self.base_lr = self.lr = lr
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.epochs = epochs
        self.nminibatches = nminibatches

        # Default model config
        if config is None:
            config = {'model': [{'model_id': 'policy_model'}]}

        # Model related objects
        self.model = None
        self.sess = None
        self.train_op = None
        self.pg_loss = None
        self.vf_loss = None
        self.entropy = None
        self.clip_rate = None
        self.kl = None

        # Placeholder for training targets
        self.advantage_ph = tf.placeholder(dtype=tf.float32, shape=(None,))
        self.return_ph = tf.placeholder(dtype=tf.float32, shape=(None,))
        self.old_neglogp_ph = tf.placeholder(dtype=tf.float32, shape=(None,))
        self.old_v_ph = tf.placeholder(dtype=tf.float32, shape=(None,))
        self.lr_ph = tf.placeholder(dtype=tf.float32, shape=[])
        
        self.model_cls = model_cls
        self.observation_space = observation_space
        self.action_space = action_space

        if config is not None:
            self.load_config(config)

        self.model_instances = None
        self._init_model_instances(config)

        self.build()

    def build(self) -> None:
        #取最后一个实例
        self.model = self.model_instances[-1]
        # tf.reduce_mean()用于计算tensor(张量)沿着指定的数轴(即tensor的某一维度)上的平均值，用作降维或者计算tensor的平均值。
        self.entropy = tf.reduce_mean(self.model.entropy)
        '''
        tf.clip_by_value(1-y,1e-10,1.0)，这个 语句是在tensorflow实战Google深度学习框架中看见的，可以参看63页，运用的是交叉熵而不是二次代价函数。
        功能：可以将一个张量中的数值限制在一个范围之内。（可以避免一些运算错误:可以保证在进行log运算时，不会出现log0这样的错误或者大于1的概率）
            参数：（1）1-y：input数据（2）1e-10、1.0是对数据的限制。
            当1-y小于1e-10时，输出1e-10；
            当1-y大于1e-10小于1.0时，输出原值；
            当1-y大于1.0时，输出1.0；
        '''
        vpredclipped = self.old_v_ph + tf.clip_by_value(self.model.vf - self.old_v_ph, -self.clip_range,
                                                        self.clip_range)
        # Unclipped value
        vf_losses1 = tf.square(self.model.vf - self.return_ph)

        # Clipped value
        vf_losses2 = tf.square(vpredclipped - self.return_ph)
        self.vf_loss = 0.5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

        # Calculate ratio (pi current policy / pi old policy)
        ratio = tf.exp(self.old_neglogp_ph - self.model.neglogp_a)

        # Defining Loss = - J is equivalent to max J
        pg_losses = -self.advantage_ph * ratio
        pg_losses2 = -self.advantage_ph * tf.clip_by_value(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)

        # Final PG loss
        self.pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))

        # Total loss
        loss = self.pg_loss - self.entropy * self.ent_coef + self.vf_loss * self.vf_coef

        # Stat
        self.kl = tf.reduce_mean(self.model.neglogp_a - self.old_neglogp_ph)
        '''
        sess = tf.Session() 
        sess.run(tf.logical_or(True, False))  # True
        A = [True, True, False, False]
        B = [True, False, True, False]
        sess.run(tf.logical_or(A,B))  # array([ True,  True,  True, False])
        '''
        clipped = tf.logical_or(ratio > (1 + self.clip_range), ratio < (1 - self.clip_range))
        '''
        tf.cast()函数介绍和示例
        tf.cast(x, dtype, name=None)
        释义：数据类型转换
        
        x，输入张量
        dtype，转换数据类型
        name，名称
        '''
        self.clip_rate = tf.reduce_mean(tf.cast(clipped, tf.float32))
        #获取参数
        params = tf.trainable_variables(self.model.scope)
        #优化器
        trainer = tf.train.AdamOptimizer(learning_rate=self.lr_ph, epsilon=1e-5)
        '''
        构建一个函数y=w*x+b，其中[x ， b]作为参数列表传入，对于参数x，其值为50.0，其梯度为w，在此例中为10.0，因此返回值的第一个元素为(x的梯度，参数x)；
        同理对于参数b，其值为2.0，其梯度为1.0，因此返回值的第二个元素为(b的梯度，参数b)。
        '''
        grads_and_var = trainer.compute_gradients(loss, params)
        # grads_and_var应该是[(1,2),(3,4)]的格式 zip(*grads_and_var) 得到[(1,3),(2,4)]
        grads, var = zip(*grads_and_var)

        if self.max_grad_norm is not None:
            '''
            clip_by_global_norm作用：简单来说，就是利用梯度裁剪的方式避免梯度爆炸，“梯度爆炸”自己可查阅相关资料理解。
            给定张量t_list的元组或列表以及裁剪率clip_norm，此操作将返回裁剪后的list_clipped的张量列表以及t_list中所有张量的全局范数（global_norm）。 
            或者，如果您已经为t_list计算了全局范数，则可以使用use_norm指定全局范数。
            '''
            grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
        grads_and_var = list(zip(grads, var))
        # model.trainable_variables找到需要更新的变量，并用trainer.apply_gradients更新权重
        self.train_op = trainer.apply_gradients(grads_and_var)
        self.sess = self.model.sess
        # Initialize variables
        '''
        global_variables_initializer返回一个用来初始化计算图中所有global variable的op。
        这个op到底是啥，还不清楚。
        函数中调用了variable_initializer()和global_variables()
        global_variables()返回一个Variable list，里面保存的是gloabal variables。
        variable_initializer()将Variable list中的所有Variable取出来，将其variable.initializer属性做成一个op group。
        然后看Variable类的源码可以发现， variable.initializer就是一个assign op。
        所以： sess.run(tf.global_variables_initializer()) 就是run了所有global Variable的assign op，这就是初始化参数的本来面目。
        '''
        self.sess.run(tf.global_variables_initializer())

    def sample(self, state: Any, *args, **kwargs) -> Tuple[Any, dict]:
        action, value, neglogp = self.model.forward(state)
        return action, {'value': value, 'neglogp': neglogp}

    def learn(self, training_data, *args, **kwargs):
        '''
        从training data(键值对形式)中for循环取出每个key 用key从training_data中取出数据 作为learn的处理后的data
        '''
        data = [training_data[key] for key in ['state', 'return', 'action', 'value', 'neglogp', 'legal_action']]
        '''
        数据可能是多维的 data[0]得到有多少轮 data[i][j]能得到某轮某数据？
        '''
        nbatch = len(data[0])
        nbatch_train = nbatch // self.nminibatches

        inds = np.arange(nbatch)
        '''
        在字典中查找某个值时，若key不存在时则会返回一个KeyError错误而不是一个默认值，这时候可以使用defaultdict函数。
        注意：使用dict[key]=value时，若key不存在则报错；使用dict.get(key)时，若key不存在则会返回一个默认值。
        '''
        stats = defaultdict(list)
        for _ in range(self.epochs):
            np.random.shuffle(inds)
            # range(0, 30, 5)  # 步长为 5
            # [0, 5, 10, 15, 20, 25] 注意没有30
            # 这里为什么要设置步长呢？
            for start in range(0, nbatch, nbatch_train):
                end = start + nbatch_train
                # 取某一段 这段长度就是步长长度
                mbinds = inds[start:end]
                # for循环 每次从data中取出arr(每个arr就是一个nbatch的数据) 再从arr中取出上面那一段长度的数据
                # for循环取数据之后 组合作为slices
                slices = (arr[mbinds] for arr in data)
                '''
                def train(arg1,arg2,arg3):
                    print(f"arg1:{arg1},arg2:{arg2},arg3:{arg3}")
                list_ = [[1,2,3],[4,5,6],[7,8,9]]
                train(*list_)
                #res:  arg1:[1, 2, 3],arg2:[4, 5, 6],arg3:[7, 8, 9]
                '''
                '''
                ret:    {
                        'pg_loss': pg_loss,
                        'vf_loss': vf_loss,
                        'entropy': entropy,
                        'clip_rate': clip_rate,
                        'kl': kl
                        }
                '''
                ret = self.train(*slices)
                # 新的stats补充到stats
                for k, v in ret.items():
                    stats[k].append(v)
        # 每次都计算mean值 作为dict返回
        return {k: np.array(v).mean() for k, v in stats.items()}

    def train(self, obs, returns, actions, values, neglogps, legal_action):
        advs = returns - values
        # advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        td_map = {
            self.model.x_ph: obs,
            self.model.a_ph: actions,
            self.advantage_ph: advs,
            self.return_ph: returns,
            self.lr_ph: self.lr,
            self.old_neglogp_ph: neglogps,
            self.old_v_ph: values,
            self.model.legal_action: legal_action
        }
        # 不知道为什么可以返回这么多东西 target的run是重定义了？
        _, pg_loss, vf_loss, entropy, clip_rate, kl = self.sess.run(
            [self.train_op, self.pg_loss, self.vf_loss, self.entropy, self.clip_rate, self.kl], td_map)
        return {
            'pg_loss': pg_loss,
            'vf_loss': vf_loss,
            'entropy': entropy,
            'clip_rate': clip_rate,
            'kl': kl
        }

    def prepare_training_data(self, trajectory: List[Tuple[Any, Any, Any, Any, Any, dict]]) -> Dict[str, np.ndarray]:
        mb_states, mb_actions, mb_rewards, mb_dones, next_state, mb_extras = trajectory
        mb_values = np.asarray([extra_data['value'] for extra_data in mb_extras])
        mb_neglogp = np.asarray([extra_data['neglogp'] for extra_data in mb_extras])
        mb_legalac = np.asarray([extra_data['legal_action'] for extra_data in mb_extras])

        last_values = self.model.forward(next_state)[1]
        mb_values = np.concatenate([mb_values, last_values[np.newaxis]])

        mb_deltas = mb_rewards + self.gamma * mb_values[1:] * (1.0 - mb_dones) - mb_values[:-1]

        nsteps = len(mb_states)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(nsteps)):
            nextnonterminal = 1.0 - mb_dones[t]
            mb_advs[t] = lastgaelam = mb_deltas[t] + self.gamma * self.lam * nextnonterminal * lastgaelam

        mb_returns = mb_advs + mb_values[:-1]
        data = [sf01(arr) for arr in [mb_states, mb_returns, mb_actions, mb_values[:-1], mb_neglogp, mb_legalac]]
        name = ['state', 'return', 'action', 'value', 'neglogp', 'legal_action']
        return dict(zip(name, data))

    def post_process_training_data(self, training_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return training_data

    def set_weights(self, weights, *args, **kwargs) -> None:
        self.model.set_weights(weights)

    def get_weights(self, *args, **kwargs) -> Any:
        return self.model.get_weights()

    def save(self, path: Path, *args, **kwargs) -> None:
        self.model.save(path)

    def load(self, path: Path, *args, **kwargs) -> None:
        self.model.load(path)

    def preprocess(self, state: Any, *args, **kwargs) -> Any:
        pass

    def update_sampling(self, current_step: int, total_steps: int, *args, **kwargs) -> None:
        pass
    # pass是为什么
    def update_training(self, current_step: int, total_steps: int, *args, **kwargs) -> None:
        pass
    '''
    union() 取并集，效果等同于 | ，重复元素只会出现一次，但是括号里可以是 list，tuple，其他 ， 甚至是 dict
    '''
    def _init_model_instances(self, config: Union[dict, None]) -> None:
        """Initialize model instances"""
        self.model_instances = []

        def create_model_instance(_c: dict):
            ret = {}
            for k, v in _c.items():
                if k in valid_config:
                    ret[k] = v
                else:
                    warnings.warn(f"Invalid config item '{k}' ignored", RuntimeWarning)
            self.model_instances.append(self.model_cls(self.observation_space, self.action_space, **ret))

        if config is not None and 'model' in config:
            model_config = config['model']
            valid_config = get_config_params(self.model_cls)
            # 如果model_config是一个list
            if isinstance(model_config, list):
                for _, c in enumerate(model_config):
                    #就一个个创建实例
                    create_model_instance(c)
            elif isinstance(model_config, dict):
                # 如果是dict(list中的单个元素) 就创建一次即可
                create_model_instance(model_config)
        # 这里不知道是什么意思
        else:
            self.model_instances.append(self.model_cls(self.observation_space, self.action_space))


def get_config_params(obj_or_cls) -> List[str]:
    """
    Return configurable parameters in 'Agent.__init__' and 'Model.__init__' which appear after 'config'
    :param obj_or_cls: An instance of 'Agent' / 'Model' OR their corresponding classes (NOT base classes)
    :return: A list of configurable parameters
    """
    '''
    inspect.signature（fn）.parameters获取函数参数的参数名，参数的属性，参数的默认值
    import inspect
        def foo(a,b=1,*c,d,**kw):
            pass
        
        # 获取函数参数返回一个有序字典
        parms = inspect.signature(foo).parameters
        print(parms)
        # 
        # 获取参数名，参数属性，参数默认值
        for name,parm in parms.items():
            print(name,parm.kind,parm.default)
    '''
    sig = list(inspect.signature(obj_or_cls.__init__).parameters.keys())

    config_params = []
    config_part = False
    for param in sig:
        if param == 'config':
            # Following parameters should be what we want
            config_part = True
        elif param in {'args', 'kwargs'}:
            pass
        elif config_part:
            config_params.append(param)
    # 拿到config之后的参数
    return config_params

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
