import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
tf.reset_default_graph()
#import numba as nb

#for reproducible
np.random.seed(1)
tf.set_random_seed(1)

class PolicyGradient:
    def __init__(self, n_features, n_actions, learning_rate=0.001, reward_decay=0.99, output_graph=False):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        self._build_net()
        self.sess = tf.Session()
        if output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
    '''
    基于 TensorFlow 的神经网络，用于预测行动的概率分布。Pnet
    '''
    def _build_net(self):
        #输入：state，action，value，占位符设置类型
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")
        '''
        定义两层网络
        '''
        #fc1 'batch normalization if any'
        layer = tf.layers.dense(
            inputs=self.tf_obs,
            units=20,
            activation=tf.nn.tanh,  # tanh activation
            name='fc1'
        )
        #fc2
        all_act = tf.layers.dense(
            inputs=layer,
            units=self.n_actions,
            activation=None,
            name='fc2'
        )
        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')  # use softmax to convert to probability

        with tf.name_scope('loss'):
            neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
    '''
    计算出所有行动的概率,根据行动的概率随机选择一个行动
    输入：observation
    输出：action
    '''
    def pro_choose_action(self, observation):
        #调用了sess.run函数，并将观察结果作为输入，计算出所有行动的概率
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation})

        #使用了numpy库中的random.choice函数，根据行动的概率随机选择一个行动。
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action
    '''
    计算出所有行动的概率,根据行动的概率,选择最大的一个进行行动
    输入：observation
    输出：action
    它使用了numpy库中的argmax函数，选择概率最大的行动作为最终决策。
    '''
    def fix_choose_action(self, observation): #choose an action w.r.t max probability
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation})
        #print('fix_choose_action', prob_weights.ravel())
        action = np.argmax(prob_weights.ravel())
        return action
    '''
    记录agent在环境中的状态、行动和reward。
    并相应的记录在三个列表中
    '''
    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)
    '''
    训练智能体的策略网络
    '''
    def learn(self):
        # discount and normalize episode reward
        #计算agent在本轮训练中的reward。具体来说，它将每一步的回报进行折扣，并将所有回报进行归一化，以便训练模型。
        discounted_ep_rs_norm = self._discount_rewards()

        # train on episode
        #使用TensorFlow的Session.run函数，通过梯度下降算法调整策略网络的参数，
        #以便让策略网络尽可能地预测出智能体在本轮训练中的最优策略
        self.sess.run(self.train_op, feed_dict={
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             self.tf_vt: np.array(discounted_ep_rs_norm) # shape=[None, ]
        })
        #数据清空之后，learn函数就结束了。该函数会返回本轮训练中agent的reward，然后强化学习算法会进入下一轮训练。
        # 在下一轮训练中，agent会在新的环境中采取新的行动，并获得新的回报，然后算法会再次调用learn函数，进行新的训练。
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data
        
        return discounted_ep_rs_norm
    '''
    计算agent在一个训练轮次中的rewards
    '''
    # 幂函数次幂
    def reward_fn(self,reward):
        exponent = 3
        return reward ** exponent

    def _discount_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        #使用一个循环，从后向前遍历每一步的reward，并将每一步的reward与折扣因子（gamma）相乘
        #然后将结果与之前的reward累加，得到折扣后的回报
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.reward_fn(self.ep_rs[t])
            discounted_ep_rs[t] = running_add
        # normalize episode rewards
        #将折扣后的reward进行归一化。
        #这个步骤的目的是为了让不同训练轮次中智能体的回报具有相同的量纲，以便训练模型。
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs
    '''
    用于保存当前训练的强化学习模型
    '''
    def save(self, checkpoint):
        #使用TensorFlow的Saver类创建一个Saver对象。该对象用于将当前的模型参数保存。
        saver = tf.train.Saver()
        #调用Saver对象的save函数，将当前的模型参数保存到指定的checkpoint文件中。
        saver.save(self.sess, checkpoint)
    '''
    使用TensorFlow的Saver类创建一个Saver对象。该对象用于读取checkpoint文件中的模型参数
    '''
    def load(self, checkpoint):
        #使用TensorFlow的Saver类创建一个Saver对象。该对象用于读取checkpoint文件中的模型参数
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint)
        if ckpt and ckpt.model_checkpoint_path:
            #调用Saver对象的restore函数，从指定的checkpoint文件中读取模型参数
            print('training from last checkpoint', checkpoint)
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            #NewCheckpointReader函数，从checkpoint文件中读取模型的参数，并将这些参数赋值给类的成员变量
        reader = pywrap_tensorflow.NewCheckpointReader(checkpoint+'trained_model.ckpt')
        #Print tensor name and values
#        var_to_shape_map = reader.get_variable_to_shape_map()
#        for key in var_to_shape_map:
#            print("tensor_name: ", key)
#            print(reader.get_tensor(key))
        self.bias_1 = reader.get_tensor('fc1/bias')
        self.kernel_1 = reader.get_tensor('fc1/kernel')
        self.bias_2 = reader.get_tensor('fc2/bias')
        self.kernel_2 = reader.get_tensor('fc2/kernel')

    #softmax函数实现
    def softmax(self, x):
        x = x - np.max(x)
        exp_x = np.exp(x)
        softmax_x = exp_x / np.sum(exp_x)
        return softmax_x
    #激活函数relu的实现
    def relu(self, x):
        return np.maximum(0,x)
    #激活函数sigmoid的实现
    def sigmoid(self, x):
        s = 1 / (1 + np.exp(-x))
        return s
    #激活函数tach的实现
    def tanh(self, x):
        exp_x = np.exp(x)
        exp_nx = np.exp(-x)
        return (exp_x - exp_nx) / (exp_x + exp_nx)
    '''
    接收一个观察值，并通过计算两层神经网络的输出来选择一个动作
    输入：一个oberavation
    输出：根据行动的概率选择一个行动并返回
    '''
    def quick_time_action(self, observation): # matrix implementation for fast efficiency when the model is ready
        l1 = self.tanh(np.dot(observation, self.kernel_1) + self.bias_1)
        pro = self.softmax(np.dot(l1, self.kernel_2) + self.bias_2)
        action = np.random.choice(range(self.n_actions), p=pro[0])
        return action

        
