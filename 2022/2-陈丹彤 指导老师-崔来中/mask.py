import tensorflow.compat.v1 as tf

import ray
from ray.rllib.models import Model
from ray.rllib.models.tf.misc import normc_initializer

#在网上搜索Rllib模型，可以看到一张红绿色的模型图。
#从Environment开始，输入action，输出observation，obs通过预处理器Preprocessor和过滤器Filter，
#然后送入Policy的Model中，Model的输出送入Action Distribution模块决定下一个动作。
#Environment、Preprocessor、Model、Action Distribution模块可以用户自定义的，Filter和Policy Loss表示内置模块。

class PartitionMaskModel(Model):
    """Model that only allows the partitioning action at the first level.
    仅允许在第一级执行分区操作的模型。
    """
    #重写父类Model中的方法
    #input_dict包括(obs、prev_action和prev_reward、is_training)，
    #obs在neurocuts_env.py中的NeuroCutsEnv设置了
    #options是内置模型超参数，其中"fcnet_hiddens": [256, 256]为全连接网络隐层神经元个数列表
    #并返回指定输出大小的特征层和浮点张量。
    def _build_layers_v2(self, input_dict, num_outputs, options):
        mask = input_dict["obs"]["action_mask"]

        last_layer = input_dict["obs"]["real_obs"]
        hiddens = options["fcnet_hiddens"] #fcnet_hiddens在run_neurocuts.py中的run_experiments函数里被设定为[512,512]
        
       #tf.layers.dense用于添加一个全连接层
        #tf.layers.dense(
            #inputs,                 #层的输入
            #units,                  #该层的输出维度
            #activation=None,        #激活函数
            #use_bias=True,          
            #kernel_initializer=None,    # 卷积核的初始化器
            #bias_initializer=tf.zeros_initializer(),  # 偏置项的初始化器
            #kernel_regularizer=None,    # 卷积核的正则化
            #bias_regularizer=None,      # 偏置项的正则化
            #activity_regularizer=None, 
            #kernel_constraint=None,
            #bias_constraint=None,
            #trainable=True,
            #name=None,  # 层的名字
            #reuse=None  # 是否重复使用参数
        #)
        for i, size in enumerate(hiddens):
            label = "fc{}".format(i)
            last_layer = tf.layers.dense(
                last_layer,
                size,
                #normc_initializer(std=1.0, axis=0)返回TensorFlow的参数初始值设定项。std为标准偏差
                kernel_initializer=normc_initializer(1.0), 
                activation=tf.nn.tanh,
                name=label)
        #action_logits为全连接层的最后一层，神经元个数需要是_build_layers_v2()参数中的num_outputs
        # logits这个词一般指的就是最终全连接层的输出
        action_logits = tf.layers.dense(
            last_layer,
            num_outputs,
            kernel_initializer=normc_initializer(0.01),
            activation=None,
            name="fc_out")

        if num_outputs == 1:
            return action_logits, last_layer

        # Mask out invalid actions (use tf.float32.min for stability)。掩盖无效动作（使用tf.float32.min以获得稳定性）
        #tf.maximum()函数用于按元素返回两个指定张量的最大值
        #张量（tensor）是多维数组，目的是把向量、矩阵推向更高的维度。
        inf_mask = tf.maximum(tf.log(mask), tf.float32.min)
        masked_logits = inf_mask + action_logits

        return masked_logits, last_layer
