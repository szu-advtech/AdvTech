from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from keras.backend import get_session
# from tensorflow.keras.backend import get_session

'''
这里卷积的输入必须都写成NHWC的格式,也就是channel必须放最后(因为原代码不完善),
然而tensorflow也有直接支持此格式的卷积层,故没什么大问题.
'''
# from https://stackoverflow.com/questions/39088489/tensorflow-periodic-padding
# original name: periodic_padding_flexible
def circular_pad(tensor, axis,padding=1):# 实现循环padding
    """
        add periodic padding to a tensor for specified axis
        tensor: input tensor
        axis: on or multiple axis to pad along, int or tuple
        padding: number of cells to pad, int or tuple

        return: padded tensor
        为指定轴的张量添加周期性填充
        张量:输入张量
        轴:在或多个轴上沿整型或元组填充
        Padding:填充的单元格数，int或tuple
        返回:填充张量
    """
    '''
    isinstance(object, classinfo):
    * object -- 实例对象。
    * classinfo -- 可以是直接或间接类名、基本类型或者由它们组成的元组。
    return  如果对象的类型与参数二的类型（classinfo）相同则返回 True，否则返回 False。
    isinstance() 与 type() 区别：
    type() 不会认为子类是一种父类类型，不考虑继承关系。
    isinstance() 会认为子类是一种父类类型，考虑继承关系。
    如果要判断两个类型是否相同推荐使用 isinstance()。
    '''
    if isinstance(axis,int):
        axis = (axis,)
    if isinstance(padding,int):
        padding = (padding,)

    ndim = len(tensor.shape)
    # zip函数可以在axis,padding中迭代元素
    for ax,p in zip(axis,padding):
        # create a slice object that selects everything from all axes,
        # except only 0:p for the specified for right, and -p: for left
        # 语法：class slice(start, stop[, step])
        # 参数：start起始，stop终止，step步长
        # 返回值：返回一个slice对象
        #  start 和 step 参数默认为 None。
        ind_right = [slice(-p,None) if i == ax else slice(None) for i in range(ndim)]
        ind_left = [slice(0, p) if i == ax else slice(None) for i in range(ndim)]
        right = tensor[ind_right]
        left = tensor[ind_left]
        middle = tensor
        tensor = tf.concat([right,middle,left], axis=ax)

    return tensor

def cirbasicblock(input,title,filter_num,firstpad,stride=1):# 返回一个构造后的块
    activ = tf.nn.relu
    convout = activ(conv(circular_pad(input,(1,2),firstpad), '{}c1'.format(title), nf=filter_num, rf=3, stride=stride, init_scale=np.sqrt(2)))
    convout = conv(circular_pad(convout,(1,2),(1,1)), '{}c2'.format(title), nf=filter_num, rf=3, stride=1, init_scale=np.sqrt(2))
    if stride != 1:
        resout = conv(input, '{}c_res'.format(title), nf=filter_num, rf=1, stride=stride, init_scale=np.sqrt(2))
    else:
        resout = input

    output = activ(convout + resout)
    return output    


def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        # lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4:  # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v  # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)

    return _ortho_init
'''
placeholder（占位符）
placeholder是TensorFlow的占位符节点，由placeholder方法创建，其也是一种常量，但是由用户在调用run方法是传递的，也可以将placeholder理解为一种形参。即其不像constant那样直接可以使用，需要用户传递常数值。
变量主要针对的神经网络中的参数，placeholder主要针对的样本中的输入数据X及其对应的Y。
因为如果每轮迭代中输入数据都都以 x=tf.constant([[3., 3.]])形式输入的话(必须转换成张量)，计算图将会非常之大，这是因为每生成一个常量，都会在计算图中增加一个节点
如果经过成千上万次的迭代的话，那就会生成非常多的节点，利用率也比较低。
TensorFlow提供了placeholder机制来解决了这个问题
placeholder相当于定义了一个位置，这个位置中的数据在运行时再指定。
这样程序中就不需要生成大量的常量来提供输入数据，而只需要将数据通过placeholder传入Tensorflow计算图。
创建方式
X = tf.placeholder(dtype=tf.float32, shape=[144, 10], name='X')
参数说明
dtype：数据类型，必填，默认为value的数据类型，传入参数为tensorflow下的枚举值（float32，float64.......）
shape：数据形状，选填，不填则随传入数据的形状自行变动，可以在多次调用中传入不同形状的数据
name：常量名，选填，默认值不重复，根据创建顺序为（Placeholder，Placeholder_1，Placeholder_2.......）
'''
def placeholder(dtype=tf.float32, shape=None):
    return tf.placeholder(dtype=dtype, shape=combined_shape(None, shape))
    
'''
return (length, shape) if np.isscalar(shape) else (length, *shape):
if(np.isscalar(shape)) return (lengtn,shape)
return (length,*shape)

isscalar():
如果num的类型是标量类型，则isscalar（）函数将返回True。

*:
1.乘法
2.收集多余信息 a,b,*c = [1,2,3,4] c是列表
3.函数收集参数 放在一个元组tuple里
4.函数收集关键字参数 传参的时候*号不会识别关键字 需要两个**
5.分配tuple参数 形参有俩，传入一个*param ， param=[1.2]
6.分配字典参数 形参有俩，传入一个**param ， param={'x':1,'y'=2}

(length, *shape):
这里*shape 可能是分解tuple的
'''
def combined_shape(length, shape=None):
    if shape is None:
        return (length,) #(length,)表示一维
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)


def conv(x, scope, *, nf, rf, stride, pad='VALID', init_scale=1.0, data_format='NHWC', one_dim_bias=False):
    '''
    nf: 几个卷积核(输出的channel数)
    rf: 卷积核宽
    '''
    if data_format == 'NHWC':
        channel_ax = 3
        strides = [1, stride, stride, 1]
        bshape = [1, 1, 1, nf]
    elif data_format == 'NCHW':
        channel_ax = 1
        strides = [1, 1, stride, stride]
        bshape = [1, nf, 1, 1]
    else:
        raise NotImplementedError
    bias_var_shape = [nf] if one_dim_bias else [1, nf, 1, 1]
    nin = x.get_shape()[channel_ax].value
    wshape = [rf, rf, nin, nf]
    with tf.variable_scope(scope):
        '''
        该函数共有十一个参数，常用的有：名称name、变量规格shape、变量类型dtype、变量初始化方式initializer、所属于的集合collections。
        常见的initializer有：常量初始化器tf.constant_initializer、正太分布初始化器tf.random_normal_initializer、截断正态分布初始化器tf.truncated_normal_initializer、均匀分布初始化器tf.random_uniform_initializer。
        '''
        w = tf.get_variable("w", wshape, initializer=ortho_init(init_scale))
        b = tf.get_variable("b", bias_var_shape, initializer=tf.constant_initializer(0.0))
        if not one_dim_bias and data_format == 'NHWC':
            b = tf.reshape(b, bshape)
        '''
        def conv2d(input,  # 张量输入
			filter, # 卷积核参数
			strides, # 步长参数
			padding, # 卷积方式
			use_cudnn_on_gpu=None, # 是否是gpu加速
            data_format=None,  # 数据格式，与步长参数配合，决定移动方式
            name=None): # 名字，用于tensorboard图形显示时使用
        https://blog.csdn.net/qq_30934313/article/details/86626050
        '''
        return tf.nn.conv2d(x, w, strides=strides, padding=pad, data_format=data_format) + b


def fc(x, scope, nh, *, init_scale=1.0, init_bias=0.0):
    with tf.variable_scope(scope):
        nin = x.get_shape()[1].value
        w = tf.get_variable("w", [nin, nh], initializer=ortho_init(init_scale))
        b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(init_bias))
        return tf.matmul(x, w) + b


def conv_to_fc(x):
    '''
    np.prod
    返回给定轴上的数组元素的乘积。
    reshape:
    [-1,nh]
    多少行不知道 反正是nh列
    '''
    nh = np.prod([v.value for v in x.get_shape()[1:]])
    x = tf.reshape(x, [-1, nh])
    return x

'''
基本思想是:with所求值的对象必须有一个enter()方法，一个exit()方法。
紧跟with后面的语句被求值后，返回对象的__enter__()方法被调用，这个方法的返回值将被赋值给as后面的变量。当with后面的代码块全部被执行完之后，将调用前面返回对象的exit()方法。
'''
class ACCNNModel():
    def __init__(self, observation_space, action_space, config=None, model_id='0', *args, **kwargs):
        with tf.variable_scope(model_id):
            self.x_ph = placeholder(shape=observation_space, dtype='uint8')
            self.encoded_x_ph = tf.to_float(self.x_ph)
            self.a_ph = placeholder(dtype=tf.int32)  #只传dtype or shape length怎么确定
            self.legal_action = placeholder(shape = (None,))

        self.logits = None
        self.vf = None
        '''
        get_session()
        返回后端使用的TF会话。
        如果默认的TensorFlow会话可用，我们将返回它。
        否则，我们将返回全局Keras会话。
        如果此时不存在全局Keras会话：我们将创建一个新的全局会话。
        请注意，您可以通过K.set_session（sess）手动设置全局会话。
        返回：
        返回TensorFlow会话。
        '''
        session = get_session()
        self.sess = session
        self.observation_space = observation_space
        self.action_space = action_space
        self.model_id = model_id
        self.config = config
        self.scope = model_id

        # Build up model
        self.build()

        # Build assignment ops
        self._weight_ph = None
        self._to_assign = None
        self._nodes = None
        self._build_assign()

        # Build saver
        self.saver = tf.train.Saver(tf.trainable_variables())

        pd = CategoricalPd(self.logits)
        self.action = pd.sample()
        self.neglogp = pd.neglogp(self.action)
        self.neglogp_a = pd.neglogp(self.a_ph)
        self.entropy = pd.entropy()
        '''
        # 必须要使用global_variables_initializer的场合
        # 含有tf.Variable的环境下，因为tf中建立的变量是没有初始化的，也就是在debug时还不是一个tensor量，而是一个Variable变量类型
        size_out = 10
        tensor = tf.Variable(tf.random_normal(shape=[size_out]))
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)  # initialization variables
            print(sess.run(tensor))
        # 可以不适用初始化的场合
        # 不含有tf.Variable、tf.get_Variable的环境下
        # 比如只有tf.random_normal或tf.constant等
        size_out = 10
        tensor = tf.random_normal(shape=[size_out])  # 这里debug是一个tensor量哦
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            # sess.run(init)  # initialization variables
            print(sess.run(tensor))
        '''
        init_op = tf.global_variables_initializer()
        '''
        会话的主要作用就是拥有并管理TensorFlow程序运行时的所有资源，当所有计算完成之后需要关闭会话来帮助系统回收资源，否则就可能出现资源泄露。
        在TensorFlow中，有两种用于运行计算图（graph）的会话（session）
        tf.Session( )
        tf.InteractivesSession( )
        使用"with"语句，自动关闭会话
        在TensorFlow中变量tf.Variable的作用就是保存和更新神经网络中的参数的
        TensorFlow随机数生成函数
        TensorFlow常数生成函数
        要使用变量，首先要通过session的run运行初始化变量，因为变量也是一个张量，上面的只是定义了计算过程，并没有真正执行：
        sess.run（w2.initializer）#初始化w2真正赋值
        #一次初始化所有变量
        init_op=tf.initialize_all_variables()
        sess.run(init_op)
        '''
        session.run(init_op)

    def set_weights(self, weights, *args, **kwargs) -> None:
        # 更新参数 字典形式存储
        feed_dict = {self._weight_ph[var.name]: weight
                     for (var, weight) in zip(tf.trainable_variables(scope=self.scope), weights)}
        # sess.run究竟是如何运行的
        self.sess.run(self._nodes, feed_dict=feed_dict)
    '''
    tf.trainable_variables()
    顾名思义，这个函数可以也仅可以查看可训练的变量，在我们生成变量时，无论是使用tf.Variable()还是tf.get_variable()生成变量，都会涉及一个参数trainable,其默认为True。
    对于一些我们不需要训练的变量，比较典型的例如学习率或者计步器这些变量，我们都需要将trainable设置为False，这时tf.trainable_variables() 就不会打印这些变量。
    另一个问题就是，如果变量定义在scope域中，是否会有不同。实际上，tf.trainable_variables()是可以通过参数选定域名的
    如我们只希望查看‘var’域中的变量，我们可以通过加入scope参数的方式实现：
    
    session.run()
    run(self, fetches, feed_dict=None, options=None, run_metadata=None)
    其中常用的fetches和feed_dict就是常用的传入参数。fetches主要指从计算图中取回计算结果进行放回的那些placeholder和变量，而feed_dict则是将对应的数据传入计算图中占位符，它是字典数据结构只在调用方法内有效。
    '''
    def get_weights(self, *args, **kwargs) -> Any:
        return self.sess.run(tf.trainable_variables(self.scope))

    def save(self, path: Path, *args, **kwargs) -> None:
        self.saver.save(self.sess, str(path))

    def load(self, path: Path, *args, **kwargs) -> None:
        self.saver.restore(self.sess, str(path))

    def _build_assign(self):
        self._weight_ph, self._to_assign = dict(), dict()
        '''
        在创造变量(tf.Variable、tf.get_variable等)时，都会有一个trainable的选项，表示该变量是否可训练，这个函数会返回图中所有trainable=True的变量。
        tf.get_variable和tf.Variable的默认选项是True，而tf.constant只能是False：
        '''
        variables = tf.trainable_variables(self.scope)
        for var in variables:
            self._weight_ph[var.name] = tf.placeholder(var.value().dtype, var.get_shape().as_list())
            self._to_assign[var.name] = var.assign(self._weight_ph[var.name])
        self._nodes = list(self._to_assign.values())

    def forward(self, states: Any, legal_action, *args, **kwargs) -> Any:
        #这里run不确定下一步是去哪
        return self.sess.run([self.action, self.vf, self.neglogp], feed_dict={self.x_ph: states, self.legal_action: legal_action})

    def build(self, *args, **kwargs) -> None:
        with tf.variable_scope(self.scope):
            with tf.variable_scope('cnn_base'):
                '''
                tf.cast(x, dtype, name=None)
                x，输入张量
                dtype，转换数据类型
                name，名称
                '''
                # scaled_images = tf.cast(self.encoded_x_ph, tf.float32) / 255.
                input_images = tf.cast(self.encoded_x_ph, tf.float32)
                # relu称为线性整流函数(修正线性单元)，tf.nn.relu()用于将输入小于0的值增幅为0，输入大于0的值不变。
                activ = tf.nn.relu

                outstem = activ(conv(circular_pad(input_images,(1,2),(1,1)), 'c_stem', nf=16, rf=3, stride=1, init_scale=np.sqrt(2)))
                outstem = tf.nn.max_pool(circular_pad(outstem,(1,2),(1,1)),[1,2,2,1],[1,1,1,1],padding='VALID')
                
                outresnet = cirbasicblock(outstem,"rb1_1",16,(1,1),1)
                outresnet = cirbasicblock(outresnet,"rb1_2",16,(1,1),1)
                outresnet = cirbasicblock(outresnet,"rb2_1",16,(1,1),2)
                outresnet = cirbasicblock(outresnet,"rb2_2",16,(1,1),1)
                outresnet = cirbasicblock(outresnet,"rb3_1",32,(1,1),2)
                outresnet = cirbasicblock(outresnet,"rb3_2",32,(1,1),1)
                outresnet = cirbasicblock(outresnet,"rb4_1",32,(0,1),2)
                outresnet = cirbasicblock(outresnet,"rb4_2",32,(1,1),1)
                outresnet = conv_to_fc(outresnet)

                latent = activ(fc(outresnet, 'fc1', nh=64, init_scale=np.sqrt(2)))

            with tf.variable_scope('pi'):
                pih1 = activ(fc(latent, 'pi_fc1', 64, init_scale=0.01))
                pih2 = activ(fc(pih1, 'pi_fc2', 64, init_scale=0.01))
                #有时候在做分类任务时，如果一些类别明确不会被分类到，可以通过mask把logits非法部分置为较大的负数。
                logits_without_mask = fc(pih2, 'pi_fc3', self.action_space, init_scale=0.01)
                #logit原本是一个函数，它是sigmoid函数 但在深度学习中，logits就是最终的全连接层的输出，而非其本意。
                self.logits = logits_without_mask + 1000. *tf.to_float(self.legal_action-1)

            with tf.variable_scope('v'):
                vh1 = activ(fc(latent, 'v_fc1', 64, init_scale=0.01))
                vh2 = activ(fc(vh1, 'v_fc2', 64, init_scale=0.01))
                '''
                压缩这个张量：如values为shape([1, 2, 1, 3, 1, 1])，经历tf.squeeze(values)，shape变成([2,3])
                可以通过指定axis来删除特定维度1的维度: 如values为shape([1, 2, 1, 3, 1, 1])，经历tf.squeeze(values,[0])，shape变成([2, 1, 3, 1, 1])
                '''
                self.vf = tf.squeeze(fc(vh2, 'v_fc3', 1, init_scale=0.01), axis=1)

class CategoricalPd:
    def __init__(self, logits):
        self.logits = logits
    '''
    首先，明确一点，tf.argmax可以认为就是np.argmax。tensorflow使用numpy实现的这个API。 
　　 简单的说，tf.argmax就是返回最大的那个数值所在的下标。  
    '''
    def mode(self):
        return tf.argmax(self.logits, axis=-1)

    def logp(self, x):
        return -self.neglogp(x)

    def neglogp(self, x):
        # return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=x)
        # Note: we can't use sparse_softmax_cross_entropy_with_logits because
        #       the implementation does not allow second-order derivatives...
        if x.dtype in {tf.uint8, tf.int32, tf.int64}:
            # one-hot encoding
            x_shape_list = x.shape.as_list()
            '''
            a='python'
            b=a[::-1]
            print(b) #nohtyp
            c=a[::-2]
            print(c) #nhy
            #从后往前数的话，最后一个位置为-1
            d=a[:-1] #从位置0到位置-1之前的数
            print(d) #pytho
            e=a[:-2] #从位置0到位置-2之前的数
            print(e) #pyth
            '''
            logits_shape_list = self.logits.get_shape().as_list()[:-1]
            ''' 
            a=[1,2,3]
            b=[5,6,7]
            c=zip(a,b)
            print(next(c))
            print(next(c))
            print(next(c))
            '''
            for xs, ls in zip(x_shape_list, logits_shape_list):
                if xs is not None and ls is not None:
                    '''
                    Python assert（断言）用于判断一个表达式，在表达式条件为 false 的时候触发异常。
                    断言可以在条件不满足程序运行的情况下直接返回错误，而不必等待程序运行后出现崩溃的情况
                    '''
                    assert xs == ls, 'shape mismatch: {} in x vs {} in logits'.format(xs, ls)
            '''
            one_hot(indices, depth, on_value=None, off_value=None, axis=None, dtype=None, name=None)
            Returns a one-hot tensor.
            indices表示输入的多个数值，通常是矩阵形式；depth表示输出的尺寸。
            由于one-hot类型数据长度为depth位,其中只用一位数字表示原输入数据，这里的on_value就是这个数字，默认值为1，one-hot数据的其他位用off_value表示，默认值为0。
            tf.one_hot()函数规定输入的元素indices从0开始，最大的元素值不能超过（depth - 1），因此能够表示depth个单位的输入。若输入的元素值超出范围，输出的编码均为 [0, 0 … 0, 0]。
            indices = 0 对应的输出是[1, 0 … 0, 0], indices = 1 对应的输出是[0, 1 … 0, 0], 依次类推，最大可能值的输出是[0, 0 … 0, 1]。
            '''
            x = tf.one_hot(x, self.logits.get_shape().as_list()[-1])
        else:
            # already encoded
            assert x.shape.as_list() == self.logits.shape.as_list()
            '''
            Odds（A）= 发生事件A的概率  /  不发生事件A的概率    (公式2）
            请注意Logit一词的分解，对它（it）Log（取对数），这里“it”就是Odds。下面我们就可以给出Logit的定义了
            与概率不同的地方在于，Logit的一个很重要的特性，就是它没有上下限
            通过变换，Logit的值域没有上下界限，这就给建模提供了方便。
            很显然，死亡率P和x是正相关的，但由于P的值域在[0,1]之间，而x的取值范围要宽广得多。P不太可能是x的线性关系或二次函数，一般的多项式函数也不太适合，这就给此类函数的拟合（回归分析）带来麻烦。
            此外，当P接近于0或1的时候，即使一些因素变化很大，P的值也不会有显著变化。
            例如，对于高可靠系统，可靠度P已经是0.997了，倘若在改善条件、提高工艺和改进体系结构，可靠度的提升只能是小数点后后三位甚至后四位，单纯靠P来度量，已经让我们无所适从，不知道改善条件、提高工艺和改进体系结构到底有多大作用。
            再比如，宏观来看，灾难性天气发送的概率P非常低（接近于0），但这类事件类似于黑天鹅事件（特征为：影响重大、难以预测及事后可解释），由于P对接近于0的事件不敏感，通过P来度量，很难找到刻画发生这类事件的前兆信息。
            这时，Logit函数的优势就体现出来了。从图1可以看出，在P=0或P=1附近，Logit非常敏感（值域变化非常大）。通过Logit变换，P从0到1变化时，Logit是从到。Logit值域的不受限，让回归拟合变得容易了！
            在化学反应里，催化剂能改变反应物化学反应速率而不改变化学平衡，且本身的质量和化学性质在化学反应前后都没有发生改变。
            如果你认真观察的话，就会发现，它其实就是在神经网络种广泛使用的Sigmoid函数，又称对数几率函数（logistic function）。
            通常，我们把公式（5）表示的便于拟合的“概率替代物”称为logits。事实上，在多分类（如手写识别等）中，某种分类器的输出（即分类的打分），也称为logits，即使它和Odds的本意并没有多大联系，但它们通过某种变换，也能变成“概率模型”，比如下面我们即将讲到的Softmax变换。
            tf.nn.softmax_cross_entropy_with_logits(
                _sentinel=None,
                labels=None,
                logits=None,
                dim=-1,
                name=None
            )
            这个函数的功能就是计算labels和logits之间的交叉熵（cross entropy）。
            第一个参数基本不用。此处不说明。
            第二个参数label的含义就是一个分类标签，所不同的是，这个label是分类的概率，比如说[0.2,0.3,0.5]，labels的每一行必须是一个概率分布（即概率之合加起来为1）。
            现在来说明第三个参数logits，logit的值域范围[-inf,+inf]（即正负无穷区间）。我们可以把logist理解为原生态的、未经缩放的，可视为一种未归一化的l“概率替代物”，如[4, 1, -2]。它可以是其他分类器（如逻辑回归等、SVM等）的输出。
            例如，上述向量中“4”的值最大，因此，属于第1类的概率最大，“1”的值次之，所以属于第2类的概率次之。
            交叉熵（Cross Entropy）是Shannon信息论中一个重要概念，主要用于度量两个概率分布间的差异性信息。
            由于logis本身并不是一个概率，所以我们需要把logist的值变化成“概率模样”。这时Softmax函数该出场了。Softmax把一个系列的概率替代物（logits）从[-inf, +inf] 映射到[0,1]。除此之外，Softmax还保证把所有参与映射的值累计之和等于1，变成诸如[0.95, 0.05, 0]的概率向量。这样一来，经过Softmax加工的数据可以当做概率来用（如图2所示）。
            经过softmax的加工，就变成“归一化”的概率（设为p1），这个新生成的概率p1，和labels所代表的概率分布（设为p2）一起作为参数，用来计算交叉熵。
            这个差异信息，作为我们网络调参的依据，理想情况下，这两个分布尽量趋近最好。如果有差异（也可以理解为误差信号），我们就调整参数，让其变得更小，这就是损失（误差）函数的作用。
            最终通过不断地调参，logit被锁定在一个最佳值（所谓最佳，是指让交叉熵最小，此时的网络参数也是最优的）。
            （1）如果labels的每一行是one-hot表示，也就是只有一个地方为1（或者说100%），其他地方为0（或者说0%），还可以使用tf.sparse_softmax_cross_entropy_with_logits()。之所以用100%和0%描述，就是让它看起来像一个概率分布。
            （2）tf.nn.softmax_cross_entropy_with_logits（）函数已经过时 (deprecated)，它在TensorFlow未来的版本中将被去除。取而代之的是
            tf.nn.softmax_cross_entropy_with_logits_v2（）。
             (3)参数labels,logits必须有相同的形状 [batch_size, num_classes] 和相同的类型(float16, float32, float64)中的一种，否则交叉熵无法计算。
            （4）tf.nn.softmax_cross_entropy_with_logits 函数内部的 logits 不能进行缩放，因为在这个工作会在该函数内部进行（注意函数名称中的 softmax ，它负责完成原始数据的归一化），如果 logits 进行了缩放，那么反而会影响计算正确性。
            '''
        return tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=x)

    def kl(self, other):
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        a1 = other.logits - tf.reduce_max(other.logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        ea1 = tf.exp(a1)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        z1 = tf.reduce_sum(ea1, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), axis=-1)

    def entropy(self):
        '''
        tf.reduce_max()函数
        tf.reduce_max(
            input_tensor,
            axis=None,
            name=None,
            keepdims=False  #是否保持矩形原狀
        ）
        参数解释：
        input_tensor：输入数据，tensor、array、dataframe 都可以
        axis：表示维度，从0开始表示最外层维度，也可以用-1表示最内层维度；
                  [0, [1, [2, [3,[...]]]]],或者[[[[[...], -4], -3], -2], -1]  数字表示对应[ ]的维度。当axis位默认表示求全局最大值。
                axis=0指的是计算矩阵每列的最大值，axis=1计算行最大值
        keepdims：默认时，维度会减少1，为True时，保持维度不变。
        name：操作名称
        reduction_indices:axis的旧名称(已经弃用）
        '''
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        '''
        tf.reduce_sum(
            input_tensor, 
            axis=None, 
            keepdims=None,
            name=None)
        tf.reduce_sum()作用是按一定方式计算张量中元素之和
        input_tensor为待处理张量；
        axis指定按哪个维度进行加和，默认将所有元素进行加和；
        keepdims默认为False，表示不维持原来张量的维度，反之维持原张量维度；
        name用于定义该操作名字。
        '''
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=-1)

    def sample(self):
        '''
        tf.random_uniform()：用于从均匀分布中输出随机值。
        def random_uniform(shape,
                           minval=0,
                           maxval=None,
                           dtype=dtypes.float32,
                           seed=None,
                           name=None):
        shape： 张量形状
        minval： 随机值范围下限，默认0
        maxval:   随机值范围上限，如果 dtype 是浮点，则默认为1 。
        dtype：   输出的类型：float16、float32、float64、int32、orint64
        seed：    一个 Python 整数.用于为分布创建一个随机种子
        random seed:在用随机函数产生随机数的时候，我们总会设一个随机种子，那这个随机种子是什么，设随机种子有什么用呢？
        我们知道计算机产生的随机数都是伪随机数，是利用算法产生的一系列数。因此，需要给函数一个随机值作为初始值，以此基准不断迭代得到一系列随机数。这个初始值就叫做随机种子。
        name：  操作的名称(可选)
        '''
        u = tf.random_uniform(tf.shape(self.logits), dtype=self.logits.dtype)
        # axis=-1事实上就是表示压缩最后一个维度
        return tf.argmax(self.logits - tf.log(-tf.log(u)), axis=-1)
