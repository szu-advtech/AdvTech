import tensorflow as tf
import numpy as np

# 定义训练轮次
training_steps = 30000

# 定义输入的数据和对应的标签并在 for 循环里进行填充
data = []
label = []

for i in range(200):
    x1 = np.random.uniform(-1, 1)
    x2 = np.random.uniform(0, 2)

    # 这里对 x1,x2 进行判断，如果产生的点落在半径为1的圆内，则label为0，否则为1
    if x1 ** 2 + x2 ** 2 <= 1:
        data.append([np.random.normal(x1, 0.1), np.random.normal(x2, 0.1)])
        label.append(0)
    else:
        data.append([np.random.normal(x1, 0.1), np.random.normal(x2, 0.1)])
        label.append(1)

# numpy 的 hstack() 函数用于在水平方向将元素堆起来

data = np.hstack(data).reshape(-1, 2)
label = np.hstack(label).reshape(-1, 1)


# 定义完成前向传播的隐层
def hidden_layer(input_tensor, weight1, bias1, weight2, bias2, weight3, bias3):
    layer1 = tf.nn.relu(tf.matmul(input_tensor, weight1) + bias1)
    layer2 = tf.nn.relu(tf.matmul(layer1, weight2) + bias2)
    return tf.matmul(layer2, weight3) + bias3


xs = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
ys = tf.placeholder(tf.float32, shape=(None, 1), name="y-output")

# 定义权重参数和偏置参数
weight1 = tf.Variable(tf.truncated_normal([2, 10], stddev=0.1))
bias1 = tf.Variable(tf.constant(0.1, shape=[10]))
weight2 = tf.Variable(tf.truncated_normal([10, 10], stddev=0.1))
bias2 = tf.Variable(tf.constant(0.1, shape=[10]))
weight3 = tf.Variable(tf.truncated_normal([10, 1], stddev=0.1))
bias3 = tf.Variable(tf.constant(0.1, shape=[1]))

# 计算 data 数组长度
sample_size = len(data)

# 得到隐藏层前向传播结果
y = hidden_layer(xs, weight1, bias1, weight2, bias2, weight3, bias3)

# 定义损失函数
error_loss = tf.reduce_sum(tf.pow(ys-y, 2))
tf.add_to_collection("losses", error_loss)

# 参数L2正则化
regularizer = tf.contrib.layers.l2_regularizer(0.01)
retularization = regularizer(weight1) + regularizer(weight2) + regularizer(weight3)
tf.add_to_collection("losses", retularization)

# get_collection函数获取指定集合中的所有个体，这里是获取所有损失值，并在 add_n() 函数中进行加和运算
loss = tf.add_n(tf.get_collection("losses"))

# 定义一个优化器
train_op = tf.train.AdamOptimizer(0.05).minimize(loss)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(training_steps):
        sess.run(train_op, feed_dict={xs: data, ys: label})

        # 每迭代 2000次 输出一个loss值
        if i % 2000 == 0:
            loss_value = sess.run(loss, feed_dict={xs: data, ys: label})
            print("After %d steps, mse_loss: %f" %(i, loss_value))




