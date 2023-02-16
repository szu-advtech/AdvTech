import tensorflow as tf

a = tf.constant([1,2,3,4,5,6], shape=[2, 3])
b = tf.constant([1,2,3], shape=[1, 3])

c = tf.multiply(a, b)

with tf.Session() as sess:
    temp = sess.run(c)
print(temp)