# thsi script is used for the validation of MF model
import pickle
import numpy as np
import tensorflow as tf

domain = 'target'
if domain == 'source':
    ckpt_path = './MF/mf_s'
    graph_path = './MF/mf_s/mf_s.ckpt.meta'
else:
    ckpt_path = './MF/mf_t'
    graph_path = './MF/mf_t/mf_t.ckpt.meta'

# restore test code
new_graph = tf.Graph()
with tf.Session(graph=new_graph) as sess:
    loader = tf.train.import_meta_graph(graph_path)
    loader.restore(sess, tf.train.latest_checkpoint(ckpt_path))
    U, V = sess.run(["U:0", "V:0"])

with open('./Mt.pkl', 'rb') as f:
    R = pickle.load(f)

print(R.nnz)
R = R.toarray()
n, m = R.shape
batch_size = 1000
batch_num = np.int32(np.ceil(n / batch_size))
sum = 0
loss = 0

for i in range(batch_num):
    start = i * batch_size
    end = (i + 1) * batch_size
    if end > n:
        end = n

    batch_R = R[start:end, :]
    mask = np.int64(batch_R > 0)
    pred = np.dot(U[start:end, :], V.T)
    err = mask * (batch_R - pred)

    sum += np.sum(mask)
    loss += np.sum(np.square(err))

print('MSE:', loss / sum)