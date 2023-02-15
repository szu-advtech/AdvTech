import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from setting import *
from GH_itpair2 import GH
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = select_gpu

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


gpuconfig = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=per_process_gpu_memory_fraction))

with tf.Session(config=gpuconfig) as sess:
    model = GH(sess)
    t1 = time.time()
    model.Train() if phase == 'train' else model.test(phase)
    t2 = time.time()
    print (t2-t1)
