from model_train import T_CNN
from utils import (
  imsave,
  prepare_data
)
import numpy as np
import tensorflow as tf

import pprint
import os

flags = tf.app.flags
flags.DEFINE_integer("epoch", 400, "训练批次，测试时无用")
flags.DEFINE_integer("batch_size", 16, "每个批次训练集大小")
flags.DEFINE_integer("image_height", 112, "输入训练图片的高度")
flags.DEFINE_integer("image_width", 112, "输出训练图片的宽度")
flags.DEFINE_integer("label_height", 112 ,"输入参考图片的高度")
flags.DEFINE_integer("label_width", 112, "输入参考图片的宽度")
flags.DEFINE_float("learning_rate", 0.001, "学习率")
flags.DEFINE_float("beta1", 0.5, "使用adam动量梯度下降时的参数")
flags.DEFINE_integer("c_dim", 3, "输入图像的色彩通道数")
flags.DEFINE_string("checkpoint_dir", "", "参数文件夹路径")
flags.DEFINE_string("sample_dir", "sample", "样本文件夹")
flags.DEFINE_string("test_data_dir", "test", "测试时输入的样本")
flags.DEFINE_boolean("is_train", True, "训练——True；测试——False")
FLAGS = flags.FLAGS


pp = pprint.PrettyPrinter()

def main(_):
  pp.pprint(flags.FLAGS.__flags)

  # 寻找参数文件
  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)
  
  with tf.Session() as sess:
    srcnn = T_CNN(sess, 
                  image_height=FLAGS.image_height,
                  image_width=FLAGS.image_width, 
                  label_height=FLAGS.label_height, 
                  label_width=FLAGS.label_width, 
                  batch_size=FLAGS.batch_size,
                  c_dim=FLAGS.c_dim, 
                  checkpoint_dir=FLAGS.checkpoint_dir,
                  sample_dir=FLAGS.sample_dir
                  )
    print("--------------------------------")
    print(FLAGS.image_height)
    srcnn.train(FLAGS)
    
if __name__ == '__main__':
  tf.app.run()
