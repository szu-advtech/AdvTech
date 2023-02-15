from model import T_CNN
from utils import *
import numpy as np
from absl import flags
import tensorflow as tf
import cv2 as cv

import pprint
import os

flags.DEFINE_integer("epoch",400,"训练批次，测试时无用")
flags.DEFINE_integer("batch_size",1,"每个批次训练集大小")
flags.DEFINE_integer("image_height",112,"输入训练图片的高度")
flags.DEFINE_integer("image_width",112,"输出训练图片的宽度")
flags.DEFINE_integer("label_height",112,"输入参考图片的高度")
flags.DEFINE_integer("label_width",112,"输入参考图片的宽度")
flags.DEFINE_float("learning_rate",0.001,"学习率")
flags.DEFINE_float("beta1",0.5,"使用adam动量梯度下降时的参数")
flags.DEFINE_integer("c_dim",3,"输入图像的色彩通道数")
flags.DEFINE_integer("c_depth_dim",1,"Dimension of depth. [1]")
flags.DEFINE_string("checkpoint_dir","", "参数文件夹路径")
flags.DEFINE_string("sample_dir","sample","样本文件夹")
flags.DEFINE_string("test_data_dir","test","测试时输入的样本")
flags.DEFINE_boolean("is_train",True,"训练——True；测试——False")
FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()

def main(_):

    # 寻找参数文件
    if not os.path.exists(FLAGS.checkpoint_dir):
      os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
      os.makedirs(FLAGS.sample_dir)

    # 生成需要进行测试图像名称
    filenames = os.listdir('')
    data_dir = os.path.join(os.getcwd(),'')
    data = glob.glob(os.path.join(data_dir,"*.jpg"))
    test_data_list = data + glob.glob(os.path.join(data_dir,"*.png"))+glob.glob(os.path.join(data_dir,"*.bmp"))+glob.glob(os.path.join(data_dir,"*.jpeg"))
    print(data_dir,test_data_list)

    # 对每一张图片进行训练得到相应结果
    for ide in range(0,len(test_data_list)):
        image_test = cv.imread(test_data_list[ide],1)
        shape = image_test.shape
        print(ide)
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
          # with tf.device('/cpu:0'):
            srcnn = T_CNN(sess,
                      image_height=shape[0],
                      image_width=shape[1],
                      label_height=FLAGS.label_height,
                      label_width=FLAGS.label_width,
                      batch_size=FLAGS.batch_size,
                      c_dim=FLAGS.c_dim,
                      c_depth_dim=FLAGS.c_depth_dim,
                      checkpoint_dir=FLAGS.checkpoint_dir,
                      sample_dir=FLAGS.sample_dir,
                      test_image_name = test_data_list[ide],
                      id = ide
                      )
            print("--------------------------------")
            print(FLAGS.image_height)
            print("Loop1")
            srcnn.train(FLAGS)
            sess.close()
        tf.compat.v1.get_default_graph().finalize()

if __name__ == '__main__':
  tf.app.run()
