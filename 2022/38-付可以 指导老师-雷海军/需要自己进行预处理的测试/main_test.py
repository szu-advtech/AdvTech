from model import T_CNN
from utils import *
import numpy as np
import tensorflow as tf

import pprint
import os

flags = tf.app.flags
flags.DEFINE_integer("epoch", 400, "训练批次，测试时无用")
flags.DEFINE_integer("batch_size", 1, "每个批次训练集大小")
flags.DEFINE_integer("image_height", 112, "输入训练图片的高度")
flags.DEFINE_integer("image_width", 112, "输出训练图片的宽度")
flags.DEFINE_integer("label_height", 112 ,"输入参考图片的高度")
flags.DEFINE_integer("label_width", 112, "输入参考图片的宽度")
flags.DEFINE_float("learning_rate", 0.001, "学习率")
flags.DEFINE_float("beta1", 0.5, "使用adam动量梯度下降时的参数")
flags.DEFINE_integer("c_dim", 3, "输入图像的色彩通道数")
flags.DEFINE_integer("c_depth_dim", 1, "Dimension of depth. [1]")
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

  # 生成需要进行测试图像名称
  filenames = os.listdir('test_real')
  data_dir = os.path.join(os.getcwd(), 'test_real')
  data = glob.glob(os.path.join(data_dir, "*.png"))
  test_data_list = data + glob.glob(os.path.join(data_dir, "*.jpg"))+glob.glob(os.path.join(data_dir, "*.bmp"))+glob.glob(os.path.join(data_dir, "*.jpeg"))
  # 生成测试图像经过白平衡处理后的图像名称
  filenames1 = os.listdir('wb_real')
  data_dir1 = os.path.join(os.getcwd(), 'wb_real')
  data1 = glob.glob(os.path.join(data_dir1, "*.png"))
  test_data_list1 = data1 + glob.glob(os.path.join(data_dir1, "*.jpg"))+glob.glob(os.path.join(data_dir1, "*.bmp"))+glob.glob(os.path.join(data_dir1, "*.jpeg"))
  # 生成测试图像经过CLAHE处理后的图像名称
  filenames2 = os.listdir('ce_real')
  data_dir2 = os.path.join(os.getcwd(), 'ce_real')
  data2 = glob.glob(os.path.join(data_dir2, "*.png"))
  test_data_list2 = data2 + glob.glob(os.path.join(data_dir2, "*.jpg"))+glob.glob(os.path.join(data_dir2, "*.bmp"))+glob.glob(os.path.join(data_dir2, "*.jpeg"))
  # 生成测试图像经过伽马校正处理后的图像名称
  filenames3 = os.listdir('gc_real')
  data_dir3 = os.path.join(os.getcwd(), 'gc_real')
  data3 = glob.glob(os.path.join(data_dir3, "*.png"))
  test_data_list3 = data3 + glob.glob(os.path.join(data_dir3, "*.jpg"))+glob.glob(os.path.join(data_dir3, "*.bmp"))+glob.glob(os.path.join(data_dir3, "*.jpeg"))

  # 对每一张图片进行训练得到相应结果
  for ide in range(0,len(test_data_list)):
    image_test =  get_image(test_data_list[ide],is_grayscale=False)
    wb_test =  get_image(test_data_list1[ide],is_grayscale=False)
    ce_test =  get_image(test_data_list2[ide],is_grayscale=False)
    gc_test =  get_image(test_data_list3[ide],is_grayscale=False)
    shape = image_test.shape
    tf.reset_default_graph()
    with tf.Session() as sess:
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
                  test_wb_name = test_data_list1[ide],
                  test_ce_name = test_data_list2[ide],
                  test_gc_name = test_data_list3[ide],
                  id = ide
                  )

        srcnn.train(FLAGS)
        sess.close()
    tf.get_default_graph().finalize()
if __name__ == '__main__':
  tf.app.run()
