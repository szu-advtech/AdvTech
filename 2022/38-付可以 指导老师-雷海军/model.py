import utils

import cv2 as cv

import time
import os
import numpy as np
import tensorflow as tf
from ops import *

class T_CNN(object):

  def __init__(self,
               sess,
               image_height=460,
               image_width=620,
               label_height=460,
               label_width=620,
               batch_size=1,
               c_dim=3,
               c_depth_dim=1,
               checkpoint_dir=None,
               sample_dir=None,
               test_image_name = None,
               test_wb_name = None,
               test_ce_name = None,
               test_gc_name = None,
               id = None
               ):

    self.sess = sess
    self.is_grayscale = (c_dim == 1)
    print("&&&&&&&&&&&&&&&&&&&&&&")
    print(image_height)
    self.image_height = image_height
    self.image_width = image_width
    self.label_height = label_height
    self.label_width = label_width
    self.batch_size = batch_size
    self.dropout_keep_prob=0.5
    self.test_image_name = test_image_name
    self.test_wb_name = test_wb_name
    self.test_ce_name = test_ce_name
    self.test_gc_name = test_gc_name
    self.id = id
    self.c_dim = c_dim
    self.df_dim = 64
    self.checkpoint_dir = checkpoint_dir
    self.sample_dir = sample_dir
    self.build_model()

  def build_model(self):
    self.images = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='images')
    self.images_wb = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='images_wb')
    self.images_ce = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='images_ce')
    self.images_gc = tf.compat.v1.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='images_gc')
    self.pred_h = self.model()
    self.saver = tf.compat.v1.train.Saver()

  def train(self, config):

    # 根据传入的文件名称对图片进行读取
    image_testm = cv.imread(self.test_image_name, cv.IMREAD_COLOR)
    image_test = cv.cvtColor(image_testm, cv.COLOR_BGR2RGB) / 255
    shape = image_test.shape
    expand_test = image_test[np.newaxis,:,:,:]
    expand_zero = np.zeros([self.batch_size-1,shape[0],shape[1],shape[2]])
    batch_test_image = np.append(expand_test,expand_zero,axis = 0)

    # 生成三类增强后的图像，并对其尺寸进行修改
    # 生成白平衡图像
    wb_test = white_balance(image_testm, 0.5)
    cv.waitKey(0)
    cv.destroyAllWindows()
    wb_test = cv.cvtColor(wb_test, cv.COLOR_BGR2RGB) / 255
    shape = wb_test.shape
    expand_test = wb_test[np.newaxis,:,:,:]
    expand_zero = np.zeros([self.batch_size-1,shape[0],shape[1],shape[2]])
    batch_test_wb = np.append(expand_test,expand_zero,axis = 0)


    # 生成CLAHE图像
    ce_test = cv.cvtColor(image_testm,cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    ce_test = clahe.apply(ce_test)
    ce_test = cv.cvtColor(ce_test,cv.COLOR_GRAY2BGR)
    ce_test = cv.cvtColor(ce_test,cv.COLOR_BGR2RGB) / 255
    shape = ce_test.shape
    expand_test = ce_test[np.newaxis,:,:,:]
    expand_zero = np.zeros([self.batch_size-1,shape[0],shape[1],shape[2]])
    batch_test_ce = np.append(expand_test,expand_zero,axis = 0)

    # 生成gamma校正图像
    gc_test = adjust_gamma(image_testm,0.7)
    gc_test = cv.cvtColor(gc_test,cv.COLOR_BGR2RGB) / 255
    shape = gc_test.shape
    expand_test = gc_test[np.newaxis,:,:,:]
    expand_zero = np.zeros([self.batch_size-1,shape[0],shape[1],shape[2]])
    batch_test_gc = np.append(expand_test,expand_zero,axis = 0)
    print(self.test_wb_name)

    tf.compat.v1.global_variables_initializer().run()

    counter = 0
    start_time = time.time()

    if self.load(self.checkpoint_dir):
      print(" [*] 加载参数文件成功！！！")
    else:
      print(" [!] 加载参数文件失败？？？")
    start_time = time.time()
    result_h  = self.sess.run(self.pred_h, feed_dict={self.images: batch_test_image,self.images_wb: batch_test_wb,self.images_ce: batch_test_ce,self.images_gc: batch_test_gc})
    all_time = time.time()
    final_time=all_time - start_time
    print(final_time)


    _,h ,w , c = result_h.shape
    # # 将结果矩阵转换成图像三通道矩阵，并进行保存
    for id in range(0,1):
        result_h0 = result_h[id,:,:,:].reshape(h , w , 3)
        result_h0 = result_h0.squeeze()
        image_path0 = os.path.join(os.getcwd(),"sample")
        output_name = os.path.basename(self.test_image_name)
        # print(image_path0, output_name)
        image_path = os.path.join(image_path0, output_name)
        imsave_lable(result_h0, image_path)
        print(image_path)


  def model(self):
    with tf.compat.v1.variable_scope("main_branch") as scope:
      # 主八层网络
      conb0 = tf.concat(axis = 3, values = [self.images,self.images_wb,self.images_ce,self.images_gc])
      conv_wb1 = tf.nn.relu(conv2d(conb0, 16,128, k_h=7, k_w=7, d_h=1, d_w=1,name="conv2wb_1"))
      conv_wb2 = tf.nn.relu(conv2d(conv_wb1, 128,128, k_h=5, k_w=5, d_h=1, d_w=1,name="conv2wb_2"))
      conv_wb3 = tf.nn.relu(conv2d(conv_wb2, 128,128, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2wb_3"))
      conv_wb4 = tf.nn.relu(conv2d(conv_wb3, 128,64, k_h=1, k_w=1, d_h=1, d_w=1,name="conv2wb_4"))
      conv_wb5 = tf.nn.relu(conv2d(conv_wb4, 64,64, k_h=7, k_w=7, d_h=1, d_w=1,name="conv2wb_5"))
      conv_wb6 = tf.nn.relu(conv2d(conv_wb5, 64,64, k_h=5, k_w=5, d_h=1, d_w=1,name="conv2wb_6"))
      conv_wb7 = tf.nn.relu(conv2d(conv_wb6, 64,64, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2wb_7"))
      conv_wb77 =tf.nn.sigmoid(conv2d(conv_wb7, 64,3, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2wb_77"))

      # FTU网络
      conb00 = tf.concat(axis = 3, values = [self.images,self.images_wb])
      conv_wb9 = tf.nn.relu(conv2d(conb00, 3,32, k_h=7, k_w=7, d_h=1, d_w=1,name="conv2wb_9"))
      conv_wb10 = tf.nn.relu(conv2d(conv_wb9, 32,32, k_h=5, k_w=5, d_h=1, d_w=1,name="conv2wb_10"))
      wb1 =tf.nn.relu(conv2d(conv_wb10, 32,3, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2wb_11"))

      conb11 = tf.concat(axis = 3, values = [self.images,self.images_ce])
      conv_wb99 = tf.nn.relu(conv2d(conb11, 3,32, k_h=7, k_w=7, d_h=1, d_w=1,name="conv2wb_99"))
      conv_wb100 = tf.nn.relu(conv2d(conv_wb99, 32,32, k_h=5, k_w=5, d_h=1, d_w=1,name="conv2wb_100"))
      ce1 =tf.nn.relu(conv2d(conv_wb100, 32,3, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2wb_111"))

      conb111 = tf.concat(axis = 3, values = [self.images,self.images_gc])
      conv_wb999 = tf.nn.relu(conv2d(conb111, 3,32, k_h=7, k_w=7, d_h=1, d_w=1,name="conv2wb_999"))
      conv_wb1000 = tf.nn.relu(conv2d(conv_wb999, 32,32, k_h=5, k_w=5, d_h=1, d_w=1,name="conv2wb_1000"))
      gc1 =tf.nn.relu(conv2d(conv_wb1000, 32,3, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2wb_1111"))

      # 将主网络输出图像按通道进行切分
      weight_wb,weight_ce,weight_gc=tf.split(conv_wb77,3,3)
      # 将上面切分得到的与FTU得到的三个置信图进行结合
      output1=tf.add(tf.add(tf.multiply(wb1,weight_wb),tf.multiply(ce1,weight_ce)),tf.multiply(gc1,weight_gc))
      print(output1)
    return output1

  # 加载参数文件
  def load(self, checkpoint_dir):
    print(" [*] 正在读取参数文件！！！")
    model_dir = "%s_%s" % ("coarse", self.label_height)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        return True
    else:
        return False

