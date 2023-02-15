from utils import ( 
  imsave,
  prepare_data
)

import time
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from ops import *
import vgg

class T_CNN(object):

  def __init__(self, 
               sess, 
               image_height=230,
               image_width=310,
               label_height=230, 
               label_width=310,
               batch_size=2,
               c_dim=3, 
               checkpoint_dir=None, 
               sample_dir=None
               ):
    print("=================1====================")
    self.sess = sess
    self.is_grayscale = (c_dim == 1)
    self.image_height = image_height
    self.image_width = image_width
    self.label_height = label_height
    self.label_width = label_width
    self.batch_size = batch_size
    self.dropout_keep_prob=0.5


    self.c_dim = c_dim
    self.df_dim = 64
    self.checkpoint_dir = checkpoint_dir
    self.sample_dir = sample_dir
    self.vgg_dir='vgg_pretrained/imagenet-vgg-verydeep-19.mat'
    self.CONTENT_LAYER = 'relu5_4'
    self.build_model()

  def build_model(self):
    print("=================2====================")
    self.images = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='images')
    self.images_wb = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='images_wb')
    self.images_ce = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='images_ce')
    self.images_gc = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='images_gc')
    self.labels_image = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='labels_image')


    self.images_test = tf.placeholder(tf.float32, [1, self.image_height, self.image_width, self.c_dim], name='images_test')
    self.images_test_wb = tf.placeholder(tf.float32, [1, self.image_height, self.image_width, self.c_dim], name='images_test_wb')
    self.images_test_ce = tf.placeholder(tf.float32, [1, self.image_height, self.image_width, self.c_dim], name='images_test_ce')
    self.images_test_gc = tf.placeholder(tf.float32, [1, self.image_height, self.image_width, self.c_dim], name='images_test_gc')
    self.labels_test = tf.placeholder(tf.float32, [1,self.label_height,self.label_width, self.c_dim], name='labels_test')
    
    self.pred_h1= self.model()
    
    print('^^^^^^^^^^^^^^^^^^^^^^^^^^^')
    print(self.pred_h1)

    # 为了使训练结果更好，这里使用了vgg19对网络训练的结果再进行训练
    self.enhanced_texture_vgg1 = vgg.net(self.vgg_dir, vgg.preprocess(self.pred_h1 * 255))
    self.labels_texture_vgg = vgg.net(self.vgg_dir, vgg.preprocess(self.labels_image* 255))
    self.loss_texture1 =tf.reduce_mean(tf.square(self.enhanced_texture_vgg1[self.CONTENT_LAYER]-self.labels_texture_vgg[self.CONTENT_LAYER]))
    
    self.loss_h1= tf.reduce_mean(tf.abs(self.labels_image-self.pred_h1))
    self.loss = 0.05*self.loss_texture1+ self.loss_h1
    t_vars = tf.trainable_variables()

    self.saver = tf.train.Saver(max_to_keep=0)
    
  def train(self, config):
    print("=================3====================")
    if config.is_train:
      # 如果是有验证集，那么就读取下面指定五个训练数据集的文件夹和五个验证数据集的文件夹
      data_train_list = prepare_data(self.sess, dataset="input_train")
      data_wb_train_list = prepare_data(self.sess, dataset="input_wb_train")
      data_ce_train_list = prepare_data(self.sess, dataset="input_ce_train")
      data_gc_train_list = prepare_data(self.sess, dataset="input_gc_train")
      image_train_list = prepare_data(self.sess, dataset="gt_train")

      data_test_list = prepare_data(self.sess, dataset="input_test")
      data_wb_test_list = prepare_data(self.sess, dataset="input_wb_test")
      data_ce_test_list = prepare_data(self.sess, dataset="input_ce_test")
      data_gc_test_list = prepare_data(self.sess, dataset="input_gc_test")
      image_test_list = prepare_data(self.sess, dataset="gt_test")

      # 将图像数据按相同的模式进行打乱
      # 这里设置seed是为了使打乱后的图片仍然是对应的
      seed = 568
      np.random.seed(seed)
      np.random.shuffle(data_train_list)
      np.random.seed(seed)
      np.random.shuffle(data_wb_train_list)
      np.random.seed(seed)
      np.random.shuffle(data_ce_train_list)
      np.random.seed(seed)
      np.random.shuffle(data_gc_train_list)
      np.random.seed(seed)
      np.random.shuffle(image_train_list)

    else:
      # 如果是没有验证集，那么就只读取下面指定五个文件夹种的图像数据
      data_test_list = prepare_data(self.sess, dataset="input_test")
      data_wb_test_list = prepare_data(self.sess, dataset="input_wb_test")
      data_ce_test_list = prepare_data(self.sess, dataset="input_ce_test")
      data_gc_test_list = prepare_data(self.sess, dataset="input_gc_test")
      image_test_list = prepare_data(self.sess, dataset="gt_test")



    # sample_data_files = data_test_list[16:20]
    # sample_wb_data_files = data_wb_test_list[16:20]
    # sample_ce_data_files = data_ce_test_list[16:20]
    # sample_gc_data_files = data_gc_test_list[16:20]
    # sample_image_files = image_test_list[16:20]

    # sample_data = [
    #       get_image(sample_data_file,
    #                 is_grayscale=self.is_grayscale) for sample_data_file in sample_data_files]
    # sample_lable_image = [
    #       get_image(sample_image_file,
    #                 is_grayscale=self.is_grayscale) for sample_image_file in sample_image_files]

    # sample_inputs_data = np.array(sample_data).astype(np.float32)
    # sample_inputs_lable_image = np.array(sample_lable_image).astype(np.float32)


    self.train_op = tf.train.AdamOptimizer(config.learning_rate,0.9).minimize(self.loss)
    tf.global_variables_initializer().run()
    
    
    counter = 0
    start_time = time.time()

    # 加载已有的参数文件，如果没有就重新建立
    if self.load(self.checkpoint_dir):
      print(" [*] 参数加载成功！！！")
    else:
      print(" [!] 参数加载失败？？？")

    if config.is_train:
      print("Training...")
      print(config.epoch)
      loss = np.ones(config.epoch)

      for ep in range(config.epoch):
        
        batch_idxs = len(data_train_list) // config.batch_size
        for idx in range(0, batch_idxs):
          # 根据设置的batch_size对数据集进行划分读取
          batch_files       = data_train_list[idx*config.batch_size:(idx+1)*config.batch_size]
          batch_files_wb       = data_wb_train_list[idx*config.batch_size:(idx+1)*config.batch_size]
          batch_files_ce       = data_ce_train_list[idx*config.batch_size:(idx+1)*config.batch_size]
          batch_files_gc       = data_gc_train_list[idx*config.batch_size:(idx+1)*config.batch_size]
          batch_image_files = image_train_list[idx*config.batch_size : (idx+1)*config.batch_size]


          batch_ = [
          get_image(batch_file,
                    is_grayscale=self.is_grayscale) for batch_file in batch_files]
          batch_wb = [
          get_image(batch_wb_file,
                    is_grayscale=self.is_grayscale) for batch_wb_file in batch_files_wb]
          batch_ce = [
          get_image(batch_ce_file,
                    is_grayscale=self.is_grayscale) for batch_ce_file in batch_files_ce]
          batch_gc = [
          get_image(batch_gc_file,
                    is_grayscale=self.is_grayscale) for batch_gc_file in batch_files_gc]
          batch_labels_image = [
          get_image(batch_image_file,
                    is_grayscale=self.is_grayscale) for batch_image_file in batch_image_files]
          
          # print(batch_)
          # 将数据转换为array格式
          batch_input = np.array(batch_).astype(np.float32)
          batch_wb_input = np.array(batch_wb).astype(np.float32)
          batch_ce_input = np.array(batch_ce).astype(np.float32)
          batch_gc_input = np.array(batch_gc).astype(np.float32)
          batch_image_input = np.array(batch_labels_image).astype(np.float32)

          # 使用这一批量样本进行训练，并将计数加一
          counter += 1
          _, err = self.sess.run([self.train_op, self.loss ], feed_dict={self.images: batch_input, self.images_wb: batch_wb_input, self.images_ce: batch_ce_input, self.images_gc: batch_gc_input, self.labels_image:batch_image_input})

          if counter % 100 == 0:
            print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" \
              % ((ep+1), counter, time.time()-start_time, err ))
          
          # 如果是该次epoch的最后一次训练，使用验证机进行验证
          if idx  == batch_idxs-1: 
            batch_test_idxs = len(data_test_list) // config.batch_size
            err_test =  np.ones(batch_test_idxs)
            for idx_test in range(0,batch_test_idxs):

              sample_data_files = data_train_list[idx_test*config.batch_size:(idx_test+1)*config.batch_size]
              sample_wb_files = data_wb_train_list[idx_test*config.batch_size : (idx_test+1)*config.batch_size]
              sample_ce_files = data_ce_train_list[idx_test*config.batch_size : (idx_test+1)*config.batch_size]
              sample_gc_files = data_gc_train_list[idx_test*config.batch_size : (idx_test+1)*config.batch_size]
              sample_image_files = image_train_list[idx_test*config.batch_size : (idx_test+1)*config.batch_size]
             
              sample_data = [get_image(sample_data_file,
                            is_grayscale=self.is_grayscale) for sample_data_file in sample_data_files]
              sample_wb_image = [get_image(sample_wb_file,
                                    is_grayscale=self.is_grayscale) for sample_wb_file in sample_wb_files]
              sample_ce_image = [get_image(sample_ce_file,
                                    is_grayscale=self.is_grayscale) for sample_ce_file in sample_ce_files]
              sample_gc_image = [get_image(sample_gc_file,
                                    is_grayscale=self.is_grayscale) for sample_gc_file in sample_gc_files]

              sample_lable_image = [get_image(sample_image_file,
                                    is_grayscale=self.is_grayscale) for sample_image_file in sample_image_files]

              sample_inputs_data = np.array(sample_data).astype(np.float32)
              sample_inputs_wb_image = np.array(sample_wb_image).astype(np.float32)
              sample_inputs_ce_image = np.array(sample_ce_image).astype(np.float32)
              sample_inputs_gc_image = np.array(sample_gc_image).astype(np.float32)
              sample_inputs_lable_image = np.array(sample_lable_image).astype(np.float32)


              err_test[idx_test] = self.sess.run(self.loss, feed_dict={self.images: sample_inputs_data, self.images_wb: sample_inputs_wb_image, self.images_ce: sample_inputs_ce_image, self.images_gc: sample_inputs_gc_image,self.labels_image:sample_inputs_lable_image})    

            loss[ep]=np.mean(err_test)
            # print(loss)
            self.save(config.checkpoint_dir, counter)


      
  def model(self):
    print("=================4====================")
    with tf.variable_scope("main_branch") as scope3: 
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
      print(wb1)

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

    return output1
 
  # 保存参数
  def save(self, checkpoint_dir, step):
    model_name = "coarse.model"
    model_dir = "%s_%s" % ("coarse", self.label_height)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)

  # 加载参数
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
