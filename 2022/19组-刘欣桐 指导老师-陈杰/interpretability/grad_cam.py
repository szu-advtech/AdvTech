# -*- coding: utf-8 -*-
"""
Created on 2022/10/4 上午9:37

@author:lxt

"""
import numpy as np
import cv2


class GradCAM(object):
    """
    1: 网络不更新梯度,输入需要梯度更新
    2: 使用目标类别的得分做反向传播
    """

    def __init__(self, net, layer_name):
        """
        self.net = net#网络模型
        self.layer_name = layer_name#层名
        self.feature = None#特征图
        self.gradient = None#存放gradient图
        self.net.eval()#网络值
        self.handlers = []#任务列表
        self._register_hook()# 保留中间变量的导数
          """
        self.net = net
        self.layer_name = layer_name
        self.feature = None
        self.gradient = None
        self.net.eval()#网络值
        self.handlers = []#任务列表
        self._register_hook()# 保留中间变量的导数

    def _get_features_hook(self, module, input, output):
        """
         特征图的函数
        """
        self.feature = output
        print("feature shape:{}".format(output.size()))

    def _get_grads_hook(self, module, input_grad, output_grad):
        """
        获得梯度的函数
        :param input_grad: tuple, input_grad[0]: None
                                   input_grad[1]: weight
                                   input_grad[2]: bias
        :param output_grad:tuple,长度为1
        ：hook作用：获取某些变量的中间结果的。Pytorch默认在反向传播中不保留中间梯度，会自动舍弃图计算的中间结果，所以想要获取这些数值就需要使用hook函数。hook函数在使用后应及时删除，以避免每次都运行钩子增加运行负载。
        :return:
        """
        self.gradient = output_grad[0]

    def _register_hook(self):
        for (name, module) in self.net.named_modules():
            if name == self.layer_name:
                self.handlers.append(module.register_forward_hook(self._get_features_hook))#modelue方法的前向传播hook函数
                self.handlers.append(module.register_backward_hook(self._get_grads_hook))#反向传播hook函数

    def remove_handlers(self):
        #删除任务
        for handle in self.handlers:
            handle.remove()

    def __call__(self, inputs, index):
        """

        :param inputs: [1,3,H,W] -batch size*channel*H*W
        :param index: class id
        :return:
        """
        self.net.zero_grad() #梯度置零
        output = self.net(inputs)  # [1,num_classes]
        #取最大类别的值作为target，这样计算的结果是模型对该类最感兴趣的cam图
        if index is None:
            index = np.argmax(output.cpu().data.numpy())
        target = output[0][index]#？
        target.backward()#得到梯度图

        gradient = self.gradient[0].cpu().data.numpy()  # [C,H,W]
        weight = np.mean(gradient, axis=(1, 2))  # [C]  先按照2再按照1平均

        feature = self.feature[0].cpu().data.numpy()  # [C,H,W]

        cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        cam = np.sum(cam, axis=0)  # [H,W]   C求和
        cam = np.maximum(cam, 0)  # ReLU

        # 数值归一化
        cam -= np.min(cam)
        cam /= np.max(cam)
        # resize to 224*224
        cam = cv2.resize(cam, (224, 224))
        return cam


class GradCamPlusPlus(GradCAM):
    def __init__(self, net, layer_name):
        super(GradCamPlusPlus, self).__init__(net, layer_name)

    def __call__(self, inputs, index):
        """

        :param inputs: [1,3,H,W]
        :param index: class id
        :return:
        """
        self.net.zero_grad() # 梯度置0
        output = self.net(inputs)  # [1,num_classes]
        if index is None:
            index = np.argmax(output.cpu().data.numpy())
        target = output[0][index]
        target.backward()

        gradient = self.gradient[0].cpu().data.numpy()  # [C,H,W]
        gradient = np.maximum(gradient, 0.)  # ReLU
        indicate = np.where(gradient > 0, 1., 0.)  # 示性函数 np.where(condition,x,y)
        norm_factor = np.sum(gradient, axis=(1, 2))  # [C]归一化  channel方向做归一化
        for i in range(len(norm_factor)):
            norm_factor[i] = 1. / norm_factor[i] if norm_factor[i] > 0. else 0.  # 避免除零
        alpha = indicate * norm_factor[:, np.newaxis, np.newaxis]  # [C,H,W] #np.newaxis增加维度

        weight = np.sum(gradient * alpha, axis=(1, 2))  # [C]  alpha*ReLU(gradient)

        feature = self.feature[0].cpu().data.numpy()  # [C,H,W]

        cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        cam = np.sum(cam, axis=0)  # [H,W]
        # cam = np.maximum(cam, 0)  # ReLU

        # 数值归一化
        cam -= np.min(cam)
        cam /= np.max(cam)
        # resize to 224*224
        cam = cv2.resize(cam, (224, 224))
        return cam
