import torchvision
from PIL import Image

import numpy as np
import torch
import torch.nn as nn

from torchattacks.attack import Attack

from nets.yolo_training import YOLOLoss
from utils.utils import get_classes, get_anchors

import os

class TOG(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=10, random_start=True)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model, eps=8/255,
                 alpha=2/255, steps=10, random_start=True):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ['default', 'targeted']

    def forward(self, images, targets):
        r"""
        Overridden.
        """
        os.environ["CUDA_VISIBLE_DEVICES"] = '7'
        images = images.clone().detach().to(self.device)
        targets = [ann.clone().detach().to(self.device) for ann in targets]



        loss = self.yololoss()

        adv_images = images.clone().detach()
        v_w = torch.zeros(adv_images.shape)
        s_w = torch.zeros(adv_images.shape)
        cost = 0
        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            # a, b, c, d = adv_images.shape
            adv_images.requires_grad = True
            # imgs = self.Pretreat(adv_images)
            # print(imgs.shape)
            outputs = self.get_logits(adv_images)
            # ----------------------#
            #   计算损失
            # ----------------------#
            loss_value_all = 0
            for l in range(len(outputs)):
                loss_item = loss(l, outputs[l], targets)
                loss_value_all += loss_item
            cost = loss_value_all
            print(cost.item())

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() - self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images

    def yololoss(self):
        anchors_path = 'model_data/yolo_anchors.txt'
        classes_path = 'model_data/voc_classes.txt'
        # ----------------------------------------------------#
        #   获取classes和anchor
        # ----------------------------------------------------#
        class_names, num_classes = get_classes(classes_path)
        anchors, num_anchors = get_anchors(anchors_path)
        # ----------------------------------------------------#
        #   输入图像size
        # ----------------------------------------------------#
        input_shape = [416, 416]
        # ----------------------------------------------------#
        #   Cuda    是否使用Cuda
        #           没有GPU可以设置成False
        # ---------------------------------#
        Cuda = True
        # ---------------------------------------------------------------------#
        #   anchors_mask    用于帮助代码找到对应的先验框，一般不修改。
        # ---------------------------------------------------------------------#
        anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        # ----------------------#
        #   获得损失函数
        # ----------------------#
        yolo_loss = YOLOLoss(anchors, num_classes, input_shape, Cuda, anchors_mask)

        return yolo_loss

