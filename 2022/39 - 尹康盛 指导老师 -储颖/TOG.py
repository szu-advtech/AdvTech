import torchvision
from PIL import Image

import numpy as np
import torch
import torch.nn as nn

from torchattacks.attack import Attack

from nets.yolo_training import YOLOLoss
from utils.utils import get_classes, get_anchors


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
        images, meta = self.Pretreat(images)
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

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() - self.alpha * grad.sign()
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


    def Pretreat(self, images):
        input_shape = [416, 416]
        # ------------------------------#
        #   获得图像的高宽与目标高宽
        # ------------------------------#
        image_datas = []
        h, w = input_shape
        a, b, ih, iw = images.shape
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2
        for n in range(len(images)):
            image = images[n, :, :, :]
            # print(image)
            image = torchvision.transforms.ToPILImage()(image.float())
            # print(image)
            image = image.resize((nw, nh), Image.BICUBIC)
            # print(image)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)
            image_datas.append(image_data)
            # print(image_data)
        image_datas = torch.from_numpy(np.array(image_datas)).type(torch.FloatTensor).permute(0, 3, 1, 2).contiguous()
        meta = ((w-nw) // 2, nw + (w - nw) // 2, (h - nh) // 2, nh + (h - nh) // 2 )
        return image_datas, meta


