import torch
import torch.nn.functional as F
import cv2
import numpy as np


from statistics import mode, mean


class SaveValues():
    def __init__(self, m):
        # register a hook to save values of activations and gradients
        self.activations = None
        self.gradients = None
        self.forward_hook = m.register_forward_hook(self.hook_fn_act)
        self.backward_hook = m.register_backward_hook(self.hook_fn_grad)

    def hook_fn_act(self, module, input, output):
        self.activations = output

    def hook_fn_grad(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


class CAM(object):
    """ Class Activation Mapping """

    def __init__(self, model, target_layer):
        """
        参数：
            模型：获取CAM的基础模型，该模型具有全局池和完全连接层。
            target_layer:GAP层之前的卷积层
        """

        self.model = model
        self.target_layer = target_layer

        # save values of activations and gradients in target_layer

        for (name, module) in self.model.named_modules():
            if name == self.target_layer:
                self.values = SaveValues(module)

    def forward(self, x, idx=None):
        """
        Args:
            x: input image. shape =>(1, 3, H, W)
        Return:
            heatmap: class activation mappings of the predicted class预测类的类激活映射
        """

        # object classification
        score = self.model(x)

        prob = F.softmax(score, dim=1)

        if idx is None:
            prob, idx = torch.max(prob, dim=1)
            idx = idx.item()
            prob = prob.item()
            print("predicted class ids {}\t probability {}".format(idx, prob))

        # cam can be calculated from the weights of linear layer and activations
        weight_fc = list(
            self.model._modules.get('fc').parameters())[0].to('cpu').data

        cam = self.getCAM(self.values, weight_fc, idx)

        return cam, idx

    def __call__(self, x):
        return self.forward(x)

    def getCAM(self, values, weight_fc, idx):
        '''
        values: the activations and gradients of target_layer 目标层的feature map和梯度
            activations: feature map before GAP.  shape => (1, C, H, W)
        weight_fc: the weight of fully connected layer.  shape => (num_classes, C)全链接层的权重
        idx: predicted class id
        cam: class activation map.  shape => (1, num_classes, H, W)
        '''

        cam = F.conv2d(values.activations, weight=weight_fc[:, :, None, None])
        _, _, h, w = cam.shape

        # class activation mapping only for the predicted class
        # cam is normalized with min-max.
        cam = cam[:, idx, :, :]
        cam -= torch.min(cam)
        cam /= torch.max(cam)
        cam = cam.view(1, 1, h, w)

        return cam.data


class GradCAM(CAM):
    """ Grad CAM """

    def __init__(self, model, target_layer):
        super().__init__(model, target_layer)

        """
        Args:
            model: a base model to get CAM, which need not have global pooling and fully connected layer.
            target_layer: conv_layer you want to visualize 想要可视化的任何一个卷积层
        """

    def forward(self, x, idx=None):
        """
        Args:
            x: input image. shape =>(1, 3, H, W)
            idx: ground truth index => (1, C)
        Return:
            heatmap: class activation mappings of the predicted class
        """

        # anomaly detection
        score = self.model(x)

        prob = F.softmax(score, dim=1)

        if idx is None:
            prob, idx = torch.max(prob, dim=1)
            idx = idx.item()
            prob = prob.item()
            print("predicted class ids {}\t probability {}".format(idx, prob))

        # caluculate cam of the predicted class
        cam = self.getGradCAM(self.values, score, idx)

        return cam, idx

    def __call__(self, x):
        return self.forward(x)

    def getGradCAM(self, values, score, idx):
        '''
        values: the activations and gradients of target_layer
            activations: feature map before GAP.  shape => (1, C, H, W)
        score: the output of the model before softmax
        idx: predicted class id
        cam: class activation map.  shape=> (1, 1, H, W)
        '''

        self.model.zero_grad()

        score[0, idx].backward(retain_graph=True)

        activations = values.activations
        gradients = values.gradients
        n, c, _, _ = gradients.shape
        alpha = gradients.view(n, c, -1).mean(2)
        alpha = alpha.view(n, c, 1, 1)

        # shape => (1, 1, H', W')
        cam = (alpha * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam -= torch.min(cam)
        cam /= torch.max(cam)

        return cam.data


class GradCAMpp(CAM):
    """ Grad CAM plus plus """

    def __init__(self, model, target_layer):
        super().__init__(model, target_layer)
        """
        Args:
            model: a base model
            target_layer: conv_layer you want to visualize
        """

    def forward(self, x, idx=None):
        """
        Args:
            x: input image. shape =>(1, 3, H, W)
        Return:
            heatmap: class activation mappings of predicted classes
        """

        # object classification
        score = self.model(x)

        prob = F.softmax(score, dim=1)

        if idx is None:
            prob, idx = torch.max(prob, dim=1)
            idx = idx.item()
            prob = prob.item()
            print("predicted class ids {}\t probability {}".format(idx, prob))

        # caluculate cam of the predicted class
        cam = self.getGradCAMpp(self.values, score, idx)

        return cam, idx

    def __call__(self, x):
        return self.forward(x)

    def getGradCAMpp(self, values, score, idx):
        '''
        values: the activations and gradients of target_layer
            activations: feature map before GAP.  shape => (1, C, H, W)
        score: the output of the model before softmax. shape => (1, n_classes)
        idx: predicted class id
        cam: class activation map.  shape=> (1, 1, H, W)
        '''

        self.model.zero_grad()

        score[0, idx].backward(retain_graph=True)

        activations = values.activations
        gradients = values.gradients
        n, c, _, _ = gradients.shape

        # calculate alpha
        numerator = gradients.pow(2)
        denominator = 2 * gradients.pow(2)
        ag = activations * gradients.pow(3)
        denominator += ag.view(n, c, -1).sum(-1, keepdim=True).view(n, c, 1, 1)
        denominator = torch.where(
            denominator != 0.0, denominator, torch.ones_like(denominator))
        alpha = numerator / (denominator + 1e-7)

        relu_grad = F.relu(score[0, idx].exp() * gradients)
        weights = (alpha * relu_grad).view(n, c, -1).sum(-1).view(n, c, 1, 1)

        # shape => (1, 1, H', W')
        cam = (weights * activations).sum(1, keepdim=True)
        cam = F.relu(cam)
        cam -= torch.min(cam)
        cam /= torch.max(cam)

        return cam.data


class SmoothGradCAMpp(CAM):
    """ Smooth Grad CAM plus plus """

    def __init__(self, model, target_layer, n_samples=25, stdev_spread=0.15):
        super().__init__(model, target_layer)
        """
        Args:
            model: a base model
            target_layer: conv_layer you want to visualize
            n_sample: the number of samples样本数
            stdev_spread: standard deviationß标准偏差
        """

        self.n_samples = n_samples
        self.stdev_spread = stdev_spread

    def forward(self, x, idx=None):
        """
        Args:
            x: input image. shape =>(1, 3, H, W)
        Return:
            heatmap: class activation mappings of predicted classes
        """

        stdev = self.stdev_spread / (x.max() - x.min())
        std_tensor = torch.ones_like(x) * stdev

        indices = []
        probs = []

        for i in range(self.n_samples):
            self.model.zero_grad()

            x_with_noise = torch.normal(mean=x, std=std_tensor)
            x_with_noise.requires_grad_()

            score = self.model(x_with_noise)

            prob = F.softmax(score, dim=1)

            if idx is None:
                prob, idx = torch.max(prob, dim=1)
                idx = idx.item()
                probs.append(prob.item())

            indices.append(idx)

            score[0, idx].backward(retain_graph=True)

            gradient = self.values.gradients[0].cpu().data.numpy()  # [C,H,W]
            weight = np.mean(gradient, axis=(1, 2))  # [C]

            feature = self.values.activations[0].cpu().data.numpy()  # [C,H,W]

            cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
            cam = np.sum(cam, axis=0)  # [H,W]
            cam = np.maximum(cam, 0)  # ReLU

            # 数值归一化
            cam -= np.min(cam)
            cam /= np.max(cam)
            # resize to 224*224


            # activations = self.values.activations
            # gradients = self.values.gradients
            # n, c, _, _ = gradients.shape

            # calculate alpha
            # numerator = gradients.pow(2)
            # denominator = 2 * gradients.pow(2)
            # ag = activations * gradients.pow(3)
            # denominator += \
            #     ag.view(n, c, -1).sum(-1, keepdim=True).view(n, c, 1, 1)
            # denominator = torch.where(
            #     denominator != 0.0, denominator, torch.ones_like(denominator))
            # alpha = numerator / (denominator + 1e-7)
            #
            # relu_grad = F.relu(score[0, idx].exp() * gradients)
            # weights = \
            #     (alpha * relu_grad).view(n, c, -1).sum(-1).view(n, c, 1, 1)
            #
            # # shape => (1, 1, H', W')
            # cam = (weights * activations).sum(1, keepdim=True)
            # cam = F.relu(cam)
            # cam -= torch.min(cam)
            # cam /= torch.max(cam)
            #


            #TODO cam 是什么？怎么求和
            if i == 0:

                total_cams = cam
            else:
                total_cams += cam  # [H,W]

        #TODO 求total_cam的平均值
        total_cams /= self.n_samples


        total_cams = cv2.resize(total_cams, (224, 224))
        return total_cams
        # idx = mode(indices)
        # prob = mean(probs)
        #
        # print("predicted class ids {}\t probability {}".format(idx, prob))
        # return total_cams.data, idx
        # indexnew = None
        # if indexnew is None:
        #     aa =total_cams.cpu().data.numpy()
        #     indexnew = np.argmax(aa)
        # target = total_cams[0][0]
        # target.backward()

        # gradient = self.values.gradients[0].cpu().data.numpy()  # [C,H,W]
        # weight = np.mean(gradient, axis=(1, 2))  # [C]
        #
        # feature = self.values.activations[0].cpu().data.numpy()  # [C,H,W]
        #
        # cam = feature * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
        # cam = np.sum(cam, axis=0)  # [H,W]
        # cam = np.maximum(cam, 0)  # ReLU
        #
        # # 数值归一化
        # cam -= np.min(cam)
        # cam /= np.max(cam)
        # # resize to 224*224
        # cam = cv2.resize(cam, (224, 224))
        # return cam

    def __call__(self, x):
        return self.forward(x)
