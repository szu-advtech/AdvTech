import torch

import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

import torch.nn as nn
from torch.nn import init
from torch.nn import functional
from torchvision import models
from torch.autograd import Variable
# import pretrainedmodels


######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f = False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return x,f
        else:
            x = self.classifier(x)
            return x

class AvgPooling(nn.Module):
    def __init__(self, input_feature_size, embeding_fea_size=1024, dropout=0.5):
        super(self.__class__, self).__init__()

        # embeding
        self.embeding_fea_size = embeding_fea_size
        self.embeding = nn.Linear(input_feature_size, embeding_fea_size)
        self.embeding_bn = nn.BatchNorm1d(embeding_fea_size)
        init.kaiming_normal(self.embeding.weight, mode='fan_out')
        init.constant(self.embeding.bias, 0)
        init.constant(self.embeding_bn.weight, 1)
        init.constant(self.embeding_bn.bias, 0)
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs):
        # net = inputs.mean(dim = 1)
        net = inputs
        # eval_feas = functional.normalize(net, p=2, dim=1)
        net = self.embeding(net)
        net = self.embeding_bn(net)
        net = functional.normalize(net, p=2, dim=1)
        net = self.drop(net)
        # return net, eval_feas
        return net

# Define the ResNet50-based Model
class ft_net(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2):
        super(ft_net, self).__init__()

        model_ft = models.resnet50(pretrained=True)
        # model_ft=torch.load('saved_res50.pkl')
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        # self.average_pooling = AvgPooling(2048, embeding_fea_size = 114)
        self.classifier = ClassBlock(2048, class_num, droprate)

    def forward(self, x):
        # print(x.size())
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x

class Backbone(nn.Module):

    def __init__(self, dropout=0,  embeding_fea_size=1024, fixed_layer=True):
        super(self.__class__, self).__init__()
        self.CNN = resnet50(dropout=dropout, fixed_layer=fixed_layer)
        self.avg_pooling = AvgPooling(input_feature_size=2048, embeding_fea_size = embeding_fea_size, dropout=dropout)

    def forward(self, x):
        # Assumed that we have reshaped (batch, samples, ...) to (batch * samples, ...)
        assert len(x.data.shape) == 4
        # oriShape = x.data.shape
        # x = x.view(-1, oriShape[2], oriShape[3], oriShape[4])

        # resnet encoding
        resnet_feature = self.CNN(x)

        # reshape back into (batch, samples, ...) TODO Should be back to batch*samples ?
        # resnet_feature = resnet_feature.view(oriShape[0], 1, -1)

        # avg pooling
        # output = resnet_feature
        output = self.avg_pooling(resnet_feature)
        return output


## The resnet50 model
# __all__ = ['ResNet', 'resnet50']


class ResNet(nn.Module):
    __factory = {
        18: models.resnet18,
        34: models.resnet34,
        50: models.resnet50,
        101: models.resnet101,
        152: models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0, fixed_layer=True):
        super(ResNet, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling

        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)

        self.base = ResNet.__factory[depth](pretrained=pretrained)

        if fixed_layer:
            # fix layers [conv1 ~ layer2]
            fixed_names = []
            for name, module in self.base._modules.items():
                if name == "layer3":
                    assert fixed_names == ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2"]
                    break
                fixed_names.append(name)
                for param in module.parameters():
                    param.requires_grad = False

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = self.base.fc.in_features

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal(self.feat.weight, mode='fan_out')
                init.constant(self.feat.bias, 0)
                init.constant(self.feat_bn.weight, 1)
                init.constant(self.feat_bn.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes)
                init.normal(self.classifier.weight, std=0.001)
                init.constant(self.classifier.bias, 0)

        if not self.pretrained:
            self.reset_params()

    def forward(self, x):

        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            x = module(x)

        if self.cut_at_pooling:
            return x

        x = functional.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)

        return x

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)


def resnet50(**kwargs):
    return ResNet(50, **kwargs)


'''
# debug model structure
# Run this code with:
python model.py
'''
if __name__ == '__main__':
# Here I left a simple forward function.
# Test the model, before you train it. 
#     net = ft_net(751, stride=1)
    net = Backbone(embeding_fea_size=512)
    net.classifier = nn.Sequential()
    # print(net)
    input = Variable(torch.FloatTensor(8, 3, 256, 128))
    output = net(input)
    print('net output size:')
    # print(output.shape)
    print(output.shape)
