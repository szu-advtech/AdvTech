"""Model definition."""

from torch import nn
from transforms import GroupMultiScaleCrop
from transforms import GroupRandomHorizontalFlip
from transforms import GroupRation
from transforms import GroupScale2
import torchvision
from transforms import GroupTranslate
from transforms import GroupCenterCrop
import torch
from nets.facenet import Facenet
import numpy as np
from MI_network2 import MI_network
# from Vgg_16 import Vgg_face
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class Model(nn.Module):
    def __init__(self, num_class, num_segments, representation,
                 base_model='resnet152'):
        super(Model, self).__init__()
        self._representation = representation
        self.num_segments = num_segments
        self.num_classes=num_class
        self.input_size=1024
        self.hidden_size=1024
        self._MMI=MI_network()
        self.backbone="mobilenet"
        self.model_path="/data/jlzhang/Py_protect/2022-08-05/facenet-pytorch-main/model_data/facenet_mobilenet.pth"
        self.num_layers=1 if self._representation not in ['residual','mv'] else 1

        print(("""
Initializing model:
    base model:         {}.
    input_representation:     {}.
    num_class:          {}.
    num_segments:       {}.
        """.format(base_model, self._representation, num_class, self.num_segments)))

        self._prepare_base_model(base_model)

        self._prepare_tsn(num_class)  #为了设置resnet 最后的全连接层的输出通道的
        # self._prepare_tsn2(num_class)
    #
    def _prepare_tsn(self, num_class):

        # feature_dim = getattr(self.base_model, 'fc').in_features  #利用in_features 获得输入 然后下一步执行修改输出。
        # setattr(self.base_model, 'fc', nn.Linear(feature_dim, num_class))
        # out_dim=getattr(self.resnet,'layer3')[0]
        # out_dim=getattr(out_dim,'conv1').in_channels


        feature_dim = getattr(self.resnet, 'fc').in_features  #利用in_features 获得输入 然后下一步执行修改输出。
        # setattr(self.base_model, 'fc', nn.Linear(feature_dim, num_class))
        self.fc2=nn.Linear(2048,1024)

        # feature_dim=feature_dim*self.num_segments
        feature_dim = 1024 * self.num_segments
        # feature_dim=feature_dim*self.num_segments

        self.fc=nn.Linear(feature_dim,num_class)
        if self._representation == 'residual':
            self.data_bn = nn.BatchNorm2d(3)
        # self.alpha = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.Sigmoid())


        # self.alpha=nn.Sequential(nn.Linear(feature_dim,1),nn.Sigmoid())

        # self.se = nn.Sequential(
        #     nn.Conv2d(feature_dim,feature_dim//16,kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(feature_dim//16,feature_dim,kernel_size=1),
        #     nn.Sigmoid())

    def _prepare_tsn2(self, num_class):

        # feature_dim = getattr(self.base_model, 'fc').in_features  #利用in_features 获得输入 然后下一步执行修改输出。
        # setattr(self.base_model, 'fc', nn.Linear(feature_dim, num_class))
        # out_dim=getattr(self.resnet,'layer3')[0]
        # out_dim=getattr(out_dim,'conv1').in_channels
        self.fc = nn.Sequential(*list(self.vgg16_bn.children())[-1])
        self.fc[6]=nn.Linear(4096, num_class)

        # feature_dim = getattr(self.fc, 'fc').in_features  # 利用in_features 获得输入 然后下一步执行修改输出。
        # setattr(self.base_model, 'fc', nn.Linear(feature_dim, num_class))

        # feature_dim=feature_dim*self.num_segments
        #
        # self.fc2 = nn.Linear(feature_dim, num_class)
        # self.alpha = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.Sigmoid())

        # self.alpha=nn.Sequential(nn.Linear(feature_dim,1),nn.Sigmoid())

        # self.se = nn.Sequential(
        #     nn.Conv2d(feature_dim,feature_dim//16,kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(feature_dim//16,feature_dim,kernel_size=1),
        #     nn.Sigmoid())

        if self._representation == 'mv':
            setattr(self.base_model, '0',
                    nn.Conv2d(2, 64,
                              kernel_size=(7, 7),
                              stride=(2, 2),
                              padding=(3, 3),
                              bias=False))
            self.data_bn = nn.BatchNorm2d(2)
        if self._representation == 'residual':
            self.data_bn = nn.BatchNorm2d(3)


    def _prepare_base_model(self, base_model):

        if 'resnet' in base_model:
            # self.base_model = getattr(torchvision.models, base_model)(pretrained=True)

            self.resnet = getattr(torchvision.models, base_model)(pretrained=True)
            self.base_model=nn.Sequential(*list(self.resnet.children())[:-2])
            self.av = nn.Sequential(*list(self.resnet.children())[-2:-1])

            self.mobilenet= Facenet(backbone=self.backbone, num_classes=self.num_classes, pretrained=False)

            model_dict = self.mobilenet.state_dict()
            pretrained_dict = torch.load(self.model_path)
            load_key, no_load_key, temp_dict = [], [], {}
            for k, v in pretrained_dict.items():
                if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                    temp_dict[k] = v
                    load_key.append(k)
                else:
                    no_load_key.append(k)
            model_dict.update(temp_dict)
            self.mobilenet.load_state_dict(model_dict)

            self.facenet=nn.Sequential(*list(self.mobilenet.children())[:-4])

            # 冻结network1的全部参数和network2的部分参数
            # self.av = nn.Sequential(*list(self.resnet.children())[8:9])

            # in_put = getattr(self.resnet, 'layer3')[0].conv1.in_channels
            # self.out_channels=getattr(self.resnet,'layer3')[0]

            # self.conv_out=nn.Sequential(*list(getattr(self.resnet,'layer3')[0].children())[:3])
            # self.conv2_out = nn.Sequential(*list(getattr(self.resnet, 'layer4')[0].children())[:3])
            # self.av=nn.Sequential(*list(self.resnet.children())[8:9])

            # self.lstm_input = getattr(self.resnet, 'layer3')[0].conv1.out_channels
            # self.fc=nn.Linear(self.input_size,self.input_size)

            # self.lstm=nn.LSTM(input_size=self.input_size,hidden_size=self.hidden_size,num_layers=self.num_layers,batch_first=True)
            # self.lstm.flatten_parameters()
            # self.lstm_input = getattr(self.resnet, 'layer3')[0].conv1.out_channels

            #
            # in_put2=getattr(self.resnet, 'layer4')[1].conv2.out_channels

            # self.base_model2 = nn.Sequential(*list(self.resnet.children())[7:8])


            #
            # self.se = nn.Sequential(
            #     nn.AdaptiveAvgPool2d((1, 1)),
            #     nn.Conv2d(in_put,in_put//16,kernel_size=1),
            #     nn.ReLU(),
            #     nn.Conv2d(in_put//16,in_put,kernel_size=1),
            #     nn.Sigmoid())

            # self.se2 = nn.Sequential(
            #     nn.AdaptiveAvgPool2d((1, 1)),
            #     nn.Conv2d(in_put2,in_put2//16,kernel_size=1),
            #     nn.ReLU(),
            #     nn.Conv2d(in_put2//16,in_put2,kernel_size=1),
            #     nn.Sigmoid())

            # self.space = nn.Sequential(
            #     nn.Conv2d(2,1,kernel_size=7,padding=3),
            #     nn.Sigmoid())
            #


            self._input_size = 224
        elif 'vgg' in base_model:
            # self.base_model = getattr(torchvision.models, base_model)(pretrained=True)

            self.vgg16_bn = getattr(torchvision.models, base_model)(pretrained=True)
            # self.vgg16_bn = Vgg_face()
            # self.vgg16_bn.load_state_dict(torch.load("/data/jlzhang/Py_protect/2022-08-05/vgg.pth"))
            self.base_model = nn.Sequential(*list(self.vgg16_bn.children())[:-1])

            self.mobilenet= Facenet(backbone=self.backbone, num_classes=self.num_classes, pretrained=False)

            model_dict = self.mobilenet.state_dict()
            pretrained_dict = torch.load(self.model_path)
            load_key, no_load_key, temp_dict = [], [], {}
            for k, v in pretrained_dict.items():
                if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                    temp_dict[k] = v
                    load_key.append(k)
                else:
                    no_load_key.append(k)
            model_dict.update(temp_dict)
            self.mobilenet.load_state_dict(model_dict)

            self.facenet=nn.Sequential(*list(self.mobilenet.children())[:-4])

            self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                                batch_first=True)

            # self.base_model = nn.Sequential(*list(self.vgg16_bn.children())[:-7])
            # self.bn = nn.BatchNorm1d(25088)
            # self.av = nn.AdaptiveAvgPool2d(7,7)

            # in_put = getattr(self.resnet, 'layer3')[0].conv1.in_channels
            # self.out_channels=getattr(self.resnet,'layer3')[0]

            # self.conv_out=nn.Sequential(*list(getattr(self.resnet,'layer3')[0].children())[:3])
            # self.conv2_out = nn.Sequential(*list(getattr(self.resnet, 'layer4')[0].children())[:3])
            # self.av=nn.Sequential(*list(self.resnet.children())[8:9])

            # self.lstm_input = getattr(self.resnet, 'layer3')[0].conv1.out_channels
            # self.fc=nn.Linear(self.input_size,self.input_size)

            # self.lstm.flatten_parameters()
            # self.lstm_input = getattr(self.resnet, 'layer3')[0].conv1.out_channels

            #
            # in_put2=getattr(self.resnet, 'layer4')[1].conv2.out_channels

            # self.base_model2 = nn.Sequential(*list(self.resnet.children())[7:8])

            #
            # self.se = nn.Sequential(
            #     nn.AdaptiveAvgPool2d((1, 1)),
            #     nn.Conv2d(in_put,in_put//16,kernel_size=1),
            #     nn.ReLU(),
            #     nn.Conv2d(in_put//16,in_put,kernel_size=1),
            #     nn.Sigmoid())

            # self.se2 = nn.Sequential(
            #     nn.AdaptiveAvgPool2d((1, 1)),
            #     nn.Conv2d(in_put2,in_put2//16,kernel_size=1),
            #     nn.ReLU(),
            #     nn.Conv2d(in_put2//16,in_put2,kernel_size=1),
            #     nn.Sigmoid())

            # self.space = nn.Sequential(
            #     nn.Conv2d(2,1,kernel_size=7,padding=3),
            #     nn.Sigmoid())
            #

            self._input_size = 224
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))
    def forward(self, input,den_input):
        input = input.view((-1, ) + input.size()[-3:])
        den_input = den_input.view((-1,) + den_input.size()[-3:])
        if self._representation in ['mv', 'residual']:
            input = self.data_bn(input)

        input= self.base_model(input)

        feature=input
        input=self.av(input)


        input=input.flatten(1)
        input=self.fc2(input)

        den_input = self.facenet(den_input)
        den_input = den_input.flatten(1)

        # input=self.av(input)
        # input=self.base_model[1](input)
        # fearture = input

        ###添加通道注意力

        # input=input.reshape((-1,self.num_segments)+input.size()[1:])
        #
        # input=input.reshape((-1,input.size()[1]*input.size()[2])+input.size()[3:])
        #
        # avg_out = torch.mean(input, dim=1, keepdim=True)
        # max_out, _ = torch.max(input, dim=1, keepdim=True)
        # x= torch.cat([avg_out, max_out], dim=1)
        # x=self.space(x)
        # input=input*x
        ###添加通道注意力

        # input=self.av(input)
        #
        # input=self.conv_out(input)
        #
        # input=self.conv2_out(input)
        #
        # input=self.av(input)

        # input = input.view(input.size(0), -1)

        # input=self.fc(input)
        # # self.lstm.flatten_parameters()


        # # ##VGG16_bn:add
        # input=input.view(-1,self.num_segments,input.size()[1])
        # input=torch.sum(input, dim=1)
        #
        # input, _ = self.lstm(input)

        #VGG16_bn:cat
        input=input.view(-1,self.num_segments,input.size()[1])

        den_input2=den_input[:,None,:]
        den_input2=den_input2.repeat(1,self.num_segments,1,)
        input2=abs(input-den_input2)

        input=input+input2

        # input,_=self.lstm(input)



        # den_input=den_input[:,None,:]
        # den_input=den_input.repeat(1,self.num_segments,1,)
        # input2=abs(input-den_input)
        #
        # input=input+input2



        ###abs
        # den_input2=den_input[:,None,:]
        # den_input2=den_input2.repeat(1,self.num_segments,1,)
        # input2=abs(input-den_input2)
        #
        # input=input+input2
        ###abs

        input = input.reshape(-1, input.size()[2] * self.num_segments)

        #####互信息####
        # input = input.reshape(-1, input.size()[2]*self.num_segments)
        #
        #
        # MI_1=self._MMI(input,den_input)
        #
        # mv_input2=den_input[1:]
        # mv_input3=den_input[0:1]
        # mv_input4=torch.cat([mv_input2, mv_input3], dim=0)
        #
        # MI_2 = self._MMI(input, mv_input4)

        # input=input.reshape(input.size()[0], self.num_segments,-1)
        # input = input[:,7,:]




        # input=torch.sum(input, dim=1)

        # input = self.bn(input)


        # input,_=self.lstm(input)

        # input=input.reshape(-1,input.size()[2])

        # input=input.reshape(input.size(0),-1)

        # input2=self.se(input)

        # input2=self.alpha(input)
        # input=input2*input

        # input = nn.Dropout(0.5)(input)
        # input=input.contiguous().view(-1,input.size()[2])
        # attention_weight = self.alpha(input)
        # input=input
        base_out=self.fc(input)
        # base_out = self.fc(input)

        # return base_out,fearture
        return base_out,feature,2

    # def forward(self, input):
    #     input = input.view((-1, ) + input.size()[-3:])
    #     if self._representation in ['mv', 'residual']:
    #         input = self.data_bn(input)
    #     input= self.base_model(input)
    #     fearture = input
    #
    #     ###添加通道注意力
    #
    #     # input=input.reshape((-1,self.num_segments)+input.size()[1:])
    #     #
    #     # input=input.reshape((-1,input.size()[1]*input.size()[2])+input.size()[3:])
    #     #
    #     # avg_out = torch.mean(input, dim=1, keepdim=True)
    #     # max_out, _ = torch.max(input, dim=1, keepdim=True)
    #     # x= torch.cat([avg_out, max_out], dim=1)
    #     # x=self.space(x)
    #     # input=input*x
    #     ###添加通道注意力
    #
    #     input=self.av(input)
    #     #
    #     # input=self.conv_out(input)
    #     #
    #     # input=self.conv2_out(input)
    #     #
    #     # input=self.av(input)
    #
    #     input = input.view(input.size(0), -1)
    #
    #     # input=self.fc(input)
    #     # # self.lstm.flatten_parameters()
    #
    #     input=input.view(-1,self.num_segments,input.size()[1])
    #
    #
    #     input,_=self.lstm(input)
    #
    #     input=input.reshape(-1,input.size()[2])
    #
    #     # input=input.reshape(input.size(0),-1)
    #
    #     # input2=self.se(input)
    #
    #     # input2=self.alpha(input)
    #     # input=input2*input
    #
    #     # input = nn.Dropout(0.5)(input)
    #     # input=input.contiguous().view(-1,input.size()[2])
    #     # attention_weight = self.alpha(input)
    #     # input=input
    #     base_out=self.fc2(input)
    #     # base_out = self.fc(input)
    #
    #     # return base_out,fearture
    #     return base_out


    @property
    def crop_size(self):
        return self._input_size

    @property
    def scale_size(self):
        return self._input_size * 256 // 224

    def get_augmentation(self):
        if self._representation in ['mv', 'residual']:
            scales = [1, .875, .75]
            # scales = [1.05, .95, .90]
        else:
            scales = [1, .875, .75, .66]
            # scales = [1.10, 1.05, .95, .90]

        print('Augmentation scales:', scales)
        # return torchvision.transforms.Compose(
        #     [GroupMultiScaleCrop(self._input_size, scales),
        #      GroupRandomHorizontalFlip(is_mv=(self._representation == 'mv'))])
        return torchvision.transforms.Compose(
            [GroupScale2(is_mv=(self._representation == 'mv')),
            GroupMultiScaleCrop(self._input_size, scales),
             # GroupCenterCrop(self._input_size),
             GroupRandomHorizontalFlip(is_mv=(self._representation == 'mv')),
             GroupRation(is_mv=(self._representation == 'mv')),
             GroupTranslate(is_mv=(self._representation == 'mv'))])
