import torch

from model.layers import *
import math


def init_weights(layer):
    """Init weights for layers w.r.t. the original paper."""
    layer_name = layer.__class__.__name__
    if layer_name.find("Conv") != -1:
        layer.weight.data.normal_(0.0, 0.02)
    elif layer_name.find("BatchNorm") != -1:
        layer.weight.data.normal_(1.0, 0.02)
    elif type(layer) == nn.Linear:
        layer.weight.data.normal_(0.0, 1e-4)


class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(config.input_dim, 64, kernel_size=(5, 1)),
            nn.MaxPool2d((2, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(5, 1)),
            nn.MaxPool2d((2, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 1)),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(config.output_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, config.cls_num),
        )

    def forward(self, x):
        x = x.unsqueeze(dim=3)  # input size: (batch_size, channel, win， 1)
        x = self.cnn(x)
        x = x.reshape(x.size(0), -1)
        feature = x
        x = self.classifier(x)
        return feature, x
        # return x


class ConvLSTMv1(nn.Module):
    def __init__(self, config):
        super(ConvLSTMv1, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(config.input_dim, 64, kernel_size=(5, 1)),
            nn.Conv2d(64, 64, kernel_size=(5, 1)),
            nn.Conv2d(64, 64, kernel_size=(5, 1)),
            nn.Conv2d(64, 64, kernel_size=(5, 1)),
        )

        self.lstm = nn.LSTM(64, 128, 2, batch_first=True)
        self.dropout = nn.Dropout()
        self.classifier = nn.Linear(128, config.cls_num)

    def forward(self, x):
        x = x.unsqueeze(dim=3)  # input size: (batch_size, channel, win， 1)
        x = self.features(x)  # [b, 64 , h , w]
        x = x.view(x.shape[0], -1, 64)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = x[:, -1, :]
        x = x.view(x.shape[0], 128)
        x = self.classifier(x)

        return x


class ConvLSTMv2(nn.Module):
    def __init__(self, config):
        super(ConvLSTMv2, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(config.input_dim, 64, kernel_size=(5, 1)),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(64, 64, kernel_size=(5, 1)),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(64, 32, kernel_size=(3, 1)),
        )

        self.lstm = nn.LSTM(32, 128, 1, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(128, config.cls_num)

    def forward(self, x):
        x = x.unsqueeze(dim=3)  # input size: (batch_size, channel, win， 1)
        x = self.features(x)  # [b, 64 , h , w]
        x = x.view(x.shape[0], -1, 32)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = x[:, -1, :]
        x = x.view(x.shape[0], 128)
        x = self.classifier(x)

        return x


class Feature_CNN(nn.Module):
    def __init__(self, config):
        super(Feature_CNN, self).__init__()
        self.config = config
        if config.attention_block == 'ECAnet':
            self.attention_model = ECAnet(64)
        elif config.attention_block == 'SEnet':
            self.attention_model = SEnet(64)
        elif config.attention_block == 'CBAMblock':
            self.attention_model = CBAMblock(64)

        self.cnn = nn.Sequential(
            nn.Conv2d(config.input_dim, 64, kernel_size=(5, 1)),
            nn.MaxPool2d((2, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(5, 1)),
            nn.MaxPool2d((2, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 1)),
            nn.ReLU(),
        )
        # for name, param in self.cnn.named_parameters():
        #     if 'weight' in name:
        #         torch.nn.init.xavier_normal(param)

    def forward(self, x):
        x = x.unsqueeze(dim=3)  # input size: (batch_size, channel, win， 1)
        # print(x.size())
        x = self.cnn(x)
        # print(x.size())
        if self.config.attention_block != 'None':
            x = self.attention_model(x)
        x = x.reshape(x.size(0), -1)
        # print(x.size())
        return x


class Feature_ConvLSTMv2(nn.Module):
    def __init__(self, config):
        super(Feature_ConvLSTMv2, self).__init__()

        self.config = config
        if config.attention_block == 'ECAnet':
            self.attention_model = ECAnet(128)
        elif config.attention_block == 'SEnet':
            self.attention_model = SEnet(128)
        elif config.attention_block == 'CBAMblock':
            self.attention_model = CBAMblock(128)

        self.features = nn.Sequential(
            nn.Conv2d(config.input_dim, 64, kernel_size=(5, 1)),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(64, 64, kernel_size=(5, 1)),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(64, 32, kernel_size=(3, 1)),
        )

        self.lstm = nn.LSTM(32, 128, 1, batch_first=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.unsqueeze(dim=3)  # input size: (batch_size, channel, win， 1)
        x = self.features(x)  # [b, 64 , h , w]
        x = x.view(x.shape[0], -1, 32)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = x[:, -1, :]
        x = x.view(x.shape[0], 128)
        # print(x.size())

        return x


class Feature_disentangle(nn.Module):
    def __init__(self, config):
        super(Feature_disentangle, self).__init__()

        self.config = config
        if config.disentangler_attention_block == 'ECAnet':
            self.attention_block = D_ECAnet(int(config.output_dim / 4))
        elif config.disentangler_attention_block == 'SEnet':
            self.attention_block = D_SEnet(int(config.output_dim / 4))
        elif config.disentangler_attention_block == 'CBAMblock':
            self.attention_block = CBAMblock(int(config.output_dim / 4))

        self.fc1 = nn.Linear(config.output_dim, int(config.output_dim / 4))
        self.bn1_fc = nn.BatchNorm1d(int(config.output_dim / 4))

    def forward(self, x):
        # print(x.size())
        x = self.fc1(x)
        # print(x.size())
        x = F.relu(self.bn1_fc(x))
        if self.config.disentangler_attention_block != 'None':
            x = self.attention_block(x)
        # print(x.size())

        return x


class Feature_discriminator(nn.Module):
    def __init__(self, config):
        super(Feature_discriminator, self).__init__()
        self.fc1 = nn.Linear(int(config.output_dim / 4), 2)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        return x


class Reconstructor_Net(nn.Module):
    def __init__(self, config):
        super(Reconstructor_Net, self).__init__()
        self.fc = nn.Linear(int(config.output_dim / 2), config.output_dim)

    def forward(self, x):
        x = self.fc(x)
        return x


class Mine_Net(nn.Module):
    def __init__(self, config):
        super(Mine_Net, self).__init__()
        self.fc1_x = nn.Linear(int(config.output_dim / 4), int(config.output_dim / 8))
        self.fc1_y = nn.Linear(int(config.output_dim / 4), int(config.output_dim / 8))
        self.fc2 = nn.Linear(int(config.output_dim / 8), 1)

    def forward(self, x, y):
        h1 = F.leaky_relu(self.fc1_x(x) + self.fc1_y(y))
        h2 = self.fc2(h1)
        return h2


class Predictor_Net(nn.Module):
    def __init__(self, config):
        super(Predictor_Net, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(int(config.output_dim / 4), config.cls_num)
        )

    def forward(self, x):
        x = self.classifier(x)
        return x


class ECAnet(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(ECAnet,self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        padding = kernel_size // 2

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1,1,kernel_size=kernel_size, padding= padding,bias=False)
        self.sigmiod = nn.Sigmoid()

    def forward(self,x):
        b,c,h,w = x.size()
        avg = self.avg_pool(x).view(b,1,c)
        out = self.conv1d(avg)
        out = self.sigmiod(out)
        out = out.view([b,c,1,1])

        return out * x

class D_ECAnet(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(D_ECAnet, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        padding = kernel_size // 2

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmiod = nn.Sigmoid()

    def forward(self,x):
        b,c = x.size()
        x_1 = x.unsqueeze(1)
        out = self.conv1d(x_1)
        out = out.squeeze()

        return out * x

class SEnet(nn.Module):
    def __init__(self,channel, ratio=5):
        super(SEnet,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        b,c,h,w = x.size()
        avg = self.avg_pool(x).view(b,c)
        out = self.fc(avg).view(b,c,1,1)

        return out * x

class D_SEnet(nn.Module):
    def __init__(self, channel, ratio = 5):
        super(D_SEnet, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.size()
        out = self.fc(x)

        return out * x

class ChannelAttention(nn.Module):
    def __init__(self,channel,ratio=16):
        super(ChannelAttention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(channel // ratio, channel, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        b,c,h,w = x.size()
        max_pool_out = self.max_pool(x).view([b,c])
        avg_pool_out = self.avg_pool(x).view([b,c])

        max_pool_out = self.fc(max_pool_out)
        avg_pool_out = self.fc(avg_pool_out)

        out = max_pool_out + avg_pool_out
        out = self.sigmoid(out).view([b,c,1,1])

        return out * x

class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=(5, 1)):
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(2,1,kernel_size,bias=False,padding=(2,0))
        # self.conv = nn.Conv2d(2,1,kernel_size,bias=False,)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):

        avg_out = torch.mean(x,dim=1,keepdim=True)
        max_out, _ = torch.max(x,dim=1,keepdim=True)
        out = torch.cat([avg_out, max_out],dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)

        return out * x

class CBAMblock(nn.Module):
    def __init__(self,channel,ratio=16,kernel_size=(5, 1)):
        super(CBAMblock,self).__init__()
        self.channelblock = ChannelAttention(channel,ratio=ratio)
        self.spatialblock = SpatialAttention(kernel_size=kernel_size)

    def forward(self,x):
        x = self.channelblock(x)
        x = self.spatialblock(x)
        return x
