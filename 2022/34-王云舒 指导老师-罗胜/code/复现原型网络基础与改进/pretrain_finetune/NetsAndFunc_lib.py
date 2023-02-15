import torch
import torch.nn as nn
Final_Label_num=64

# acc为准确率，该函数计算准确率
def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct / total

class Net_resnet_pretrained(nn.Module):
    #继承了nn.Module便于操作，pytorch的resnet18接口的最后一层fc层的输出维度是1000，要进行最后一层的修改
    def __init__(self, model):
        super(Net_resnet_pretrained, self).__init__()
        # 取掉model的后1层
        self.resnet_layer = nn.Sequential(*list(model.children())[:-1])
        self.Linear_layer = nn.Linear(512, Final_Label_num)  # 加上一层参数修改好的全连接层
    def forward(self, x):
        x = self.resnet_layer(x)
        #print(x.size(0))
        x = x.view(x.size(0), -1)
        x = self.Linear_layer(x)
        return x

class Net_mobilenet_pretrained(nn.Module):
    #继承了nn.Module便于操作，所有的Linear数目不同的网络都需要不同的调整与修改
    def __init__(self, model):
        super(Net_mobilenet_pretrained, self).__init__()# 取掉model的最后分类层
        self.last_second_layer=nn.Sequential(*list(model.children())[:-1])
        self.last_classfy_layer=nn.Sequential\
        (
            nn.Linear(960, 640),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(640, Final_Label_num),
        )
    def forward(self, x):
        x = self.last_second_layer(x)
        #print(x.size(0))
        x = x.view(x.size(0), -1)
        #这里只要报错就可以知道最后一层需要多少神经元，调整数目即可
        x = self.last_classfy_layer(x)
        return x

class Net_shufflenet_pretrained(nn.Module):
    #继承了nn.Module便于操作
    def __init__(self, model):
        super(Net_shufflenet_pretrained, self).__init__()
        # 取掉model的后1层
        self.last_second_layer = nn.Sequential(*list(model.children())[:-1])
        self.Linear_layer = nn.Linear(1024 * 7 * 7, Final_Label_num)  # 加上一层参数修改好的全连接层
    def forward(self, x):
        x = self.last_second_layer(x)
        #print(x.size(0))
        x = x.view(x.size(0), -1)
        # x = x.view(-1, 1024 * 7 * 7)
        #上下两句view等价 可以通过再次print x看到
        x = self.Linear_layer(x)
        return x