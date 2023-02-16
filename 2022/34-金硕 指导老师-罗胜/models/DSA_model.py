import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

feature_flatten_dim = 13824
input_channel_dim = 9

class MAML_BN(nn.Module):

    def __init__(self, opt):

        super(MAML_BN, self).__init__()
        self.opt = opt



        self.vars = nn.ParameterList()
        self.vars_bn = nn.ParameterList()

        conv1 = nn.Conv1d(input_channel_dim, 32, kernel_size=3)
        bn1 = nn.BatchNorm1d(32)
        conv2 = nn.Conv1d(32, 64, kernel_size=3)
        bn2 = nn.BatchNorm1d(64)
        conv3 = nn.Conv1d(64, 128, kernel_size=3)
        bn3 = nn.BatchNorm1d(128)
        conv4 = nn.Conv1d(128, 256, kernel_size=3)
        bn4 = nn.BatchNorm1d(256)
        conv5 = nn.Conv1d(256, 512, kernel_size=3)
        bn5 = nn.BatchNorm1d(512)
        fc1 = nn.Linear(feature_flatten_dim, 1024)
        bn6 = nn.BatchNorm1d(1024)
        fc2 = nn.Linear(1024, 256)
        bn7 = nn.BatchNorm1d(256)
        fc3 = nn.Linear(256, self.opt['num_class'])

        layers = []

        layers.append(('conv', conv1))
        layers.append(('bn', bn1))
        layers.append(('conv', conv2))
        layers.append(('bn',bn2))
        layers.append(('conv', conv3))
        layers.append(('bn',bn3))
        layers.append(('conv', conv4))
        layers.append(('bn',bn4))
        layers.append(('conv', conv5))
        layers.append(('bn',bn5))
        layers.append(('fc', fc1))
        layers.append(('bn',bn6))
        layers.append(('fc', fc2))
        layers.append(('bn',bn7))
        layers.append(('fc', fc3))

        for (name, layer) in layers:
            if name == 'bn':
                w = nn.Parameter(torch.ones_like(layer.weight))
                self.vars.append(w)
                b = nn.Parameter(torch.zeros_like(layer.bias))
                self.vars.append(b)

                running_mean = nn.Parameter(torch.zeros_like(layer.weight), requires_grad=False)
                running_var = nn.Parameter(torch.ones_like(layer.weight), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])
            else:
                w = nn.Parameter(torch.ones_like(layer.weight))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                b = nn.Parameter(torch.zeros_like(layer.bias))
                self.vars.append(b)

    def forward_get_feature(self, input, vars =None, training=False):

        batch_size = input.size(0)
        if vars is None:
            vars = self.vars
        idx = 0
        bn_idx = 0


        w, b = vars[idx], vars[idx+1]
        input = F.conv1d(input, w, b)
        idx += 2
        input = F.relu(input, True)


        w, b = vars[idx], vars[idx + 1]
        running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
        input = F.batch_norm(input, running_mean, running_var, weight=w, bias=b, training=training)
        idx += 2
        bn_idx += 2


        w, b = vars[idx], vars[idx + 1]
        input = F.conv1d(input, w, b)
        idx += 2
        input = F.relu(input, True)


        w, b = vars[idx], vars[idx + 1]
        running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
        input = F.batch_norm(input, running_mean, running_var, weight=w, bias=b, training=training)
        idx += 2
        bn_idx += 2

        w, b = vars[idx], vars[idx + 1]
        input = F.conv1d(input, w, b)
        idx += 2
        input = F.max_pool1d(input, 2)
        input = F.relu(input, True)


        w, b = vars[idx], vars[idx + 1]
        running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
        input = F.batch_norm(input, running_mean, running_var, weight=w, bias=b, training=training)
        idx += 2
        bn_idx += 2

        w, b = vars[idx], vars[idx + 1]
        input = F.conv1d(input, w, b)
        idx += 2
        input = F.relu(input, True)


        w, b = vars[idx], vars[idx + 1]
        running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
        input = F.batch_norm(input, running_mean, running_var, weight=w, bias=b, training=training)
        idx += 2
        bn_idx += 2

        w, b = vars[idx], vars[idx + 1]
        input = F.conv1d(input, w, b)
        idx += 2
        input = F.max_pool1d(input, 2)
        input = F.relu(input, True)


        w, b = vars[idx], vars[idx + 1]
        running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
        input = F.batch_norm(input, running_mean, running_var, weight=w, bias=b, training=training)
        idx += 2
        bn_idx += 2



        return input.view(batch_size, -1)

    def forward(self, input, vars =None, training=True):

        if vars is None:
            vars = self.vars
        idx = 0
        bn_idx = 0


        w, b = vars[idx], vars[idx+1]
        input = F.conv1d(input, w, b)
        idx += 2
        input = F.relu(input, True)

        w, b = vars[idx], vars[idx + 1]
        running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
        input = F.batch_norm(input, running_mean, running_var, weight=w, bias=b, training=training)
        idx += 2
        bn_idx += 2


        w, b = vars[idx], vars[idx + 1]
        input = F.conv1d(input, w, b)
        idx += 2
        input = F.relu(input, True)

        w, b = vars[idx], vars[idx + 1]
        running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
        input = F.batch_norm(input, running_mean, running_var, weight=w, bias=b, training=training)
        idx += 2
        bn_idx += 2

        w, b = vars[idx], vars[idx + 1]
        input = F.conv1d(input, w, b)
        idx += 2
        input = F.max_pool1d(input, 2)
        input = F.relu(input, True)

        w, b = vars[idx], vars[idx + 1]
        running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
        input = F.batch_norm(input, running_mean, running_var, weight=w, bias=b, training=training)
        idx += 2
        bn_idx += 2

        w, b = vars[idx], vars[idx + 1]
        input = F.conv1d(input, w, b)
        idx += 2
        input = F.relu(input, True)

        w, b = vars[idx], vars[idx + 1]
        running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
        input = F.batch_norm(input, running_mean, running_var, weight=w, bias=b, training=training)
        idx += 2
        bn_idx += 2

        w, b = vars[idx], vars[idx + 1]
        input = F.conv1d(input, w, b)
        idx += 2
        input = F.max_pool1d(input, 2)
        input = F.relu(input, True)


        w, b = vars[idx], vars[idx + 1]
        running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
        input = F.batch_norm(input, running_mean, running_var, weight=w, bias=b, training=training)
        idx += 2
        bn_idx += 2



        w, b = vars[idx], vars[idx + 1]
        input = F.linear(input.view(-1, feature_flatten_dim), w, b)
        idx += 2
        input = F.relu(input, True)


        w, b = vars[idx], vars[idx + 1]
        running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
        input = F.batch_norm(input, running_mean, running_var, weight=w, bias=b, training=training)
        idx += 2
        bn_idx += 2

        w, b = vars[idx], vars[idx + 1]
        input = F.linear(input, w, b)
        idx += 2
        input = F.relu(input, True)


        w, b = vars[idx], vars[idx + 1]
        running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
        input = F.batch_norm(input, running_mean, running_var, weight=w, bias=b, training=training)
        idx += 2
        bn_idx += 2

        w, b = vars[idx], vars[idx + 1]
        input = F.linear(input, w, b)
        idx += 2

        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)

        return F.log_softmax(input, 1)

    def zero_grad(self, vars=None):
        """

        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars


class Extractor(nn.Module):

    def __init__(self, opt):
        super(Extractor, self).__init__()
        self.opt = opt

        self.feature = nn.Sequential(
            nn.Conv1d(input_channel_dim, 32, kernel_size=3),
            nn.ReLU(True),
            nn.BatchNorm1d(32),


            nn.Conv1d(32, 64, kernel_size=3),
            nn.ReLU(True),
            nn.BatchNorm1d(64),


            nn.Conv1d(64, 128, kernel_size=3),
            nn.MaxPool1d(2),
            nn.BatchNorm1d(128),
            nn.ReLU(True),


            nn.Conv1d(128, 256, kernel_size=3),
            nn.ReLU(True),
            nn.BatchNorm1d(256),


            nn.Conv1d(256, 512, kernel_size=3),
            nn.MaxPool1d(2),
            nn.ReLU(True),
            nn.BatchNorm1d(512),
        )

    def forward(self, input):
        out = self.feature(input)

        out = out.view(input.size(0), -1)
        return out


class Class_Classifier(nn.Module):

    def __init__(self, opt):
        super(Class_Classifier, self).__init__()
        self.opt = opt

        self.class_classifier = nn.Sequential(
            nn.Linear(feature_flatten_dim, 1024),
            nn.ReLU(True),
            nn.BatchNorm1d(1024),


            nn.Linear(1024, 256),
            nn.ReLU(True),
            nn.BatchNorm1d(256),


            nn.Linear(256, self.opt['num_class']))

    def forward(self, input):
        out = self.class_classifier(input)

        return F.log_softmax(out, 1)


if __name__ == '__main__':
    pass