import torch.nn as nn
import torch

class CACModel(nn.Module):
    def __init__(self, embedding, classifier, num_classes):
        super(CACModel, self).__init__()

        self.embedding = embedding
        self.classifier = classifier
        self.num_classes = num_classes

        self.anchors = nn.Parameter(torch.zeros(self.num_classes, self.num_classes).double(), requires_grad=False)

    def forward(self, inputs):
        outputs = self.embedding(inputs)
        outputs = self.classifier(outputs)

        return outputs

    def forward1(self, inputs):
        outputs = self.embedding(inputs)
        outputs = self.classifier(outputs)
        outputs = self.distance_classifier(outputs)

        return outputs

    def distance_classifier(self, x):
        n = x.size(0)
        m = self.num_classes
        d = self.num_classes

        x = x.unsqueeze(1).expand(n, m, d).double()
        anchors = self.anchors.unsqueeze(0).expand(n, m, d)
        dists = torch.norm(x - anchors, 2, 2)

        return dists

    def set_anchors(self, means):
        self.anchors = nn.Parameter(means.double(), requires_grad=False)