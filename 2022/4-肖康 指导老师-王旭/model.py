import math
import torch.nn.functional as F
from torch import nn
import torch


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.vggnet = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.MaxPool2d(2, 2)
        )


        self.net = nn.Sequential(
            nn.Conv2d(256 * 4, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.MaxPool2d(2, 2),
        )

        self.linear1=nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
        )
        self.linear2=nn.Sequential(
            nn.Linear(256, 49),
        )
        self.linear3=nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
        )
        self.linear4=nn.Sequential(
            nn.Linear(256*2, 256),
            nn.LeakyReLU(0.2),
        )
        self.linear5=nn.Sequential(
            nn.Linear(256, 1),
        )


    def forward(self, x1,x2,x3,x4,y1,y2,y3,y4):
        batch_size = x1.size(0)
        ##x1
        featureA=self.vggnet(x1)
        featureB=self.vggnet(x2)
        featureC=self.vggnet(x3)
        featureD=self.vggnet(x4)

        ##汇聚
        print('featureD', featureD.shape)
        featuremap1=torch.cat((featureA,featureB,featureC,featureD),1)
        print('featuremap11',featuremap1.shape)
        featuremap1=self.net(featuremap1).squeeze(-1).squeeze(-1)
        print('pfeaturemap1',featuremap1.shape)
        feature1_type1=self.linear1(featuremap1)
        feature1_type2=self.linear2(feature1_type1)

        feature1_content1=self.linear3(featuremap1)
        feature1_content2=torch.cat((feature1_type1, feature1_content1), 1)
        feature1_content2=self.linear4(feature1_content2)
        feature1_content2=self.linear5(feature1_content2)

        ##y
        x1=y1
        x2=y2
        x3=y3
        x4=y4
        ##x1
        featureA=self.vggnet(x1)
        featureB=self.vggnet(x2)
        featureC=self.vggnet(x3)
        featureD=self.vggnet(x4)

        ##汇聚
        featuremap2=torch.cat((featureA,featureB,featureC,featureD),1)
        featuremap2=self.net(featuremap2).squeeze(-1).squeeze(-1)
        print('featuremap2',featuremap2.shape)
        feature2_type1=self.linear1(featuremap2)
        feature2_type2=self.linear2(feature2_type1)

        feature2_content1=self.linear3(featuremap2)
        feature2_content2=torch.cat((feature2_type1, feature2_content1), 1)
        feature2_content2=self.linear4(feature2_content2)
        feature2_content2=self.linear5(feature2_content2)

        return F.leaky_relu(feature1_type2).view(batch_size, -1),F.leaky_relu(feature1_content2).view(batch_size, -1),F.leaky_relu(feature2_content2).view(batch_size, -1)




