import torch.nn as nn
import torch
# from src.neural_networks.encoder import BaseEncoder



class MI_network(nn.Module):
    def __init__(self,):
        """Local statistique nerwork

        Args:
            img_feature_channels (int): [Number of input channels]
        """

        super().__init__()
        # self.MMI_1=nn.Sequential(
        #     nn.Linear(25088,4096),
        #     nn.ReLU()
        # )
        #
        # self.MMI_2=nn.Sequential(
        #     nn.Linear(8192,4096),
        #     nn.ReLU()
        # )
        # self.MMI_3=nn.Linear(512,512)
        self.MMI_1=nn.Sequential(
            nn.Linear(32768,1024),
            nn.ReLU()
        )
        self.MMI_1_1=nn.Sequential(
            nn.Linear(1024,1024),
            nn.ReLU()
        )

        self.MMI_2=nn.Sequential(
            nn.Linear(2048,1024),
            nn.ReLU()
        )
        self.MMI_3=nn.Linear(1024,1)

    def forward(self, concat_feature: torch.Tensor,representation: torch.Tensor) -> torch.Tensor:
        # x = torch.cat([concat_feature, representation], dim=1)
        x = self.MMI_1(concat_feature)
        y = self.MMI_1_1(representation)
        # y=self.MMI_1(representation)
        x = torch.cat([x, y], dim=1)
        # x=x+y
        x=self.MMI_2(x)
        glocal_statistics = self.MMI_3(x)
        return glocal_statistics