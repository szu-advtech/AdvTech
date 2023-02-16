import numpy as np
import torch
import torch.nn as nn
from .HRblock import HighResolutionBlock

BN_MOMENTUM = 0.01
def conv3x3(in_channle, out_channel, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channle, out_channel, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS # [64, 128, 256]
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        self.deblocks = nn.ModuleList()

        self.transition_layer1 = nn.Sequential(
            nn.Conv2d(
                input_channels, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(
                64, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )

        self.transition_layer = nn.Sequential(
            nn.Conv2d(
                64, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(
                128, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.stage = HighResolutionBlock([64, 128], [64, 128])
        for idx in range(num_levels):
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))


        self.num_bev_features = c_in


    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        x = spatial_features
        x = self.transition_layer1(x)
        x1 = self.transition_layer(x)
        x_list = []
        x_list.append(x)
        x_list.append(x1)
        hr_block = self.stage(x_list)

        # for i in range(len(self.blocks)):
        #     x = self.blocks[i](x)
        #
        #     stride = int(spatial_features.shape[2] / x.shape[2])
        #     ret_dict['spatial_features_%dx' % stride] = x
        #     if len(self.deblocks) > 0:
        #         ups.append(self.deblocks[i](x))
        #     else:
        #         ups.append(x)

        for i in range(len(self.deblocks)):
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](hr_block[i]))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        # if len(self.deblocks) > len(self.blocks):
        #     x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x

        return data_dict
