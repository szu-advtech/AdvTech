# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List
#import my_point_utils as point_utils

"""
# out_channels = max(32, group_feature.shape[1]//2)
# in_channels = group_feature.shape[1]
# mlps[1] = group_feature.shape[1]+1
# mlps[0] = 32
"""
class AdaptiveSampling(nn.Module):
	def __init__(self, in_channels, out_channels, mlps: List[int]):
		super(AdaptiveSampling, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.mlp_for_feature = nn.Conv2d(in_channels + 3, out_channels*2, kernel_size=(1,1))
		self.mlp_for_xyz = nn.Conv2d(in_channels + 3, out_channels, kernel_size=(1,1))
		self.bn_for_feature = nn.BatchNorm2d(out_channels*2)
		self.bn_for_xyz = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU()

		self.mlp_group = nn.Sequential()
		# assert len(mlps) == 2
		self.mlp_group.add_module("mlp1", nn.Conv2d(out_channels, mlps[0], kernel_size=(1,1)))
		self.mlp_group.add_module("bn1", nn.BatchNorm2d(mlps[0]))
		self.mlp_group.add_module("relu1", nn.ReLU())
		self.mlp_group.add_module("mlp2", nn.Conv2d(mlps[0], mlps[1],kernel_size=(1,1)))
		self.mlp_group.add_module("bn2", nn.BatchNorm2d(mlps[1]))


	def SampleWeights(self, group_feature, group_xyz, bn=True, scaled=True):
		B, C, npoint, nsample = group_feature.shape
		normalized_xyz = group_xyz- group_xyz[:,:,:,0].unsqueeze(3).repeat(1,1,1,nsample)
		new_point = torch.cat([normalized_xyz, group_feature], dim=1) # (B, C+3, npoint, nsample)

		transformed_feature = self.relu(self.bn_for_feature(self.mlp_for_feature(new_point)))
		transformed_new_point = self.relu(self.bn_for_xyz(self.mlp_for_xyz(new_point)))
		transformed_feature1 = transformed_feature[:, :self.out_channels, :, :]
		feature = transformed_feature[:, self.out_channels:, :, :]

		transformed_new_point = transformed_new_point.permute(0,2,3,1) #(B, npoint, nsample, out_channels)
		transformed_feature1 = transformed_feature1.permute(0,2,1,3) #(B, npoint, out_channels, nsample)
		weights = torch.matmul(transformed_new_point, transformed_feature1) #(B, npoint, nsample, nsample)

		if scaled:
			weights = weights / torch.sqrt(torch.tensor([self.out_channels], dtype=torch.float)).item()
		weights = F.softmax(weights, dim=-1)

		new_group_features = torch.matmul(weights, feature.permute(0,2,3,1)) #(B, npoint, nsample, out_channels)
		new_group_features = new_group_features.permute(0,3,1,2) #(B, out_channels, npoint, nsample)

		new_group_features = self.mlp_group(new_group_features)
		new_group_weights = F.softmax(new_group_features, dim=3)
		return new_group_weights


	def forward(self, group_xyz, group_feature, num_neighbor):
		"""
		Args:
		group_xyz-> 3-dim coordinate with grouped neighbors, (B, 3, npoint, n_neighbor)
		group_feature-> C-dim features of group_xyz, (B, C, npoint, n_neighbor)
		
		"""
		shift_group_xyz = group_xyz[:,:,:,:num_neighbor]
		shift_group_feature = group_feature[:,:,:,:num_neighbor]
		sample_weight = self.SampleWeights(shift_group_feature, shift_group_xyz)
		new_weight_xyz = sample_weight[:, 0, :, :].unsqueeze(1).repeat(1,3,1,1)
		new_weight_feature = sample_weight[:, 1:, :, :]

		new_xyz = torch.sum(torch.mul(shift_group_xyz, new_weight_xyz), dim=3)
		new_feature = torch.sum(torch.mul(shift_group_feature, new_weight_feature), dim=3)
		# new_xyz = torch.mul(shift_group_xyz, new_weight_xyz)
		# new_feature = torch.mul(shift_group_feature, new_weight_feature)

		return new_xyz, new_feature

def main():
	group_xyz = torch.randn(2,3,5,3)
	group_feature = torch.randn(2,6,5,3)
	adaptive_sampling  = AdaptiveSampling(in_channels=6, out_channels=32, mlps=[32, 1+6])
	new_xyz, new_feature = adaptive_sampling(group_xyz, group_feature, num_neighbor=3)
	print(new_xyz.shape, new_feature.shape)
	print(torch.sum(torch.isnan(new_feature)), torch.sum(torch.isnan(new_xyz)))

if __name__ == '__main__':
	main()