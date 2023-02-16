# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
	"""
	Contrastive loss function.
	Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
	"""
	def __init__(self, margin=1.0, scale=1.0):
		super(ContrastiveLoss, self).__init__()
		self.margin = margin
		self.scale = scale
		self.pair_distances = nn.PairwiseDistance(p=2, eps=1e-06, keepdim=False)

	def forward(self, x1, x2, label):
		"""
		value in label(tensor) is 1 or 0 which depends on whether x1 and x2 is the same category or not
		1 : same ; 0: different
		x1 is torch.FloatTensor
		x2 is torch.FloatTensor
		"""
		#x1_norm = x1 / torch.norm(x1, p=2, dim=1, keepdim=True)
		#x2_norm = x2 / torch.norm(x2, p=2, dim=1, keepdim=True)
		x1_norm = F.normalize(x1, p=2, dim=1)
		x2_norm = F.normalize(x2, p=2, dim=1)
		l2_dist = self.pair_distances(x1_norm, x2_norm)
		contrast_loss = label * torch.pow(l2_dist, 2) * 0.5 + \
						(1-label) * torch.pow(torch.clamp(self.margin - l2_dist, min=0.0), 2) * 0.5
		contrast_loss = torch.mean(contrast_loss)
		return contrast_loss

class ContrastiveLossWithinBatch(nn.Module):
	"""
	margin: float, used for L2 contrastive loss
	scale: factor to multiply with loss
	method: str, only support "L2" or "Cosine"
	Contrastive loss function.
	Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
	"""
	def __init__(self, margin, scale=1.0, method="L2"):
		super(ContrastiveLossWithinBatch, self).__init__()
		self.margin = margin # L2: 1.0 ; Cosine: 0.5
		self.scale = scale
		self.method = method

	def forward(self, x, label):
		"""
		label: torch.LongTensor, indicates categories of x
		x is torch.FloatTensor, a batch of embeding feature vectors, size:(B,C)
		"""
		B, C = x.shape
		x_norm = F.normalize(x, p=2, dim=1)
		index_equal = torch.eye(label.shape[0], dtype=torch.uint8).cuda() # sample itself
		binary_label = (label.view(-1,1) == label.view(-1,1).permute(1, 0))
		positive_mask = (binary_label - index_equal).float()
		negative_mask = (1 - binary_label).float()
		# print("positive_mask \n{}".format(positive_mask))
		# print("negative_mask \n{}".format(negative_mask))

		
		if self.method == "L2":
			# batch hard
			temp1 = x_norm.unsqueeze(0).expand(B, B, C) # size (B, B, C)
			temp2 = x_norm.unsqueeze(1).expand(B, B, C)
			l2_dist = torch.sum(torch.pow(temp1-temp2, 2), dim=2)

			pos_dist = torch.sum(positive_mask * l2_dist, dim=1, keepdim=True) # need to be tested
			
			max_dist = torch.max(l2_dist, dim=1, keepdim=True)[0]
			neg_dist = l2_dist + max_dist * (1.0 - negative_mask)
			hardest_neg_dist = torch.min(neg_dist, dim=1, keepdim=True)[0]
			# print("hardest_negative_dist: \n{}".format(hardest_neg_dist))

			triplet_loss = torch.clamp(pos_dist + self.margin - hardest_neg_dist, min=0.0)
			# debug
			#print(contrast_loss)
			return torch.mean(triplet_loss) * self.scale

		elif self.method == "Cosine":

			dist = torch.matmul(x_norm, x_norm.permute(1, 0))

			loss = positive_mask * (1 - dist) + negative_mask * torch.clamp(dist - self.margin, min=0)
			# only upper triangular matrix is valid loss
			loss = loss*0.5
			#print(loss)
			return torch.mean(loss) * self.scale
			
		else:
			raise ValueError("sorry, please enter L2 or Cosine")


class IntermediateLoss(nn.Module):
	"""
	Based on Cosine loss 
	"""
	def __init__(self, margin=0.1, scale=2.0):
		super(IntermediateLoss, self).__init__()
		self.margin = margin
		self.scale = scale

	def forward(self, x1, x2, label):
		"""
		value in label(tensor) is 1 or 0 which depends on whether x1 and x2 is the same category or not
		1 : same ; 0: different
		x1 is torch.FloatTensor (B, N, C)
		x2 is torch.FloatTensor (B, N, C)
		"""
		
		# cosine distance version
		B, N, C = x1.shape
		x2_t = x2.permute(0,2,1) # (B, C, N)
		x1_norm = F.normalize(x1, p=2, dim=2)
		x2_norm = F.normalize(x2_t, p=2, dim=1)
		cos_dist = torch.matmul(x1_norm, x2_norm) # (B, N, N)
		cos_dist = torch.mean(cos_dist.view(B, -1), dim=1)

		loss = label * (1-cos_dist) + (1-label) * torch.clamp(cos_dist - self.margin, min=0)
		
		loss = torch.mean(loss)
		return loss

		"""
		# L2 distance version
		B, N, C = x1.shape # x1-> N1, x2->N2, though N1==N2
		x1_norm = F.normalize(x1, p=2, dim=2)
		x2_norm = F.normalize(x2, p=2, dim=2)
		x1_norm_expand = x1_norm.unsqueeze(2).expand(B, N, N, C)
		x2_norm_expand = x2_norm.unsqueeze(1).expand(B, N, N, C)
		l2_dist = torch.sum(torch.pow(x1_norm_expand-x2_norm_expand, 2), dim=3) # (B, N, N)
		l2_dist = torch.mean(l2_dist.view(B, -1), dim=1)

		# margin must be smaller than or equal to 1.414
		loss = label * l2_dist + (1-label) * torch.clamp(self.margin - l2_dist, min=0.0)

		return loss.mean()
		"""
		
class MultiContrastiveLoss(nn.Module):
	"""
	Contrastive loss function.
	Based on: NIPS2016 'Improved Deep Metric Learning with Multi-class N-pair Loss Objective'
	"""
	def __init__(self, margin, scale=1.0):
		super(MultiContrastiveLoss, self).__init__()
		self.margin = margin
		self.scale = scale

	def forward(self, x1, x2, label1, label2):
		"""
		x1 is torch.FloatTensor
		x2 is torch.FloatTensor
		label1 is label of x1
		label2 is label of x2
		This function compute distance of every pair of embeding features
		Embeddings should not be l2 normalized.
		"""
		assert len(x1.shape) == 2
		B, C = x1.shape
		x1_norm = F.normalize(x1, p=2, dim=1)
		x2_norm = F.normalize(x2, p=2, dim=1)
		
		# L2 Distance version
		# x1_norm_expand = x1_norm.unsqueeze(1).expand(B, B, C)
		# x2_norm_expand = x2_norm.unsqueeze(0).expand(B, B, C)
		# l2_dist = torch.sum(torch.pow(x1_norm_expand-x2_norm_expand, 2), dim=2)

		# # compute positive mask and negative mask
		# positive_mask = (label1.view(-1,1) == label2.view(-1,1).permute(1, 0)).float()
		# negative_mask = (1 - positive_mask).float()

		# pos_dist = torch.sum(positive_mask * l2_dist, dim=1, keepdim=True)
		# # hardest negative mining
		# max_dist = torch.max(l2_dist, dim=1, keepdim=True)[0]
		# neg_dist = l2_dist + max_dist * (1.0 - negative_mask)
		# hardest_neg_dist = torch.min(neg_dist, dim=1, keepdim=True)[0]
		# # print("L2 dist:{}".format(l2_dist))
		# # print("positive dist:{}".format(pos_dist))
		# # print("hardest_neg_dist:{}".format(hardest_neg_dist))
		# # semi-hard mining: L2(anchor, pos) < L2(anchor, neg)
		# # hardest_pos_dist = torch.max(positive_mask * l2_dist, dim=1, keepdim=True)[0]
		# # semihard_mask = (negative_mask * l2_dist)>hardest_pos_dist
		# # semihard_dist = l2_dist * semihard_mask.float()
		# # semihard_dist = torch.sum(semihard_dist, dim=1, keepdim=True)

		# contrast_loss = torch.mean(pos_dist) + torch.mean(torch.clamp(self.margin - hardest_neg_dist, min=0.0))
		# return contrast_loss
	
		# Cosine version
		cos_dist = torch.matmul(x1_norm, x2_norm.permute(1,0))

		positive_mask = (label1.view(-1,1) == label2.view(-1,1).permute(1, 0)).float()
		negative_mask = (1 - positive_mask).float()

		pos_dist = torch.sum(positive_mask * (1-cos_dist), dim=1, keepdim=True)
		hardest_neg_dist = torch.max(negative_mask*cos_dist, dim=1, keepdim=True)[0]
		# semi-hard mining: Cosine(anchor, pos) > Cosine(anchor, neg)
		# hardest_pos_dist = torch.min(positive_mask * cos_dist, dim=1, keepdim=True)[0]
		# semihard_mask = (negative_mask * cos_dist) < hardest_pos_dist
		# semihard_dist = (negative_mask * cos_dist) * semihard_mask.float()
		# semihard_dist = torch.sum(semihard_dist, dim=1, keepdim=True)
		contrast_loss = torch.mean(pos_dist) + torch.mean(torch.clamp(hardest_neg_dist-self.margin, min=0.0))
		return self.scale * contrast_loss
		
		"""
		# N-pair CrossEntropy version
		cosine_dist = torch.matmul(x1, x2.permute(1, 0))
		label_mask = (label1.view(-1,1) == label2.view(-1,1).permute(1, 0)).float()
		label_mask = label_mask/torch.sum(label_mask, dim=1, keepdim=True)

		loss_ce = F.binary_cross_entropy_with_logits(cosine_dist, label_mask)
		
		loss_l2reg = torch.sum(torch.pow(x1, 2), dim=1) + torch.sum(torch.pow(x2, 2), dim=1)
		loss_l2reg = torch.mean(loss_l2reg)
		loss_l2reg = 0.25 * loss_l2reg * 0.02

		return loss_ce + loss_l2reg
		"""
 		
if __name__ == '__main__':
	contrast_loss = MultiContrastiveLoss(margin=5, scale=1.0)
	x1 = torch.tensor([[5,3,6],[3,4,5],[3,1,3]], dtype=torch.float).cuda()
	x2 = torch.tensor([[3,2,1], [0,2,0],[3,1,1]], dtype=torch.float).cuda()
	label1 = torch.tensor([0, 1, 0], dtype=torch.long).cuda()
	label2 = torch.tensor([0, 1, 0], dtype=torch.long).cuda()
	print(contrast_loss(x1, x2, label1, label2))
