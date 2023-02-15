# -*- coding: utf-8 -*-
from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)

import torch
import torch.utils.data as data
import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
index = str(BASE_DIR).index('data')
if not BASE_DIR[0:index] in sys.path:
  sys.path.append(BASE_DIR[0:index])

import utils.pointnet2_utils as pointnet2_utils

def pc_normalize(pc):
	# pc.shape[0] is the number of points
	# pc.shape[1] is the number of feature dimensions
	# first of 3 dimensions should be (x,y,z)
	centroid = np.mean(pc[:,0:3], axis=0)
	pc[:,0:3] = pc[:,0:3] - centroid
	dist = np.max(np.sqrt(np.sum(pc[:,0:3]**2, axis=1)))
	pc[:,0:3] = pc[:,0:3] / dist
	return pc

def _get_data_files(list_filename):
	with open(list_filename) as f:
		content = f.readlines()
		filenames = [line.strip().split('\t')[0] for line in content]
		labels = [int(line.strip().split('\t')[1]) for line in content]
		return  filenames, labels

class BU3_Bos(data.Dataset):
	def __init__(self, num_points, root, transforms=None, train=True, task='id'):
		super().__init__()

		self.transforms = transforms
		self.num_points = num_points
		self.root = os.path.abspath(root)
		self.folder = 'BU3_Bos'
		self.data_dir = os.path.join(self.root, self.folder)

		self.train = train
		if task=='expression':
			if self.train:
				self.files, self.labels = _get_data_files(
											os.path.join(self.data_dir, 'train_expression_0.txt')
											)
			else:
				self.files, self.labels = _get_data_files(
											os.path.join(self.data_dir, 'test_expression_0.txt')
											)
		elif task=='id':
			if self.train:
				self.files, self.labels = _get_data_files(
											os.path.join(self.data_dir, 'BU3DFEandBos_id_all.txt')
											)

		# self.points = []
		np.random.seed(19970513)
		# self.points = np.stack(self.points, axis=0) # need to be tested
		# self.labels = np.array(self.labels)

	def __getitem__(self, idx):
		point_file, this_label = self.files[idx], self.labels[idx]
		single_p = np.loadtxt(os.path.join(self.root, point_file)) # or self.data_dir
		single_p = pc_normalize(single_p)
		if self.num_points > single_p.shape[0]:
			idx = np.ones(single_p.shape[0], dtype=np.int32)
			idx[-1] = self.num_points - single_p.shape[0] + 1
			single_p_part = np.repeat(single_p, idx, axis=0)
		else:
			# single_p_tensor = torch.from_numpy(single_p).type(torch.FloatTensor).cuda()
			# single_p_tensor = single_p_tensor.unsqueeze(0) # change to (1, N, 3)
			# fps_idx = pointnet2_utils.furthest_point_sample(single_p_tensor,self.num_points) # (1, npoint)
			# single_p_tensor = pointnet2_utils.gather_operation(single_p_tensor.transpose(1,2).contiguous(), 
			# 													fps_idx).transpose(1,2).contiguous() # (1, npoint, 3)
			# single_p_part = single_p_tensor.squeeze(0).cpu().numpy()
			single_p_idx = np.arange(0, single_p.shape[0])
			single_p_idx = np.random.choice(single_p_idx, size=(self.num_points,), replace=False)
			single_p_part = single_p[single_p_idx, :]

		points = single_p_part.copy()

		pt_idxs = np.arange(0, poinst.shape[0])
		if self.train:
			np.random.shuffle(pt_idxs)

		current_points = points[idx, pt_idxs]
		if self.transforms is not None:
			current_points = self.transforms(current_points)
		else:
			current_points = torch.from_numpy(current_points).float()

		label = torch.tensor([this_label], dtype=torch.long)

		return current_points, label

	def __len__(self):
		return len(self.files)

class FRGC_eval(data.Dataset):
	def __init__(self, num_points, root, transforms=None, task='id'):
		super().__init__()

		self.transforms = transforms
		self.num_points = num_points
		self.root = os.path.abspath(root)
		#self.folder = 'Only_pts_Bos'
		self.folder = 'FRGC_Downsample'
		self.data_dir = os.path.join(self.root, self.folder)

		if task=='id' :
			self.probe_files, self.probe_labels = _get_data_files(
										os.path.join(self.data_dir, 'probe.txt')
										)
			
			self.gallery_files, self.gallery_labels = _get_data_files(
										os.path.join(self.data_dir, 'gallery.txt')
										)

		self.probe_points = []
		self.gallery_points = []
		for file in self.probe_files:
			single_p = np.loadtxt(os.path.join(self.root, file))
			single_p = pc_normalize(single_p)
			if self.num_points > single_p.shape[0]:
				idx = np.ones(single_p.shape[0], dtype=np.int32)
				idx[-1] = self.num_points - single_p.shape[0] + 1
				single_p_part = np.repeat(single_p, idx, axis=0)
			else:
				#idxs = np.random.choice(single_p.shape[0], self.num_points, replace=False)
				#single_p_part = single_p[idxs].copy()
				single_p_tensor = torch.from_numpy(single_p).type(torch.FloatTensor).cuda()
				single_p_tensor = single_p_tensor.unsqueeze(0) # change to (1, N, 3)
				fps_idx = pointnet2_utils.furthest_point_sample(single_p_tensor,self.num_points) # (1, npoint)
				single_p_tensor = pointnet2_utils.gather_operation(single_p_tensor.transpose(1,2).contiguous(), 
																	fps_idx).transpose(1,2).contiguous() # (1, npoint, 3)
				single_p_part = single_p_tensor.squeeze(0).cpu().numpy()

			#print(single_p.shape)
			self.probe_points.append(single_p_part)
				
		self.probe_points = np.stack(self.probe_points, axis=0)
		self.probe_labels = np.array(self.probe_labels)
		

	def __getitem__(self, idx):
		#pt_idxs = np.arange(0, self.points.shape[1])
		#np.random.shuffle(pt_idxs)

		current_points = self.probe_points[idx, :].copy()
		if self.transforms is not None:
			current_points = self.transforms(current_points)
		else:
			current_points = torch.from_numpy(current_points).float()

		probe_label = torch.Tensor([self.probe_labels[idx]]).type(torch.LongTensor)

		return current_points, probe_label

	def __len__(self):
		return self.probe_points.shape[0]

	def get_gallery(self):
		for file in self.gallery_files:
			single_p = np.loadtxt(os.path.join(self.root, file))
			single_p = pc_normalize(single_p)
			if self.num_points > single_p.shape[0]:
				idx = np.ones(single_p.shape[0], dtype=np.int32)
				idx[-1] = self.num_points - single_p.shape[0] + 1
				single_p_part = np.repeat(single_p, idx, axis=0)
			else:
				#idxs = np.random.choice(single_p.shape[0], self.num_points, replace=False)
				#single_p_part = single_p[idxs].copy()
				single_p_tensor = torch.from_numpy(single_p).type(torch.FloatTensor).cuda()
				single_p_tensor = single_p_tensor.unsqueeze(0) # change to (1, N, 3)
				fps_idx = pointnet2_utils.furthest_point_sample(single_p_tensor,self.num_points) # (1, npoint)
				single_p_tensor = pointnet2_utils.gather_operation(single_p_tensor.transpose(1,2).contiguous(), 
																	fps_idx).transpose(1,2).contiguous() # (1, npoint, 3)
				single_p_part = single_p_tensor.squeeze(0).cpu().numpy()

			#print(single_p.shape)
			self.gallery_points.append(single_p_part)

		self.gallery_points = np.stack(self.gallery_points, axis=0)  # need to be tested
		self.gallery_labels = np.array(self.gallery_labels)

		if self.transforms is not None:
			g_points = self.transforms(self.gallery_points)
		else:
			g_points = torch.from_numpy(self.gallery_points).float()

		g_labels = torch.from_numpy(self.gallery_labels).type(torch.LongTensor)
		return g_points, g_labels


class FRGC_train_pair(data.Dataset):
	def __init__(self, num_points, root, transforms=None, train=True):
		super().__init__()

		self.transforms = transforms
		self.num_points = num_points
		self.root = os.path.abspath(root)
		self.folder = 'FRGC_with_virtual_id' # need to be filled
		self.data_dir = os.path.join(self.root, self.folder)

		self.train = train
		if self.train:
			self.files, self.labels = _get_data_files(
										os.path.join(self.data_dir, 'FRGC_virtual_id_all.txt')
										)
		else:
			self.files, self.labels = _get_data_files(
										os.path.join(self.data_dir, 'FRGC_virtual_id_all.txt')
										)

		# self.pair_files = self.form_pair()
		self.pair_files = self.form_positive_pair(self.files, self.labels)

		#self.points1 = np.stack(self.points1, axis=0) # need to be tested
		#self.points2 = np.stack(self.points2, axis=0)
		##self.points = np.array(self.points)
		#self.labels1 = np.array(self.labels1)
		#self.labels2 = np.array(self.labels2)

	def __getitem__(self, idx):
		file1, label1, file2, label2 = self.pair_files[idx]
		single_p1 = np.loadtxt(os.path.join(self.root, file1))
		single_p1 = pc_normalize(single_p1)

		single_p2 = np.loadtxt(os.path.join(self.root, file2))
		single_p2 = pc_normalize(single_p2)
		if self.num_points > single_p1.shape[0]:
			idx = np.ones(single_p1.shape[0], dtype=np.int32)
			idx[-1] = self.num_points - single_p1.shape[0] + 1
			single_p_part = np.repeat(single_p1, idx, axis=0)
		else:
			# single_p_tensor = torch.from_numpy(single_p1).type(torch.FloatTensor).cuda()
			# single_p_tensor = single_p_tensor.unsqueeze(0) # change to (1, N, 3)
			# fps_idx = pointnet2_utils.furthest_point_sample(single_p_tensor,self.num_points) # (1, npoint)
			# single_p_tensor = pointnet2_utils.gather_operation(single_p_tensor.transpose(1,2).contiguous(), 
			# 													fps_idx).transpose(1,2).contiguous() # (1, npoint, 3)
			# single_p_part = single_p_tensor.squeeze(0).cpu().numpy()
			
			single_p_idx = np.arange(0, single_p1.shape[0])
			single_p_idx = np.random.choice(single_p_idx, size=(self.num_points,), replace=False)
			single_p_part = single_p1[single_p_idx, :]

		points1 = single_p_part.copy()

		if self.num_points > single_p2.shape[0]:
			idx = np.ones(single_p2.shape[0], dtype=np.int32)
			idx[-1] = self.num_points - single_p2.shape[0] + 1
			single_p_part = np.repeat(single_p2, idx, axis=0)
		else:		
			# single_p_tensor = torch.from_numpy(single_p2).type(torch.FloatTensor).cuda()
			# single_p_tensor = single_p_tensor.unsqueeze(0) # change to (1, N, 3)
			# fps_idx = pointnet2_utils.furthest_point_sample(single_p_tensor,self.num_points) # (1, npoint)
			# single_p_tensor = pointnet2_utils.gather_operation(single_p_tensor.transpose(1,2).contiguous(), 
			# 													fps_idx).transpose(1,2).contiguous() # (1, npoint, 3)
			# single_p_part = single_p_tensor.squeeze(0).cpu().numpy()
			
			single_p_idx = np.arange(0, single_p2.shape[0])
			single_p_idx = np.random.choice(single_p_idx, size=(self.num_points,), replace=False)
			single_p_part = single_p2[single_p_idx, :]
		#print(single_p_part.shape)
		points2 = single_p_part.copy()
		
		# points1 and points2 are the same shape now
		pt_idxs1 = np.arange(0, points1.shape[0])
		pt_idxs2 =np.arange(0, points2.shape[0])
		if self.train:
			np.random.shuffle(pt_idxs1)
			np.random.shuffle(pt_idxs2)

		current_points1, current_points2 = points1[pt_idxs1], points2[pt_idxs2]
		if self.transforms is not None:
			current_points1 = self.transforms(current_points1)
			current_points2 = self.transforms(current_points2)
		else:
			current_points1 = torch.from_numpy(current_points1).float()
			current_points2 = torch.from_numpy(current_points2).float()

		label1 = torch.Tensor([label1]).type(torch.LongTensor)
		label2 = torch.Tensor([label2]).type(torch.LongTensor)

		return current_points1, label1, current_points2, label2

	def __len__(self):
		return len(self.pair_files)

	def form_pair(self):
		filelist2 = {} # key: label ; value: file name
		for i in np.arange(len(self.files)):
			if not self.labels[i] in filelist2.keys():
				filelist2[self.labels[i]] = [self.files[i]]
			else:
				filelist2[self.labels[i]].append(self.files[i])
		pair_list = []
		for i in np.arange(len(self.files)):
			# positive sample
			pos = -1
			this_name, this_label = self.files[i], self.labels[i]
			# for j in np.arange(len(filelist2[this_label])):
			# 	if filelist2[this_label][j] == this_name:
			# 		pos = j
			# 		break
			pos = filelist2[this_label].index(this_name)
			idx = np.random.randint(low=1, high=len(filelist2[this_label]))
			idx = (pos+idx)%len(filelist2[this_label]) # avoid picking the same sample
			pair_list.append((this_name, this_label, filelist2[this_label][idx], this_label))
			# negative sample
			p_idx = np.random.randint(low=1, high=len(filelist2.keys()))
			p_idx = (this_label+p_idx)%len(filelist2.keys())
			idx = np.random.randint(low=0, high=len(filelist2[p_idx]))
			pair_list.append((this_name, this_label, filelist2[p_idx][idx], p_idx))
		#for i in np.arange(15):
		#	print(pair_list[i])
		return pair_list

	@staticmethod
	def form_positive_pair(files, labels):
		filelist2 = {}
		for i in np.arange(len(files)):
			if not labels[i] in filelist2.keys():
				filelist2[labels[i]] = [files[i]]
			else:
				filelist2[labels[i]].append(files[i])

		pair_list = []
		for i in np.arange(len(files)):
			# form different pair of positive samples
			# pos = -1
			this_name, this_label = files[i], labels[i]
			# for j in np.arange(len(filelist2[this_label])):
			# 	if this_name == filelist2[this_label][j]:
			# 		pos = j
			# 		break
			pos = filelist2[this_label].index(this_name)
			idx = (pos+1) % len(filelist2[this_label])
			pair_list.append((this_name, this_label, filelist2[this_label][idx], this_label))
		return pair_list


if __name__ == '__main__':
	"""dset = Bosphorus(4000, "/home/cv_jcy/Pytorch_Workspace/pointnet.pytorch/3d_dataset/", train=True)
	print(len(dset), dset.points.shape)
	print(dset[99][0])
	print(dset[99][1])
	print(len(dset))"""

	# eval_set = FRGC_eval(500, root="/home/cv_jcy/Pytorch_Workspace/pointnet.pytorch/3d_dataset/")
	# print(eval_set[0][0].shape)
	# print(eval_set[0][1].shape)
	# gpts, glabels = eval_set.get_gallery()
	# print('gallery points: {} \t gallery labels: {}'.format(gpts.shape, glabels.shape))
	# print(len(eval_set))

	pair_set = FRGC_train_pair(100, root="/home/cv_jcy/Pytorch_Workspace/pointnet.pytorch/3d_dataset/")
	print(pair_set.pair_files[0:10])

