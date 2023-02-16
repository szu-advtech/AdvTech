# -*- coding: utf-8 -*-
import os
import os.path as osp
import re
import numpy as np
#from collections import OrderedDict

# for BU3DFE
def gen_expression_id(data_root):
	with open(osp.join(data_root,'BU3DFE_expression_id.txt'), 'w') as id_f:
		cls_num = 0
		category = set()
		pattern = r'[FM]\d{4}_[A-Z0-9]{6}_F3D'
		for root, dirs, filenames in os.walk(data_root):
			for file in filenames:
				if re.match(pattern, file) is not None:
					lb = file.split('_')[1][0:2] #file name e.g. F0002_FE01BL_F3D
					if lb in category:
						continue
					else:
						category.add(lb)
						id_f.write(lb+'\t'+str(cls_num)+'\n')
						cls_num+=1
	print("*****done*****")

def gen_recognition_id(data_root):
	with open(osp.join(data_root,'BU3DFE_id.txt'), 'w') as id_f:
		cls_num = 0
		category = set()
		pattern = r'[FM]\d{4}_[A-Z0-9]{6}_F3D'
		for root, dirs, filenames in sorted(os.walk(data_root), key=lambda x: x[0]):
			for file in filenames:
				if re.match(pattern, file) is not None:
					lb = file.split('_')[0] #file name e.g. F0002_FE01BL_F3D
					if lb in category:
						continue
					else:
						category.add(lb)
						id_f.write(lb+'\t'+str(cls_num)+'\n')
						cls_num+=1
	print("*****done*****")

def create_label_maps_for_expression(data_root):
	category = {}
	print("*****Reading BU3DFE_expression_id.txt*****")
	with open(osp.join(data_root, 'BU3DFE_expression_id.txt'), 'r') as id_f:
		lines = id_f.readlines()
		for line in lines:
			lb_name, lb_id = line.strip().split("\t")
			category[lb_name] = lb_id
	print("*****Writting BU3DFE_expression_all.txt*****")
	pattern = r'[FM]\d{4}_[A-Z0-9]{6}_F3D'
	with open(osp.join(data_root, 'BU3DFE_expression_all.txt'), 'w') as all_f:
		for root, dirs, filenames in os.walk(data_root):
			for file in filenames:
				#print(file, re.match(file, pattern))
				if re.match(pattern, file) is not None:
					lb_name = file.split('_')[1][0:2] # file name e.g. F0002_FE01BL_F3D 
					lb_id = category[lb_name]
					all_f.write(str(osp.join('',*root.split(osp.sep)[-2:], file))+'\t'+lb_id + '\n')

def create_label_maps_for_recognition(data_root):
	category = {}
	print("*****Reading BU3DFE_id.txt*****")
	with open(osp.join(data_root, 'BU3DFE_id.txt'), 'r') as id_f:
		lines = id_f.readlines()
		for line in lines:
			lb_name, lb_id = line.strip().split("\t")
			category[lb_name] = lb_id
	print("*****Writting BU3DFE_id_all.txt*****")
	pattern = r'[FM]\d{4}_[A-Z0-9]{6}_F3D'
	with open(osp.join(data_root, 'BU3DFE_id_all.txt'), 'w') as all_f:
		for root, dirs, filenames in os.walk(data_root):
			for file in filenames:
				#print(file, re.match(file, pattern))
				if re.match(pattern, file) is not None:
					lb_name = file.split('_')[0] # file name e.g. F0002_FE01BL_F3D 
					lb_id = category[lb_name]
					all_f.write(str(osp.join('',*root.split(osp.sep)[-2:], file))+'\t'+lb_id + '\n')

def split_train_test_for_expression(data_root):
	data_list = [] # [(human1, filename, expression_id)]
	train_set = {}
	test_set = {}
	with open(osp.join(data_root, 'BU3DFE_expression_all.txt'), 'r') as all_f:
		for line in all_f.readlines():
			f_path = line.strip().split('\t')[0]
			label_id = line.strip().split('\t')[1]
			human_id = f_path.split(osp.sep)[1]
			data_list.append((human_id, f_path, label_id))
	#print(len(data_list[0]), len(data_list[4]))
	data_list = sorted(data_list, key=lambda x: x[0])
	print(len(data_list))
	print("*****Selecting testing file*****")
	np.random.seed(19970513)
	for fold in np.arange(0,10):
		train_set = {}
		test_set = {}	
		for i in np.arange(0, 100*25):
			f_path = data_list[i][1]
			label_id = data_list[i][2]
			if i not in np.arange(fold*10*25, fold*10*25+10*25):
				if label_id not in train_set.keys():
					train_set[label_id] = [f_path]
				else:
					train_set[label_id].append(f_path)
			else:
				if label_id not in test_set.keys():
					test_set[label_id] = [f_path]
				else:
					test_set[label_id].append(f_path)
		print("******Selection over*****")
		
		with open(osp.join(data_root, 'train_expression_{}.txt'.format(fold+1)), 'w') as train_f:
			for label_id in train_set.keys():
				for path in train_set[label_id]:
					train_f.write(path + '\t' + label_id + '\n')
		with open(osp.join(data_root, 'test_expression_{}.txt'.format(fold+1)), 'w') as test_f:
			for label_id in test_set.keys():
				for path in test_set[label_id]:
					test_f.write(path + '\t' + label_id + '\n')
		print("*****done*****")
		del train_set
		del test_set

def split_train_test_for_id(data_root):
	train_set = {}
	test_set = {}
	with open(osp.join(data_root, 'BU3DFE_id_all.txt'), 'r') as all_f:
		for line in all_f.readlines():
			f_path = line.strip().split('\t')[0]
			label_id = line.strip().split('\t')[1]
			if  label_id not in train_set.keys():
				train_set[label_id] = [f_path]
			else:
				train_set[label_id].append(f_path)
	#print(len(train_set['0']), len(train_set['4']))
	print("*****Selecting testing file*****")
	np.random.seed(19971210)	
	for label_id, f_paths in train_set.items():
		sample_num = len(f_paths)
		#print(sample_num)
		idx = np.random.choice(sample_num, int(round(sample_num/5)), replace = False)
		f_paths_np = np.array(f_paths, dtype=str)
		f_paths_np_test = f_paths_np[idx].copy()
		f_paths_test = f_paths_np_test.tolist()
		# move selected file name to testing set
		test_set[label_id] = f_paths_test
		
		# remove those selected file name in training set
		#print(idx, len(f_paths))
		for i in idx:
			f_paths[i] = ''
		for i in np.arange(f_paths.count('')):
			f_paths.remove('')

	print("******Selection over*****")
	
	with open(osp.join(data_root, './train_id.txt'), 'w') as train_f:
		for label_id in train_set.keys():
			for path in train_set[label_id]:
				train_f.write(path + '\t' + label_id + '\n')
	with open(osp.join(data_root, './test_id.txt'), 'w') as test_f:
		for label_id in test_set.keys():
			for path in test_set[label_id]:
				test_f.write(path + '\t' + label_id + '\n')
	print("*****done*****")	

def split_probe_gallery_for_face_id(data_root):
	idmap = {}
	# read all samples in all_id.txt and store in idmap
	with open(osp.join(data_root,'BU3DFE_id_all.txt')) as idf:
		for line in idf.readlines():
			line = line.strip()
			fname = line.split('\t')[0]
			label = line.split('\t')[1]
			if label not in idmap.keys():
				idmap[label] = [fname]
			else:
				idmap[label].append(fname)
	probe_set = {}
	gallery_set = {}
	# Neutral face for gallery
	for label, fns in idmap.items():
		for fn in fns:
			fn_keyword = fn.split(osp.sep)[-1]
			"""need code here"""
			if fn_keyword.split('_')[1][0:2] == 'NE':
				gallery_set[label] = fn
			else:
				if label not in probe_set.keys():
					probe_set[label] = [fn]
				else:
					probe_set[label].append(fn)

	with open(osp.join(data_root, 'probe.txt'), 'w') as probef:
		for label, fns in probe_set.items():
			for fn in fns:
				probef.write(str(fn) + '\t' + label + '\n')

	with open(osp.join(data_root, 'gallery.txt'), 'w') as galleryf:
		for label, fn in gallery_set.items():
			galleryf.write(str(fn) + '\t' + label + '\n')	

if __name__ == '__main__':
	#gen_expression_id('/home/cv_jcy/Pytorch_Workspace/pointnet.pytorch/3d_dataset/Only_pts_BU3DFE')
	#create_label_maps_for_expression('/home/cv_jcy/Pytorch_Workspace/pointnet.pytorch/3d_dataset/Only_pts_BU3DFE')
	#split_train_test_for_expression('/home/cv_jcy/Pytorch_Workspace/pointnet.pytorch/3d_dataset/Only_pts_BU3DFE')
	#gen_recognition_id('/home/cv_jcy/Pytorch_Workspace/pointnet.pytorch/3d_dataset/Only_pts_BU3DFE')
	#create_label_maps_for_recognition('/home/cv_jcy/Pytorch_Workspace/pointnet.pytorch/3d_dataset/Only_pts_BU3DFE')
	#split_train_test_for_id('/home/cv_jcy/Pytorch_Workspace/pointnet.pytorch/3d_dataset/Only_pts_BU3DFE')
	split_probe_gallery_for_face_id('/home/cv_jcy/Pytorch_Workspace/pointnet.pytorch/3d_dataset/Only_pts_BU3DFE')