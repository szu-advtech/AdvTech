# -*- coding: utf-8 -*-
import os
import os.path as osp
import re
import numpy as np
#from collections import OrderedDict

# for FRGC

def gen_recognition_id(data_root):
	with open(osp.join(data_root,'FRGC_id.txt'), 'w') as id_f:
		cls_num = 0
		category = set()
		pattern = r'\d{5}d\w+'
		for root, dirs, filenames in sorted(os.walk(data_root), key=lambda x: x[0]):
			for file in filenames:
				if re.match(pattern, file) is not None:
					lb = root.split(osp.sep)[-1] # file name e.g. 04753d18
					if lb in category:
						continue
					else:
						category.add(lb)
						id_f.write(lb+'\t'+str(cls_num)+'\n')
						cls_num+=1
	print("*****done*****")


def create_label_maps_for_recognition(data_root):
	category = {}
	print("*****Reading FRGC_id.txt*****")
	with open(osp.join(data_root, 'FRGC_id.txt'), 'r') as id_f:
		lines = id_f.readlines()
		for line in lines:
			lb_name, lb_id = line.strip().split("\t")
			category[lb_name] = lb_id
	print("*****Writting FRGC_id_all.txt*****")
	pattern = r'\d{5}d\w+'
	with open(osp.join(data_root, 'FRGC_id_all.txt'), 'w') as all_f:
		for root, dirs, filenames in os.walk(data_root):
			for file in filenames:
				#print(file, re.match(file, pattern))
				if re.match(pattern, file) is not None:
					lb_name = root.split(osp.sep)[-1] # file name e.g. bs000_CAU_A22A25_0 
					lb_id = category[lb_name]
					all_f.write(str(osp.join('',*root.split(osp.sep)[-2:], file))+'\t'+lb_id + '\n')


def split_train_test_for_id(data_root):
	train_set = {}
	test_set = {}
	with open(osp.join(data_root, 'Bos_id_all.txt'), 'r') as all_f:
		for line in all_f.readlines():
			f_path = line.strip().split('\t')[0]
			if include_cover == False:
				fn = f_path.split(osp.sep)[-1]
				fn_part = fn.split('_')[1]
				if fn_part in ['O']:
					continue
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

	txt_train = 'train_id.txt'
	txt_test = 'test_id.txt'

	with open(osp.join(data_root, txt_train), 'w') as train_f:
		for label_id in train_set.keys():
			for path in train_set[label_id]:
				train_f.write(path + '\t' + label_id + '\n')
	with open(osp.join(data_root, txt_test), 'w') as test_f:
		for label_id in test_set.keys():
			for path in test_set[label_id]:
				test_f.write(path + '\t' + label_id + '\n')
	print("*****done*****")	


def split_kfold_for_face_id(data_root):
	idmap = {}
	# read all samples in all_id.txt and store in idmap
	with open(osp.join(data_root,'Bos_id_all.txt')) as idf:
		for line in idf.readlines():
			line = line.strip()
			fname = line.split('\t')[0]
			label = line.split('\t')[1]
			if label not in idmap.keys():
				idmap[label] = [fname]
			else:
				idmap[label].append(fname)

	for label, fns in idmap.items():
		#sorted_fns = sorted(fns)
		sorted_fns = np.array(fns)
		num_samples = len(fns)
		idx = np.arange(0, num_samples)
		np.random.shuffle(idx)
		for fold in np.arange(0, 5):
			if fold == 4:
				one_for_test = sorted_fns[idx[fold*10:]].copy()
			else:
				one_for_test = sorted_fns[idx[fold*10: fold*10+10]].copy()
			with open(osp.join(data_root, 'train_{}.txt'.format(fold)), 'a') as trainf:
				for fname in sorted_fns:
					if fname not in one_for_test:
						trainf.write(fname + '\t' + str(label) +'\n')

			with open(osp.join(data_root, 'test_{}.txt'.format(fold)), 'a') as testf:
				for fname in one_for_test:
					testf.write(fname + '\t' + str(label) +'\n')

def split_probe_gallery_for_face_id(data_root):
	idmap = {}
	# read all samples in all_id.txt and store in idmap
	with open(osp.join(data_root,'FRGC_id_all.txt')) as idf:
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
		idx = np.random.randint(low=0, high=len(fns), dtype=np.int32)
		gallery_set[label] = fns[idx]
		if len(fns)==1:
			continue
		for i in np.arange(0, len(fns)):
			if i!=idx:
				if label in probe_set.keys():
					probe_set[label].append(fns[i])
				else:
					probe_set[label] = [fns[i]]


	with open(osp.join(data_root, 'probe.txt'), 'w') as probef:
		for label, fns in probe_set.items():
			for fn in fns:
				probef.write(str(fn) + '\t' + label + '\n')

	with open(osp.join(data_root, 'gallery.txt'), 'w') as galleryf:
		for label, fn in gallery_set.items():
			galleryf.write(str(fn) + '\t' + label + '\n')	



if __name__ == '__main__':
	#gen_recognition_id('/home/cv_jcy/Pytorch_Workspace/pointnet.pytorch/3d_dataset/FRGC_Downsample')
	#create_label_maps_for_recognition('/home/cv_jcy/Pytorch_Workspace/pointnet.pytorch/3d_dataset/FRGC_Downsample')
	split_probe_gallery_for_face_id('/home/cv_jcy/Pytorch_Workspace/pointnet.pytorch/3d_dataset/FRGC_Downsample')
	