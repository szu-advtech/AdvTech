# -*- coding: utf-8 -*-
import os
import os.path as osp
import re
import numpy as np
#from collections import OrderedDict

# for BosphorusDB

def gen_recognition_id(data_root):
	with open(osp.join(data_root,'Bos_id.txt'), 'w') as id_f:
		cls_num = 0
		category = set()
		pattern = r'bs\d{3}_\w+'
		for root, dirs, filenames in sorted(os.walk(data_root), key=lambda x: x[0]):
			for file in filenames:
				if re.match(pattern, file) is not None:
					lb = file.split('_')[0] # file name e.g. bs000_CAU_A22A25_0
					if lb in category:
						continue
					else:
						category.add(lb)
						id_f.write(lb+'\t'+str(cls_num)+'\n')
						cls_num += 1
	print("*****done*****")


def create_label_maps_for_recognition(data_root, data_dir=""):
	category = {}
	print("*****Reading Bos_id.txt*****")
	with open(osp.join(data_root, 'Bos_id.txt'), 'r') as id_f:
		lines = id_f.readlines()
		for line in lines:
			lb_name, lb_id = line.strip().split("\t")
			category[lb_name] = lb_id
	print("*****Writting Bos_id_all.txt*****")
	pattern = r'bs\d{3}_\w+'
	with open(osp.join(data_root, 'Bos_id_all.txt'), 'w') as all_f:
		for root, dirs, filenames in os.walk(osp.join(data_root, data_dir)):
			for file in filenames:
				#print(file, re.match(file, pattern))
				if re.match(pattern, file) is not None:
					lb_name = file.split('_')[0] # file name e.g. bs000_CAU_A22A25_0 
					lb_id = category[lb_name]
					# all_f.write(str(osp.join('',*root.split(osp.sep)[-2:], file))+'\t'+lb_id + '\n')
					normal_filename = file.split('.')[0] + '_normal.' + file.split('.')[1]
					all_f.write(str(osp.join('',*root.split(osp.sep)[-1:], file)) + '\t' +
						str(osp.join('',*root.split(osp.sep)[-1:], normal_filename)) + '\t' +
						lb_id + '\n')

def create_label_maps_for_expression(data_root):
	category = {}
	print("*****Reading Bos_expression_id.txt*****")
	with open(osp.join(data_root, 'Bos_expression_id.txt'), 'r') as id_f:
		lines = id_f.readlines()
		for line in lines:
			lb_name, lb_id = line.strip().split("\t")
			category[lb_name] = lb_id
	print("*****Writting Bos_expression_all.txt*****")
	pattern = r'bs\d*_[EN]_\w*'
	with open(osp.join(data_root, 'Bos_expression_all.txt'), 'w') as all_f:
		for root, dirs, filenames in os.walk(data_root):
			for file in filenames:
				#print(file, re.match(file, pattern))
				if re.match(pattern, file) is not None:
					lb_name = file.split('_')[2][0:2] # file name e.g. bs000_E_HAPPY_0
					if lb_name == 'N':
						lb_name = 'NE' 
					lb_id = category[lb_name]
					all_f.write(str(osp.join('',*root.split(osp.sep)[-2:], file))+'\t'+lb_id + '\n')


def split_train_test_for_expression(data_root):
	data_list = [] # [(human1, filename, expression_id)]
	train_set = {}
	test_set = {}
	with open(osp.join(data_root, 'Bos_expression_all.txt'), 'r') as all_f:
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
	for fold in np.arange(0,5):
		train_set = {}
		test_set = {}	
		for i in np.arange(0, len(data_list)):
			human_id = int(data_list[i][0][2:])
			f_path = data_list[i][1]
			label_id = data_list[i][2]
			if human_id not in np.arange(fold*21, fold*21+21):
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
		
		with open(osp.join(data_root, 'train_expression_{}.txt'.format(fold)), 'w') as train_f:
			for label_id in train_set.keys():
				for path in train_set[label_id]:
					train_f.write(path + '\t' + label_id + '\n')
		with open(osp.join(data_root, 'test_expression_{}.txt'.format(fold)), 'w') as test_f:
			for label_id in test_set.keys():
				for path in test_set[label_id]:
					test_f.write(path + '\t' + label_id + '\n')
		print("*****done*****")
		del train_set
		del test_set

def split_train_test_for_id(data_root, include_cover=True):
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
	if include_cover == True:
		txt_train = 'train_id.txt'
		txt_test = 'test_id.txt'
	else:
		txt_train = 'train_id_uncover.txt'
		txt_test = 'test_id_uncover.txt'
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
	with open(osp.join(data_root,'Bos_id_all.txt'), 'r') as idf:
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
			if '_N_N_0' in str(fn_keyword):
				if label not in gallery_set.keys():
					gallery_set[label] = [fn]
				else:
					gallery_set[label].append(fn)
			#elif fn_keyword.split('_')[1] in ['E', 'N', 'CAU', 'LFAU', 'UFAU']:
			else:
				if label not in probe_set.keys():
					probe_set[label] = [fn]
				else:
					probe_set[label].append(fn)

	with open(osp.join(data_root, 'probe.txt'), 'w') as probef:
		for label, fns in probe_set.items():
			for fn in fns:
				if str(fn.split(osp.sep)[-1]).find('_noise')<0 :
					probef.write(str(fn) + '\t' + label + '\n')

	with open(osp.join(data_root, 'probe_lq.txt'), 'w') as probef:
		for label, fns in probe_set.items():
			for fn in fns:
				if str(fn.split(osp.sep)[-1]).find('_noise')>0 :
					probef.write(str(fn) + '\t' + label + '\n')

	with open(osp.join(data_root, 'gallery.txt'), 'w') as galleryf:
		for label, fns in gallery_set.items():
			for fn in fns:
				if str(fn.split(osp.sep)[-1]).find('_noise')<0 :
					galleryf.write(str(fn) + '\t' + label + '\n')

	with open(osp.join(data_root, 'gallery_lq.txt'), 'w') as galleryf:
		for label, fns in gallery_set.items():
			for fn in fns:
				if str(fn.split(osp.sep)[-1]).find('_noise')>0 :
					galleryf.write(str(fn) + '\t' + label + '\n')


def split_probe_gallery_for_face_id_with_normal(data_root):
	idmap = {}
	# read all samples in all_id.txt and store in idmap
	with open(osp.join(data_root,'Bos_id_all.txt'), 'r') as idf:
		for line in idf.readlines():
			line = line.strip()
			fname = line.split('\t')[0]
			normal_name = line.split('\t')[1]
			label = line.split('\t')[2]
			if label not in idmap.keys():
				idmap[label] = [(fname, normal_name)]
			else:
				idmap[label].append((fname, normal_name))
	probe_set = {}
	gallery_set = {}
	# Neutral face for gallery
	for label, fns in idmap.items():
		for fn in fns:
			fn_keyword = fn[0].split(osp.sep)[-1]
			if '_N_N_0' in str(fn_keyword):
				if label not in gallery_set.keys():
					gallery_set[label] = [fn]
				else:
					gallery_set[label].append(fn)
			#elif fn_keyword.split('_')[1] in ['E', 'N', 'CAU', 'LFAU', 'UFAU']:
			else:
				if label not in probe_set.keys():
					probe_set[label] = [fn]
				else:
					probe_set[label].append(fn)

	with open(osp.join(data_root, 'probe.txt'), 'w') as probef:
		for label, fns in probe_set.items():
			for fn in fns:
				if str(fn[0].split(osp.sep)[-1]).find('_noise')<0 :
					probef.write(str(fn[0]) + '\t' + str(fn[1]) + '\t' + label + '\n')

	with open(osp.join(data_root, 'probe_lq.txt'), 'w') as probef:
		for label, fns in probe_set.items():
			for fn in fns:
				if str(fn[0].split(osp.sep)[-1]).find('_noise')>0 :
					probef.write(str(fn[0]) + '\t' + str(fn[1]) + '\t' + label + '\n')

	with open(osp.join(data_root, 'gallery.txt'), 'w') as galleryf:
		for label, fns in gallery_set.items():
			for fn in fns:
				if str(fn[0].split(osp.sep)[-1]).find('_noise')<0 :
					galleryf.write(str(fn[0]) + '\t' + str(fn[1]) + '\t' + label + '\n')

	with open(osp.join(data_root, 'gallery_lq.txt'), 'w') as galleryf:
		for label, fns in gallery_set.items():
			for fn in fns:
				if str(fn[0].split(osp.sep)[-1]).find('_noise')>0 :
					galleryf.write(str(fn[0]) + '\t' + str(fn[1]) + '\t' + label + '\n')



def generate_noisy_data(dataroot, targetroot):
	"""generate noisy data from normal distribution N(0,16)"""
	if not osp.exists(targetroot):
		os.mkdir(targetroot)

	pattern = r'bs\d{3}_\w+' # file name e.g. bs000_CAU_A22A25_0
	np.random.seed(123)
	for root, dirs, files in os.walk(dataroot):
		for sub_dir in dirs:
			if not osp.exists(osp.join(targetroot, sub_dir)):
				os.mkdir(osp.join(targetroot, sub_dir))

		for file in files:
			if re.match(pattern, file) is None:
				continue
			pc_ori = np.loadtxt(osp.join(root, file))
			print(file, pc_ori.shape)
			jitter_off = np.random.normal(0, 1, pc_ori.shape)
			pc_jittered = pc_ori + jitter_off

			directory = root.split(osp.sep)[-1]
			np.savetxt(osp.join(targetroot, directory, file), pc_jittered, fmt='%.3f', delimiter=" ")

def downsample_and_save(dataroot, target_dir, pattern):
    for root, dirs, files in os.walk(dataroot):
        for directory in dirs:
            if not osp.exists(osp.join(target_dir, directory)):
                os.mkdir(osp.join(target_dir, directory))
        for file in files:
            try:
            	if re.match(pattern, file):
                	points = np.loadtxt(osp.join(root, file))
                
            except Exception as e:
                continue
            else:
                p1_pc = o3d.geometry.PointCloud()
                p1_pc.points = o3d.utility.Vector3dVector(points)
                # downsample
                down_p1_pc = p1_pc.voxel_down_sample(voxel_size=2)
                down_points = np.asarray(down_p1_pc.points)
                # save
                filename = file.split('.')[0]
                directory = root.split(osp.sep)[-1]
                np.savetxt(osp.join(target_dir, directory, filename), down_points, fmt='%.3f',delimiter=' ')
                print("Save to {}".format(osp.join(target_dir, directory, filename)))



if __name__ == '__main__':
	pattern_Bos = r'bs\d{3}_*'
	#create_label_maps_for_expression('/home/cv_jcy/Pytorch_Workspace/pointnet.pytorch/3d_dataset/Only_pts_Bos')
	#split_train_test_for_expression('/home/cv_jcy/Pytorch_Workspace/pointnet.pytorch/3d_dataset/Only_pts_Bos')
	# gen_recognition_id('/home/cv_jcy/Pytorch_Workspace/pointnet.pytorch/3d_dataset/Bos_All_Downsample_crop')
	# create_label_maps_for_recognition('/home/cv_jcy/Pytorch_Workspace/pointnet.pytorch/3d_dataset/Bos_All_Downsample_crop')
	#split_train_test_for_id('/home/cv_jcy/Pytorch_Workspace/pointnet.pytorch/3d_dataset/Bos_Downsample')
	#split_kfold_for_face_id('/home/cv_jcy/Pytorch_Workspace/pointnet.pytorch/3d_dataset/Only_pts_Bos')
	#create_label_maps_for_recognition('/home/cv_jcy/Pytorch_Workspace/pointnet.pytorch/3d_dataset/Bos_Downsample')
	#split_probe_gallery_for_face_id('/home/cv_jcy/Pytorch_Workspace/pointnet.pytorch/3d_dataset/Bos_Downsample')
	# generate_noisy_data('/home/cv_jcy/Pytorch_Workspace/pointnet.pytorch/3d_dataset/Bos_All_Downsample/', 
	# 	'/home/cv_jcy/Pytorch_Workspace/pointnet.pytorch/3d_dataset/Bos_All_Downsample_Noisy/')
	#create_label_maps_for_recognition('/home/cv_jcy/Pytorch_Workspace/pointnet.pytorch/3d_dataset/Bos_All_Downsample/')
	# split_probe_gallery_for_face_id('/home/cv_jcy/Pytorch_Workspace/pointnet.pytorch/3d_dataset/Bos_All_Downsample_crop/')
	
	# for depth images dataset
	# gen_recognition_id('/NAS_REMOTE/cv_jcy/BosphorusDB_2Dmap/depth')
	# create_label_maps_for_recognition('/NAS_REMOTE/cv_jcy/BosphorusDB_2Dmap/', data_dir='depth')
	# split_probe_gallery_for_face_id('/NAS_REMOTE/cv_jcy/BosphorusDB_noise_depth')
	split_probe_gallery_for_face_id_with_normal('/NAS_REMOTE/cv_jcy/BosphorusDB_2Dmap/')
