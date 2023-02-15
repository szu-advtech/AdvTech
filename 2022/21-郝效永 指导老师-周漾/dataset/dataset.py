import os
import random
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from PIL import Image
import PIL
import shutil
from data.skeleton import get_label_tensor

class SelfDataset():
    def __init__(self,root) -> None:
        self.root = root
        self.size = 0
        ratio_h_w = float(1101) / 750
        self.target_h = int(128 * ratio_h_w)
        self.crop_center = self.generate_cropcenter()
        self.label_paths = []
        self.image_paths = []
        self.self_pair_flag = {}
        self.ref_dict = self.get_ref()
        
        self.getpath()
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        pass
    
    def get_ref(self):
        self_pair_flag = {}
        pair_path = './dataset/deepfashion_self_pair.txt'
        with open(pair_path) as fd:
            self_pair = fd.readlines()
            self_pair = [it.strip() for it in self_pair]
        self_pair_dict = {}
        for it in self_pair:
            items = it.split(',')
            for i in range(len(items)):
                items[i] = items[i].replace('img','train_img')
            self_pair_dict[items[0]] = items[1:]
        
        ref_path = './dataset/deepfashion_ref.txt'
        with open(ref_path) as fd:
            ref = fd.readlines()
            ref = [it.strip() for it in ref]
        ref_dict = {}
        for i in range(len(ref)):
            items = ref[i].strip().split(',')
            for i in range(len(items)):
                items[i] = items[i].replace('img','train_img')
            key = items[0]
            if key in self_pair_dict.keys():
                val = [it for it in self_pair_dict[items[0]]]
                self_pair_flag[key.replace('\\','/')] = True
            else:
                val = [items[-1]]
                self_pair_flag[key.replace('\\','/')] = False
            ref_dict[key.replace('\\','/')] = [v.replace('\\','/') for v in val]

        train_test_folder = ('','')
        self.self_pair_flag = self_pair_flag
        return ref_dict

        pass

    def generate_cropcenter(self):
        height = int(128 * (float(1101)/750))
        center_y_max = height - 64
        center_y_min = 64
        center_y = int(random.uniform(0,1) * (center_y_max - center_y_min) + center_y_min)
        return center_y
        pass

    def getpath(self):
        root = self.root
        fd = open(os.path.join('./dataset/train.txt'))
        lines = fd.readlines()
        fd.close()

        #for i in range(len(lines)):
        for i in range(len(lines)):
            name = lines[i].strip()
            image_name = root + name.replace("img","train_img")
            self.image_paths.append(image_name)
            label_name = image_name.replace("train_img","pose_img")
            self.label_paths.append(label_name)
            self.size = self.size + 1
        
        pass
    def imgpath_to_labelpath(self,imgpath):
        labelpath = imgpath.replace('train_img','pose_img')
        if not os.path.exists(labelpath):
            findpath = labelpath.replace('dataset','data')
            findpath = findpath.replace('pose_img','pose')
            candidatepath = findpath.replace('.jpg','_candidate.txt')
            subsetpath = findpath.replace('.jpg','_subset.txt')
            poseimg,_ = get_label_tensor(candidatepath,subsetpath,crop_center=self.crop_center,target_h=self.target_h)
            poseimg.save(labelpath)
        return labelpath
        pass

    def labelpath_to_segtensor(self,labelpath):
        segpath = labelpath.replace('pose_img','pose')
        segpath = segpath.replace('dataset','data')
        candidatepath = segpath.replace('.jpg','_candidate.txt')
        subsetpath = segpath.replace('.jpg','_subset.txt')
        _,segtensor = get_label_tensor(candidatepath,subsetpath,crop_center=self.crop_center,target_h=self.target_h,is_seg=True)
        return segtensor

    def __getitem__(self,index):
        self.crop_center = self.generate_cropcenter()
        elf_ref_flag = 0.0
        #label_image 1024*1024
        label_path = self.label_paths[index].replace("\\",'/')
        label_image = Image.open(label_path).convert('RGB')
        label_image = self.Totensor_Image_label(label_image)#.to(self.device)

        #real_image 750*1101
        real_path = self.image_paths[index].replace("\\",'/')
        real_image = Image.open(real_path).convert('RGB')
        real_image = self.Totensor_Image_train(real_image)#.to(self.device)

        #label_seg 1024 * 1024
        seg = self.labelpath_to_segtensor(label_path)
        
        ref_label = 0
        ref_image = 0
        random_p = random.random()
        #random_p = 1.0
        if random_p < 0.7:
            key = real_path.replace('\\','/').split('deepfashionHD/')[-1]
            val = self.ref_dict[key] #val is a list
            if random_p < 0.3:
                #hard reference
                if len(val) == 1:
                    path_ref = val[0]
                else:
                    path_ref = val[1]
            else:
                #esay reference
                path_ref = val[0]
            path_ref = os.path.join(self.root,path_ref)
            if not os.path.exists(path_ref):
                path_ref = self.not_in_train_list(path_ref,is_img=True)
            ref_image = Image.open(path_ref).convert('RGB')
            #ref_image = self.Totensor_Image_train(ref_image)

            path_ref_label = self.imgpath_to_labelpath(path_ref)
            ref_label = Image.open(path_ref_label).convert('RGB')
            #ref_label = self.Totensor_Image_label(ref_label)

            if self.self_pair_flag[key] == True:
                self_ref_flag = 1.0
            else:
                self_ref_flag = 0.0

            
        else:
            pair =False
            key = real_path.replace('\\','/').split('deepfashionHD/')[-1]
            val = self.ref_dict[key] #val is a list
            ref_name = val[0]
            key_name = key
            path_ref = os.path.join(self.root,ref_name)
            if not os.path.exists(path_ref):
                path_ref = self.not_in_train_list(path_ref,is_img=True)
            ref_image = Image.open(path_ref).convert('RGB')

            path_ref_label = self.imgpath_to_labelpath(path_ref)
            ref_label = Image.open(path_ref_label).convert('RGB')
            if ref_name == key_name:
                ref_image = transforms.RandomHorizontalFlip(p=1.0)(ref_image)
                ref_label = transforms.RandomHorizontalFlip(p=1.0)(ref_label)
            pair = True
            
            if self.self_pair_flag[key] == True:
                self_ref_flag = 1.0
            else:
                self_ref_flag = 0.0

        ref_seg = self.labelpath_to_segtensor(path_ref_label)
        

        ref_label = self.Totensor_Image_label(ref_label)#.to(self.device)
        ref_image = self.Totensor_Image_train(ref_image)#.to(self.device)
        
        data = {
            'label': label_image,
            'image': real_image,
            'seg':seg,
            'self_ref': self_ref_flag,
            'ref': ref_image,
            'label_ref': ref_label,
            'seg_ref':ref_seg
        }
        

        return data
        pass

    def __len__(self):
        #return 2000
        return len(self.image_paths)
        pass

    def Totensor_Image_label(self,img):
        
        transforms_label = transforms.Compose([
            transforms.Resize(self.target_h,interpolation=Image.BICUBIC),
            transforms.Lambda(lambda img: self.__cropcenter(img)),
            transforms.Lambda(lambda img: self.__crop(img,self.crop_center,256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
        return transforms_label(img)
        pass
    
    def Totensor_Image_train(self,img):
        transforms_trains = transforms.Compose([
            transforms.Resize(128,interpolation=Image.BICUBIC),
            transforms.Lambda(lambda img:self.__crop(img,self.crop_center,128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
        return transforms_trains(img)
        pass

    def __crop(self,img,pos,size):
        y1= pos
        return img.crop((0,y1-64,128,y1 + 64))
        pass

    def __cropcenter(self,img,):
        return img.crop((int(self.target_h/2)-64, 0, int(self.target_h/2)+64, self.target_h))
        pass

    def not_in_train_list(self,path = '',is_img = True):
        if is_img:
            sourcepath = path.replace('dataset','data')
            sourcepath = sourcepath.replace('train_img','img')
            shutil.copyfile(sourcepath,path)
        return path



if __name__ == '__main__':
    test  = SelfDataset(root ="./dataset/deepfashionHD/")
    for i in range(10):
        label_image,real_image,ref_image,ref_label = test[i]
        toPIL = transforms.ToPILImage()
        l = label_image*0.5 + 0.5
        t = real_image*0.5 + 0.5
        r = ref_image*0.5 + 0.5
        z = ref_label*0.5 + 0.5
        pic = toPIL(l)
        pic.save('test1.jpg')
        pic = toPIL(t)
        pic.save('test2.jpg')
        pic = toPIL(r)
        pic.save('test3.jpg')
        pic = toPIL(z)
        pic.save('test4.jpg')
    pass