"""
Definition of PyTorch "Dataset" that iterates through compressed videos
and return compressed representations (I-frames, motion vectors,
or residuals) for training or testing.
"""

import os
import os.path
import random
import cv2
import numpy as np
import torch
import torch.utils.data as data

# from coviar import get_num_frames
# from coviar import load
from transforms import color_aug
import matplotlib.pyplot as plt

GOP_SIZE = 12

num = 0

from PIL import Image

people_path="/data/jlzhang/Py_protect/2022-08-05/pytorch-coviar-master/data/MMI-align-image-256/"


def clip_and_scale(img, size):
    return (img * (127.5 / size)).astype(np.int32)


######帧数 ，分段数量，分割数，帧类型
def get_seg_range(n, num_segments, seg, representation):
    # if representation in ['residual', 'mv']:
    #     n -= 1

    seg_size = float(n - 1) / num_segments
    seg_begin = int(np.round(seg_size * seg))
    seg_end = int(np.round(seg_size * (seg + 1)))

    ##自己添加的

    # if seg == 2:
    #     seg_begin = int(np.round(seg_size // 2))
    #     seg_end = int(np.round(seg_size + seg_begin))

    ##自己添加的
    # if seg_end >(n-1):
    #     seg_end=(n-1)

    if seg_end == seg_begin:
        seg_end = seg_begin + 1
        if seg_end > (n - 1):
            seg_end = seg_end - 1
            seg_begin = seg_begin - 1
    # if seg_end >(n-1):
    #     seg_end=(n-1)
    #     seg_begin=seg_end-1

    # if representation in ['residual', 'mv']:
    #     # Exclude the 0-th frame, because it's an I-frmae.
    #     return seg_begin + 1, seg_end + 1

    return seg_begin, seg_end


# 修改过后的
def get_gop_pos(frame_idx, representation, gop):
    # gop_index = frame_idx // GOP_SIZE
    gop_index = frame_idx

    # gop_index = gop
    gop_pos = frame_idx % GOP_SIZE
    if representation in ['residual', 'mv']:
        if gop_pos == 0:
            gop_pos = GOP_SIZE - 1
        # if gop_index==0 :
        #     gop_index=gop_index+1

        # gop_pos = random.randint(1, 11)
        # gop_pos = GOP_SIZE - 1

        ###原来的
        # if gop_pos == 0:
        #     gop_index -= 1
        #     gop_pos = GOP_SIZE - 1
        ###原来的
    # else:
    #     gop_pos = 0
    return gop_index, gop_pos


# 修改过后的

# def get_gop_pos(frame_idx, representation):
#     gop_index = frame_idx // GOP_SIZE
#
#     # gop_index = gop
#     gop_pos = frame_idx % GOP_SIZE
#     if representation in ['residual', 'mv']:
#         ###原来的
#         if gop_pos == 0:
#             gop_index -= 1
#             gop_pos = GOP_SIZE - 1
#         ###
#     else:
#         gop_pos = 0
#     return gop_index, gop_pos

class CoviarDataSet(data.Dataset):
    def __init__(self, data_root, data_name,
                 video_list,
                 representation,
                 transform,
                 num_segments,
                 is_train,
                 accumulate):

        self._data_root = data_root
        self._data_name = data_name
        self._num_segments = num_segments
        self._representation = representation
        self._transform = transform
        self._is_train = is_train
        self._accumulate = accumulate

        self._input_mean = torch.from_numpy(
            np.array([0.485, 0.456, 0.406]).reshape(
                (1, 3, 1, 1))).float()  ##就是torch.from_numpy()方法把数组转换成张量，且二者共享内存，对张量进行修改比如重新赋值，那么原始数组也会相应发生改变。
        self._input_std = torch.from_numpy(
            np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))).float()

        self._load_list(video_list)

    ##加载视频列表了
    def _load_list(self, video_list):
        self._video_list = []
        with open(video_list, 'r+') as f:
            for line in f.readlines():
                # print(line)
                video, video_num_frames, label = line.strip().split()
                # video_path = os.path.join(self._data_root, video[:-4] + '.mp4')
                video_path = os.path.join(self._data_root, video[:-4])
                self._video_list.append((
                    video_path,
                    int(label),
                    int(video_num_frames)))  # 获得视频的路径、视频的label以及视频的帧数

        print('%d videos loaded.' % len(self._video_list))

    ##修改过后的
    def _get_train_frame_index(self, num_frames, seg):
        # Compute the range of the segment.
        seg_begin, seg_end = get_seg_range(num_frames, self._num_segments, seg,
                                           representation=self._representation)

        # print("seg_begin==",seg_begin)
        # print("seg_end==",seg_end)

        # Sample one frame from the segment.
        v_frame_idx = random.randint(seg_begin, seg_end)
        gop_num = num_frames // GOP_SIZE
        gop_ex = num_frames % GOP_SIZE

        #
        # v_frame_idx = num_frames // 2
        # if seg == 0:
        #     v_frame_idx=v_frame_idx-4
        #
        # elif seg == 1:
        #     v_frame_idx=v_frame_idx-3
        #
        # elif seg == 2:
        #     v_frame_idx=v_frame_idx-2
        # elif seg == 3:
        #     v_frame_idx=v_frame_idx-1
        #
        # elif seg == 4:
        #     v_frame_idx=v_frame_idx
        #
        # elif seg == 5:
        #     v_frame_idx=v_frame_idx+1
        #
        # elif seg == 6:
        #     v_frame_idx=v_frame_idx+2
        #
        # elif seg == 7:
        #     v_frame_idx=v_frame_idx+3

        # if gop_ex!=0:
        #     gop_num=gop_num+1

        # gop_op2=gop_num//2

        gop_op = gop_num // 2
        # if seg == 0:
        #     gop_op=0
        #
        # elif seg == 1:
        #     gop_op=1
        #
        # elif seg==2:
        #     gop_op=2
        #
        # elif seg == 3:
        #     gop_op=3
        #     if gop_op >gop_num:
        #         gop_op=gop_num
        #
        # elif seg == 4:
        #     gop_op=4
        #     if gop_op >gop_num:
        #         gop_op=gop_num

        if gop_op <= 1:
            gop_op = gop_op + 1

        if seg == 0:
            gop_op = gop_op - 2
        elif seg == 1:
            gop_op = gop_op - 1

        elif seg == 2:
            gop_op = gop_op

        elif seg == 3:
            gop_op = gop_op + 1
            if gop_op > (gop_num - 1):
                gop_op = gop_num

        elif seg == 4:
            gop_op = gop_op + 2
            if gop_op == gop_num and gop_ex == 0:
                gop_op = gop_op - 1
            if gop_op > gop_num and gop_ex != 0:
                gop_op = gop_num

        # if gop_op <= 1:
        #     gop_op = gop_op + 1
        #
        # if seg == 4:
        #     gop_op = gop_op - 2
        # elif seg == 0:
        #     gop_op = gop_op - 1
        #
        # elif seg == 1:
        #     gop_op = gop_op
        #
        # elif seg == 2:
        #     gop_op=gop_op+1
        #     if gop_op>(gop_num-1)   :
        #         gop_op=gop_num
        #
        # elif seg == 3:
        #     gop_op=gop_op+2
        #     if gop_op==gop_num and gop_ex ==0 :
        #         gop_op=gop_op-1
        #     if gop_op >gop_num and gop_ex !=0:
        #         gop_op = gop_num
        #
        #

        return get_gop_pos(v_frame_idx, self._representation, gop_op)

    ##修改过后的

    # def _get_train_frame_index(self, num_frames, seg):
    #     # Compute the range of the segment.
    #     seg_begin, seg_end = get_seg_range(num_frames, self._num_segments, seg,
    #                                              representation=self._representation)
    #
    #     # Sample one frame from the segment.
    #     v_frame_idx = random.randint(seg_begin, seg_end - 1)
    #     gop_num=num_frames//GOP_SIZE
    #     # gop_op2=gop_num//2
    #     gop_op = gop_num // 2
    #     # gop_op=gop_op//2
    #     if gop_op<=1:
    #         gop_op=gop_op+1
    #
    #     if seg==0:
    #         gop_op=gop_op
    #     elif seg==1:
    #         gop_op=gop_op-1
    #
    #     elif seg==2:
    #         gop_op = gop_op-2
    #     elif seg == 3:
    #         gop_op = 0
    #
    #     # if gop_op<0 or gop_op>gop_num:
    #     #     gop_op=gop_op2
    #
    #     return get_gop_pos(v_frame_idx, self._representation,gop_op)

    ####修改过后的

    def _get_test_frame_index(self, num_frames, seg):
        if self._representation in ['mv', 'residual']:
            num_frames -= 1

        # seg_size = float(num_frames - 1) / self._num_segments
        # v_frame_idx = int(np.round(seg_size * (seg + 0.5)))

        seg_begin, seg_end = get_seg_range(num_frames, self._num_segments, seg,
                                           representation=self._representation)

        # Sample one frame from the segment.

        # print("seg_begin==",seg_begin)
        # print("seg_begin==", seg_end)

        v_frame_idx = random.randint(seg_begin, seg_end)

        gop_num = num_frames // GOP_SIZE

        gop_op = gop_num // 2
        gop_ex = num_frames % GOP_SIZE

        # v_frame_idx = num_frames // 2
        # if seg == 0:
        #     v_frame_idx = v_frame_idx - 7
        #
        # elif seg == 1:
        #     v_frame_idx = v_frame_idx - 6
        #
        # elif seg == 2:
        #     v_frame_idx = v_frame_idx - 5
        # elif seg == 3:
        #     v_frame_idx = v_frame_idx - 4
        #
        # elif seg == 4:
        #     v_frame_idx = v_frame_idx - 3
        #
        # elif seg == 5:
        #     v_frame_idx = v_frame_idx - 2
        #
        # elif seg == 6:
        #     v_frame_idx = v_frame_idx - 1
        #
        # elif seg == 7:
        #     v_frame_idx = v_frame_idx
        #
        # elif seg == 8:
        #     v_frame_idx = v_frame_idx +1
        #
        # elif seg == 9:
        #     v_frame_idx = v_frame_idx + 2
        # elif seg == 10:
        #     v_frame_idx = v_frame_idx +3
        #
        # elif seg == 11:
        #     v_frame_idx = v_frame_idx +4
        #
        # elif seg == 12:
        #     v_frame_idx = v_frame_idx + 5
        #
        # elif seg == 13:
        #     v_frame_idx = v_frame_idx + 6
        #
        # elif seg == 14:
        #     v_frame_idx = v_frame_idx + 7
        # elif seg == 15:
        #     v_frame_idx = v_frame_idx + 8

        # if gop_op <= 1:
        #     gop_op = gop_op + 1
        #
        # if seg == 0:
        #     gop_op = gop_op
        # elif seg == 1:
        #     gop_op = gop_op - 1
        #
        # elif seg == 2:
        #     gop_op = gop_op - 2

        #

        if gop_op <= 1:
            gop_op = gop_op + 1

        if seg == 0:
            gop_op = gop_op - 2
        elif seg == 1:
            gop_op = gop_op - 1

        elif seg == 2:
            gop_op = gop_op

        elif seg == 3:
            gop_op = gop_op + 1
            if gop_op > (gop_num - 1):
                gop_op = gop_num

        elif seg == 4:
            gop_op = gop_op + 2
            if gop_op == gop_num and gop_ex == 0:
                gop_op = gop_op - 1
            if gop_op > gop_num and gop_ex != 0:
                gop_op = gop_num

        #
        # if gop_op <= 1:
        #     gop_op = gop_op + 1
        #
        # if seg == 4:
        #     gop_op = gop_op - 2
        # elif seg == 0:
        #     gop_op = gop_op - 1
        #
        # elif seg == 1:
        #     gop_op = gop_op
        #
        # elif seg == 2:
        #     gop_op=gop_op+1
        #     if gop_op>(gop_num-1)   :
        #         gop_op=gop_num
        #
        # elif seg == 3:
        #     gop_op=gop_op+2
        #     if gop_op==gop_num and gop_ex ==0 :
        #         gop_op=gop_op-1
        #     if gop_op >gop_num and gop_ex !=0:
        #         gop_op = gop_num
        #
        #
        if self._representation in ['mv', 'residual']:
            v_frame_idx += 1

        return get_gop_pos(v_frame_idx, self._representation, gop_op)

    ####修改过后的

    # def _get_test_frame_index(self, num_frames, seg):
    #     if self._representation in ['mv', 'residual']:
    #         num_frames -= 1
    #     ##自己添加的
    #     num_frames=num_frames//2
    #     ##自己添加的
    #     seg_size = float(num_frames - 1) / self._num_segments
    #     v_frame_idx = int(np.round(seg_size * (seg + 0.5)))
    #
    #
    #
    #     # seg_begin, seg_end = get_seg_range(num_frames, self._num_segments, seg,
    #     #                                          representation=self._representation)
    #
    #     # Sample one frame from the segment.
    #     # v_frame_idx = random.randint(seg_begin, seg_end - 1)
    #
    #
    #     if self._representation in ['mv', 'residual']:
    #         v_frame_idx += 1
    #
    #     return get_gop_pos(v_frame_idx, self._representation)

    def __getitem__(self, index):

        if self._representation == 'mv':
            representation_idx = 1
        elif self._representation == 'residual':
            representation_idx = 2
        else:
            representation_idx = 0

        if self._is_train:
            video_path, label, num_frames = random.choice(self._video_list)
            # video_path, label, num_frames = self._video_list[index]
        else:
            video_path, label, num_frames = self._video_list[index]

        im_list = os.listdir(video_path)

        # video_list_1=video_path.split('/')[-1] ###CK+

        # im_list.sort(key=lambda x: int(x.replace(f"{video_list_1}_", "").split('.')[0]))###CK+
        # im_list.sort(key=lambda x: int(x.replace("_frame_", "").split('.')[0]))####AFEW
        im_list.sort(key=lambda x: int(x.replace("frame", "").split('.')[0]))####MMI
        # print(video_path)
        # im_list.sort(key=lambda x: int(x.replace("FirstF_", "").split('.')[0]))  ###oulua

        frames = []
        den_frames = []

        for seg in range(self._num_segments):

            if self._is_train:
                gop_index, gop_pos = self._get_train_frame_index(num_frames, seg)  ###得到视频的帧数
            else:
                gop_index, gop_pos = self._get_test_frame_index(num_frames, seg)

            # print("video_path==", video_path)
            # print("gop_index==", gop_index)
            # print("gop_pos==", gop_pos)
            # print(video_path)
            #
            # img = load(video_path, gop_index, gop_pos,
            #
            #            representation_idx, self._accumulate)

            # img = cv2.imread(sig_image_path)
            # img = load(video_path, gop_index, gop_pos,
            #
            #            representation_idx, self._accumulate)
            if gop_index >=len(im_list):
                print(video_path)
                print(gop_index)

            sig_name = im_list[gop_index]
            sig_image_path = os.path.join(video_path, sig_name)

            img = cv2.imread(sig_image_path)
            # img2=img.astype(np.uint8)
            # img2 = Image.fromarray(img2)
            # # im.save('{}/{}.jpg'.format(each_img_path, num))
            #
            # plt.imshow(img2)
            # plt.show()
            if seg==0:
                video_path_list=video_path.split('/')
                label_name=video_path_list[-2]
                each_video_name=video_path_list[-1]
                den_path=os.path.join(people_path,label_name)
                den_path=os.path.join(den_path,each_video_name)
                den_image_list=os.listdir(den_path)
                den_image_list.sort(key=lambda x: int(x.replace("frame", "").split('.')[0]))  ##MMI
                den_each_path=os.path.join(den_path,den_image_list[0])
                den_iamge = cv2.imread(den_each_path)
                den_iamge = den_iamge[..., ::-1]
                # den_iamge2 = den_iamge.astype(np.uint8)
                # den_iamge2= Image.fromarray(den_iamge2)
                # # im.save('{}/{}.jpg'.format(each_img_path, num))
                #
                # plt.imshow(den_iamge2)
                # plt.show()
                den_iamge=cv2.resize(den_iamge, (224,224),
                           cv2.INTER_LINEAR)
                den_frames.append(den_iamge)

            # if seg == 2:
            #     img = load(video_path, gop_index, gop_pos,
            #
            #                representation_idx, self._accumulate)
            # else:
            #     repre=2
            #     gop_pos2=GOP_SIZE-1
            #     img = load(video_path, gop_index, gop_pos2,
            #
            #                repre, self._accumulate)

            # img = load(video_path, 0,1,
            #            representation_idx, self._accumulate)

            if img is None:
                print('Error: loading video %s failed.' % video_path)
                img = np.zeros((256, 256, 2)) if self._representation == 'mv' else np.zeros((256, 256, 3))
            else:

                if self._representation == 'mv':
                    img = clip_and_scale(img, 20)
                    img += 128
                    img = (np.minimum(np.maximum(img, 0), 255)).astype(np.uint8)
                # elif self._representation == 'residual':
                #     img += 128
                #     img = (np.minimum(np.maximum(img, 0), 255)).astype(np.uint8)

                # elif seg != 2 :
                #     img += 128
                #     img = (np.minimum(np.maximum(img, 0), 255)).astype(np.uint8)

                #
                if self._representation == 'iframe' and self._is_train:
                    img = color_aug(img)

                    # img2 = img.astype(np.uint8)
                    # img2 = Image.fromarray(img2)

                    # im.save('{}/{}.jpg'.format(each_img_path, num))

                    # plt.imshow(img2)
                    # plt.show()

                # 在此之前一直都是BGR

                # BGR to RGB. (PyTorch uses RGB according to doc.)
                    img = img[..., ::-1]
                    # img2 = img.astype(np.uint8)
                    # img2 = Image.fromarray(img2)
                    # im.save('{}/{}.jpg'.format(each_img_path, num))

                    # plt.imshow(img2)
                    # plt.show()

            frames.append(img)


        frames = self._transform(frames)

        # for img in den_frames:
        #     global num
        #     num=num+1
        #     img=img.astype(np.uint8)
        #     img = Image.fromarray(img)
        #     # im.save('{}/{}.jpg'.format(each_img_path, num))
        #
        #     plt.imshow(img)
        #     plt.show()

        frames = np.array(frames)
        frames = np.transpose(frames, (0, 3, 1, 2))
        input = torch.from_numpy(frames).float() / 255.0

        den_frames = np.array(den_frames)
        den_frames = np.transpose(den_frames, (0, 3, 1, 2))
        den_input = torch.from_numpy(den_frames).float() / 255.0
        den_input = (den_input - self._input_mean) / self._input_std

        if self._representation == 'iframe':
            input = (input - self._input_mean) / self._input_std
        elif self._representation == 'residual':
            input = (input - 0.5) / self._input_std
        elif self._representation == 'mv':
            input = (input - 0.5)
        return input, label,den_input

    def __len__(self):
        return len(self._video_list)
