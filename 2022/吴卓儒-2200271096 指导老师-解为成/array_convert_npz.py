import numpy as np
from numpy import asarray,savez_compressed #asarray转化成数组 savez_compressed将多个数组保存到一个压缩文件中.npz格式
import tqdm
import argparse
from PIL import Image
#np.set_printoptions(threshold=np.inf)

def convert_npz(imgpath,maskpath, size=(512,512),crops=50,n_images=17):
    src_list, tar_list = list(), list()
    for i in tqdm.tqdm(range(n_images)):
        for j in range(crops):
            # load and resize the image
            filename = str(i+1)+"_"+str(j+1)+".png"
            mask_name = str(i+1)+"_mask_" + str(j+1)+".png"
            img = Image.open(imgpath + filename)
            #print(type(img))
            #print(img.size)
            #print(img.mode) RGB 3通道 真色彩
            fundus_img = np.array(img)
            #print(type(fundus_img))
            #print(fundus_img.shape)

            mask = Image.open(maskpath + mask_name)
            #print(type(mask))
            #print(mask.size)
            #print(mask.mode) L 灰度图 像素8位
            angio_img = np.array(mask)
            #print(type(angio_img))
            #print(angio_img.shape)
            angio_img = np.reshape(angio_img,(angio_img.shape[0],angio_img.shape[1],1))
            #print(angio_img.shape)

            # split into satellite and map
            src_list.append(fundus_img)
            tar_list.append(angio_img)
    return [asarray(src_list), asarray(tar_list)]

if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--input_dim', type=int, default=512)
    # parser.add_argument('--n_crops', type=int, default=50)
    # parser.add_argument('--datadir', type=str, required=True, help='path/to/data_directory',default='data')
    # parser.add_argument('--outfile_name', type=str, default='fun2angio')
    # parser.add_argument('--n_images', type=int, default=17)
    # args = parser.parse_args()

    # dataset path
    imgpath = 'data/Images/'#args.datadir+'/Images/'
    maskpath = 'data/Masks/' #args.datadir+'/Masks/'
    # load dataset
    [src_images, tar_images] = convert_npz(imgpath, maskpath, size=(512, 512), crops=50, n_images=17)
    print('Loaded: ', src_images.shape, tar_images.shape)
    # print(src_images[790][200][200][2])
    # print(tar_images[790][200][200][0])
    # # save as compressed numpy array
    src_images = np.transpose(src_images, (0, 3, 1, 2))
    tar_images = np.transpose(tar_images, (0, 3, 1, 2))
    print('Loaded: ', src_images.shape, tar_images.shape)
    filename = 'fun2angio_new_trans.npz'  # args.outfile_name+'.npz'
    savez_compressed(filename, src_images, tar_images)
    print('Saved dataset: ', filename)