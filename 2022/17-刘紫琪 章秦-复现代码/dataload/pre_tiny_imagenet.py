import os
import shutil
#因为我做分类，tinyimagenet一共有200个分类，val中每个分类有50张，所以先创建存放图片的文件夹
def mkdir(path):
    folder=os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        print(path)

if __name__=='__main__':
    #tinyimagenet提供了200分类的txt，winds.txt，根据它创建
    # file='../data/tiny-imagenet-200/val'
    # with open('../data/tiny-imagenet-200/wnids.txt', 'r') as w:
    #     for line in w.readlines():
    #         line = line.strip('\n')
    #         folder=file+'/'+line
    #         mkdir(folder)

    with open("../data/tiny-imagenet-200/val/val_annotations.txt", 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            dirlist = []
            imagelist = []
            dir = line.split()
            dir_name = dir[1:2]
            image_name = dir[0:1]
            dirlist.append(dir_name)
            imagelist.append(image_name)
            a = dirlist[0][0]
            b = imagelist[0][0]
            image_path = '../data/tiny-imagenet-200/val/images' + '/' + b
            dir_path = '../data/tiny-imagenet-200/val' + '/' + a
            shutil.copy(image_path, dir_path)