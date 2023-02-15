import os

IMG_EXTENSIONS=[
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff', '.webp', '.npy'
]
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset_rec(dir,images):
    assert os.path.isdir(dir),'%s is not a valid direction' %dir
    for root,dnames,fnames in sorted(os.walk(dir,followlinks=True)):
        for fname in fnames:
            if is_image_file(fname):
                path=os.path.join(root,fname)
                images.append(path)

def make_dataset(dir,recursive=False,read_cache=False,write_cache=False):
    """ 获取dir目录下的所有图像路径"""
    images=[]
    if read_cache:
        # os.path.join()连接两个或更多的路径名组件与'str1+str2'类似，区别如下：
        # 如果各组件名首字母不包含’\’，则函数会自动加上；
        # 如果有一个组件是一个绝对路径，则在它之前的所有组件均会被舍弃
        # 如果最后一个组件为空，则生成的路径以一个’\’分隔符结尾
        # 参考博客：https://blog.csdn.net/weixin_41644993/article/details/96842032#
        possible_filelist=os.path.join(dir,'files.list')
        # os.path.isfile()：判断某一对象(需提供绝对路径)是否为文件
        # os.path.isdir()：判断某一对象(需提供绝对路径)是否为目录
        # 可使用os.path.join()函数来创建绝对路径#
        if os.path.isfile(possible_filelist):
            with open(possible_filelist,'r') as f:
                # f.read().splitlines()读取文本文件的行数据，返回字符串列表（每一个元素代表一行数据）
                # 参考博客：https://wenku.baidu.com/view/aa88d9e28aeb172ded630b1c59eef8c75fbf9596.html#
                images=f.read().splitlines()
                return images
    if recursive:
        make_dataset_rec(dir,images)
    else:
        # os.path.islink()方法用于判断路径是否为链接（即快捷方式）#
        assert os.path.isdir(dir) or os.path.islink(dir),'%s is not a valid directory'%dir
        # sorted(): Return a new list containing all items from the iterable in ascending order
        # os.walk() 方法是一个简单易用的文件、目录遍历器。输出在目录中的文件名#
        for root,dnames,fnames in sorted(os.walk(dir)):
            # root 表示当前正在访问的文件夹路径
            # dnames 表示该文件夹下的子目录名list
            # fnames 表示该文件夹下的文件list

            for fname in fnames:
                if is_image_file(fname):
                    path=os.path.join(root,fname)
                    images.append(path)
    if write_cache:
        filelist_cache=os.path.join(dir,'files.list')
        with open(filelist_cache,'w') as f:
            for path in images:
                f.write("%s\n"%path)
            print("wrote filelist cache at %s"%filelist_cache)
    return images