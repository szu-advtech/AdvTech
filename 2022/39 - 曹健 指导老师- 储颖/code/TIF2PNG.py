import os
import cv2

# TIF_img_dir为输入TIF文件夹
# PNG_img_dir为输出PNG文件夹
def Tif_To_Png(TIF_img_dir, PNG_img_dir):
    # 创建输出目录
    if os.path.exists(PNG_img_dir):
        pass
    else:
        os.mkdir(PNG_img_dir)
    TIF_names = os.listdir(TIF_img_dir)
    for name in TIF_names:
        absolute_path = TIF_img_dir + '/' + name
        TIF_image = cv2.imread(absolute_path)
        # print(PNG_img_dir + '/' + name[:-3] + 'png')
        # print(TIF_image)
        cv2.imwrite(PNG_img_dir + '/' + name[:-3] + 'png', TIF_image)

if __name__ == '__main__':
    # 需要更改的文件夹路径
    TIF_dir = '/tmp/pycharm_project_17/data/test_mask'
    PNG_dir = '/tmp/pycharm_project_17/data/test_mask'

    Tif_To_Png(TIF_img_dir=TIF_dir, PNG_img_dir=PNG_dir)
