import os
import sys  # 导入模块


def add_prefix_files():  # 定义函数名称
    mark = '00'  # 准备添加的前缀内容
    old_names = os.listdir(path)  # 取路径下的文件名，生成列表
    for i in range(len(old_names)):  # 遍历列表下的文件名
        #if old_names[i] != sys.argv[0]:  # 代码本身文件路径，防止脚本文件放在path路径下时，被一起重命名
            #if old_name.endswith('.txt'):  # 当文件名以.txt后缀结尾时
        os.rename(os.path.join(path, old_names[i]),
                    os.path.join(path, f'00{i}.tif'))  # 重命名文件
        # 输出提示
        print(
            old_names[i], "has been renamed successfully! New name is: ", f'00{i}.tif')


if __name__ == '__main__':
    path = 'C:\\Users\\James\\Desktop\\train\\label'  # 运行程序前，记得修改主文件夹路径！
    add_prefix_files()  # 调用定义的函数，注意名称与定义的函数名一致
