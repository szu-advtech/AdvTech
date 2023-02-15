"""import os
import shutil


def move_file(root):
    # root = "E:\\mini data\\ROIs1158_spring"
    for subRoot in os.listdir(root):
        path = f'{root}/{subRoot}'
        folders = [os.path.join(path, f) for f in os.listdir(path) if not '.' in f]
        for fdsrc in folders:
            srcfiles = [os.path.join(fdsrc, f) for f in os.listdir(fdsrc)]
            for srcf in srcfiles:
                shutil.move(srcf, os.path.join(path, os.path.split(srcf)[1]))


if __name__ == "__main__":
    data_path = "E:\\mini data"
    folders = [os.path.join(data_path, f) for f in os.listdir(data_path) if not '.' in f]
    for folder in folders:
        move_file(folder)
"""

import os
import shutil

roots = ["E:\\mini data\\ROIs1158_spring","E:\\mini data\\ROIs1868_summer","E:\\mini data\\ROIs1970_fall","E:\\mini data\\ROIs2017_winter"]

for root in roots:
    # root = E:\\mini data\\ROIs1158_spring

    for subRoot in os.listdir(root):
        # subRoot = ROIs1158_spring_s1, ROIs1158_spring_s2, ROIs1158_spring_s2_cloudy

        # path = E:\mini data\ROIs1158_spring\ROIs1158_spring_s1
        path = os.path.join(root, subRoot)

        for folders in os.listdir(path):
            shutil.move(os.path.join(path, folders), root)
