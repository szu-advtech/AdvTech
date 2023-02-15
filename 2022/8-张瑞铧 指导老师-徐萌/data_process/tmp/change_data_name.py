import os

# projectPath = E:\\data
projectPath = os.getcwd()

# rootpath = C:\Users\Administrator\Desktop\pythonProject\ROIs1868_summer
rootpath = projectPath + "\\ROIs2017_winter"


# root i = C:\Users\Administrator\Desktop\pythonProject\ROIs1868_summer\si
for i in os.listdir(rootpath):
    root = os.path.join(rootpath, i)
    root_content = os.listdir(root)

    for old_name in root_content:
        if i == 's1':
            tmp = old_name.split('_s1')

        if i == 's2_cloudFree':
            tmp = old_name.split('_s2')

        if i == 's2_cloudy':
            tmp = old_name.split('_s2_cloudy')

        new_name = ''.join(tmp)
        os.rename(os.path.join(root, old_name), os.path.join(root, new_name))


print('done')