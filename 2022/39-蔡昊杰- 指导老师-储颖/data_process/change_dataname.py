import os

train_image_path = "C:\\Users\\James\\Desktop\\project\\data\\train\\img"
train_label_path = "C:\\Users\\James\\Desktop\\project\\data\\train\\label"
validation_image_path = "C:\\Users\\James\\Desktop\\project\\data\\validation\\img"
validation_label_path = "C:\\Users\\James\\Desktop\\project\\data\\validation\\label"

#获取该目录下所有文件，存入列表中
trn_List=os.listdir(train_image_path)
label_List=os.listdir(train_label_path)

n=0

for i in trn_List:
    
    #设置旧文件名（就是路径+文件名）
    oldname=train_image_path+ os.sep + trn_List[n]   # os.sep添加系统分隔符
    
    #设置新文件名
    newname = train_image_path + os.sep + '000'+str(n+1)+'.tif'
    #if oldname != newname:
    os.rename(oldname, newname)   #用os模块中的rename方法对文件改名
    print(oldname,'======>',newname)
    
    n += 1

n=0
for i in label_List:

    #设置旧文件名（就是路径+文件名）
    oldname = train_label_path + os.sep + label_List[n]   # os.sep添加系统分隔符

    #设置新文件名
    newname = train_label_path + os.sep + '000'+str(n+1)+'.tif'
    #if oldname != newname:
    os.rename(oldname, newname)  # 用os模块中的rename方法对文件改名
    print(oldname, '======>', newname)

    n += 1
