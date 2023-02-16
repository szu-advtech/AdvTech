import torch
from torch.autograd import Variable
import numpy as np

train_image_path = "./data/train/img"
train_label_path = "./data/train/label"

#train_image_path = "./Duke_PV/unet/data/train/img"
#train_label_path = "./Duke_PV/unet/data/train/label"
validation_image_path = "./Duke_PV/unet/data/val/img"
validation_label_path = "./Duke_PV/unet/data/val/label"

save_img_path = "./result/save_train_img"
#save_img_path = "./Duke_PV/unet/result/save_train_img"
save_val_img_path = "./result/save_val_img"
save_test_img_path = "./result/save_test_img"
test_out_path = './result/test_out'
test_label_path2 = './result/test_label'

weight_path = './params/unet.pth'
#weight_path = './Duke_PV/unet/params/unet.pth'
trn_loss_path = './evaluation/trn/trn_loss.txt'
#trn_loss_path = './Duke_PV/unet/evaluation/trn/trn_loss.txt'
trn_acc_path = './evaluation/trn/trn_acc.txt'
val_loss_path = './evaluation/val/val_loss.txt'
val_acc_path = './evaluation/val/val_acc.txt'
test_loss_path = './evaluation/test/test_loss.txt'
test_acc_path = './evaluation/test/test_acc.txt'

test_img_path = './data/test/img'
test_label_path = './data/test/label'

#  批大小
batch_size = 8
#  类的数目(包括背景)
classNum = 2
#  模型输入图像大小
input_size = (200, 200, 3)
#  训练模型的迭代总轮数
epochs = 200  # 100
#  初始学习率
learning_rate = 1e-3
#  预训练模型地址
premodel_path = None
#  训练模型保存地址
model_path = "C:\\Users\\James\\Desktop\\project\\model\\unet_model.hdf5"


def Probality2OneTwo(a):
    zero = torch.zeros_like(a)
    one = torch.ones_like(a)
    a = torch.where(a > 0.5, one, zero)
    # 默认张量是requires_grad=flase,设置为true，这样训练时不会报错
    return Variable(a, requires_grad=True) 
    

def printTensor2Txt(t, name):
    with open('./data/txt/' + name + '.txt', 'a') as f:
        torch.set_printoptions(profile="full")
        np.set_printoptions(threshold=np.inf)
        
        f.write(name + ' == ')
        # Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
        for i in range(len(t.detach().numpy())):
            f.write(str(t.detach().numpy()[i])+'\n')
