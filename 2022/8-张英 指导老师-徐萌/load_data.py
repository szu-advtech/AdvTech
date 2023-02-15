from torch.utils.data import DataLoader, Dataset
import torch
import h5py
import numpy as np
import glob

# path
im_train_list_x = glob.glob(r"./data_2/cloud_image/*.mat")    # 拿到所有训练集的图像
im_train_list_y = glob.glob(r"./data_2/clear_image/*.mat")     # 对应的清晰图片
# print(os.path.exists(r"./data/cloud"))
print(len(im_train_list_x))
# im_test_list_x = glob.glob(r"./data/cloud_image_valid/*.mat")  # 拿到所有测试集的图像
# im_test_list_y = glob.glob(r"./data/clear_image_valid/*.mat")   # 对应的清晰图片


def default_loader(path):
    # print(path)
    return h5py.File(path)


class MyDataset(Dataset):
    def __init__(self, im_list_x, im_list_y, loader=default_loader):
        super(MyDataset, self).__init__()
        self.loader = loader
        self.imlist_x = im_list_x
        self.imlist_Y = im_list_y

    def __getitem__(self, index):

        im_data = self.loader(self.imlist_x[index])
        im_data = im_data['tamp']
        im_data = np.array(im_data)

        im_data = torch.Tensor(im_data)

        im_label = self.loader(self.imlist_Y[index])
        im_label = im_label['tamp']
        im_label = np.array(im_label)
        im_label = torch.Tensor(im_label)

        return im_data, im_label

    def __len__(self):
        """返回样本总数"""
        return len(self.imlist_x)


train_dataset = MyDataset(im_train_list_x, im_train_list_y)
train_data_loader = DataLoader(dataset=train_dataset,
                               batch_size=8,
                               shuffle=False,
                               num_workers=0)

# test_dataset = MyDataset(im_test_list_x, im_test_list_y)
# test_data_loader = DataLoader(dataset=test_dataset,
#                                batch_size=8,
#                                shuffle=False,
#                                num_workers=0)




# # 测试dataset
# if __name__ == '__main__':
#     dataset = MyDataset(im_train_list_x, im_train_list_y)
#     x, y = dataset.__getitem__(0)
#     print(x.shape)
#     print(dataset.__len__())
#
#     print(im_train_list_x)
