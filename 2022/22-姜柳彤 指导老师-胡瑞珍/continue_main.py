import torch.optim
from torchinfo import summary
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import material_decomposition_resnetUnet as mdr
import dataset
import os
import json
import utils
import numpy as np
import random
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from unet import unet_model

# model test codes
# model = mdu.UNet()
# summary(model, input_size=(1, 1, 572, 572))
# model = rsn.resnet34()
# summary(model, input_size=(1, 3, 224, 224))
# model = mdr.res34_unet()
# summary(model, input_size=(8, 4, 256, 256))
# ---------------------------------------------------------------------------------------------------------------------
os.environ["CUDA_DEVICEC_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # 为了确定算法，保证得到一样的结果。
    torch.backends.cudnn.enabled = True  # 使用非确定性算法
    torch.backends.cudnn.benchmark = True  # 是否自动加速。


# 设置随机数种子
# setup_seed(20)


def train(data_loader, epoch, test_dataLoader, save_path, lr=1e-3):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # choose device
    # device_ids = [0, 1]
    # model = mdr.res34_unet()  # define model
    model = unet_model.UNet(4)
    model.to(device)
    model_dict = torch.load('save_models/checkpoints_7.pkl')
    model.load_state_dict(model_dict['state_dict'])
    # model = torch.nn.DataParallel(model, device_ids=device_ids)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)  # define optimizer function
    optimizer.load_state_dict(model_dict['optimizer'])
    start_epoch = model_dict['epoch']

    loss_function = torch.nn.MSELoss()  # define loss function
    loss_train = []  # record loss
    loss_test = []
    best_train_loss = 10000
    best_test_loss = 10000

    for i in range(start_epoch, epoch):
        loss = 0
        # train
        for step, (imgs_batch, mask_batch, label_base_color, label_roughness, label_metallic, label_normal) \
                in enumerate(data_loader):
            imgs_batch = imgs_batch.to(device)
            mask_batch = mask_batch.to(device)

            label_base_color = label_base_color.to(device) * mask_batch
            label_roughness = label_roughness.to(device) * mask_batch
            label_metallic = label_metallic.to(device) * mask_batch
            label_normal = label_normal.to(device) * mask_batch
            x = torch.cat([imgs_batch, mask_batch], dim=1)

            y_pred = model(x)
            y_pred = y_pred * mask_batch
            loss = loss_function(y_pred, torch.cat([label_base_color, label_roughness,
                                                    label_metallic, label_normal], dim=1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print("Epoch:{}, batch:{}, train_Loss:{:.8f}".format(i, step, loss))
                with open('loss_train.txt', mode='a') as f:
                    loss_str = "Epoch:" + str(i) + ", batch:" + str(step) + ", train_loss:" + str(loss.item()) + '\n'
                    f.write(loss_str)

            del imgs_batch, mask_batch, label_base_color, label_roughness, label_metallic, label_normal, x, y_pred

            # release space
            # torch.cuda.empty_cache()

        if loss < best_train_loss:
            best_train_loss = loss
        loss_train.append(loss)

        # test
        test_loss = 0
        total_steps = 0
        for t_steps, (t_imgs_batch, t_mask_batch, t_label_base_color, t_label_roughness,
                      t_label_metallic, t_label_normal) in enumerate(test_dataLoader):
            # input
            t_imgs_batch = t_imgs_batch.to(device)
            t_mask_batch = t_mask_batch.to(device)
            # labels(base_color, roughness, metallic, normal)
            t_label_base_color = t_label_base_color.to(device) * t_mask_batch
            t_label_roughness = t_label_roughness.to(device) * t_mask_batch
            t_label_metallic = t_label_metallic.to(device) * t_mask_batch
            t_label_normal = t_label_normal.to(device) * t_mask_batch
            # cat input
            # t_imgs_batch -> torch.Size([batch_size, 3, 256, 256])
            # t_mask_batch -> torch.Size([batch_size, 1, 256, 256])
            # t_x -> torch.Size([batch_size, 4, 256, 256])
            t_x = torch.cat([t_imgs_batch, t_mask_batch], dim=1)

            # test: no_grad
            with torch.no_grad():
                # forward propagation
                t_y_pred = model(t_x)
                t_y_pred = t_y_pred * t_mask_batch

                # compute test loss
                test_loss = loss_function(t_y_pred, torch.cat([t_label_base_color, t_label_roughness,
                                                               t_label_metallic, t_label_normal], dim=1))

            if t_steps % 100 == 0:
                print("Epoch:{}, batch:{}, test_Loss:{:.8f}".format(i, t_steps, test_loss))
                with open('loss_test.txt', mode='a') as f:
                    loss_str = "Epoch:" + str(i) + ", batch:" + str(t_steps) + ", test_loss:" + str(test_loss.item()) \
                               + '\n '
                    f.write(loss_str)

            del t_imgs_batch, t_mask_batch, t_label_base_color, t_label_roughness, \
                t_label_metallic, t_label_normal, t_x, t_y_pred

            # release space
            # torch.cuda.empty_cache()

            total_steps = total_steps + 1

        if test_loss < best_test_loss:
            best_test_loss = test_loss
        loss_test.append(test_loss / total_steps)

        file_path = save_path + "/checkpoints_" + str(i) + ".pkl"
        torch.save({'epoch': i + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_train_loss': best_train_loss,
                    'best_test_loss': best_test_loss},
                   file_path)

    return loss_train, loss_test


if __name__ == "__main__":
    # get train data's source_img, source_mask and ground truth of train_data
    print('Pre-process Start......')
    source_img, source_mask, label_base_color, label_metallic_roughness, label_normal = utils.get_path(type='train')
    print('The num of training images: ' + str(len(source_img)))
    # pre-process datasets
    package_data = dataset.arrange_datasets(source_img, source_mask,
                                            label_base_color, label_metallic_roughness, label_normal)

    t_source_img, t_source_mask, t_label_base_color, t_label_metallic_roughness, t_label_normal \
        = utils.get_path(type='test')
    print('The num of test images: ' + str(len(t_source_img)))
    t_package_data = dataset.arrange_datasets(t_source_img, t_source_mask, t_label_base_color,
                                              t_label_metallic_roughness, t_label_normal)
    print('done.')

    # initial datasets
    print("Loading path......")
    with open('config.json') as file:
        data = json.load(file)
        dir_path = data['train_path']
        test_path = data['test_path']
        save_path = data['save_path']
    print("done.")

    print("Loading training data......")
    ABO_datasets = dataset.ABO_dataset(package_data, dir_path)
    ABO_dataloader = torch.utils.data.DataLoader(ABO_datasets, batch_size=8, shuffle=True)
    print("done.")

    print("Loading test data......")
    # initial datasets
    t_ABO_datasets = dataset.ABO_dataset(t_package_data, test_path)
    t_ABO_dataloader = torch.utils.data.DataLoader(t_ABO_datasets, batch_size=8, shuffle=True)
    print("done.")

    print("Begin to train......")
    train_loss, test_loss = train(ABO_dataloader, 17, t_ABO_dataloader, save_path)  # 获得loss值，即y轴
    print("done.")

    print("save loss......")
    with open('loss.txt', mode='w') as f:
        f.write(str(train_loss) + '\n')
        f.write(str(test_loss) + '\n')
    print("done.")
# ---------------------------------------------------------------------------------------------------------------------
# x_train_loss = range(len(train_loss))  # loss的数量，即x轴
# plt.figure()
# # 去除顶部和右边框框
# ax = plt.axes()
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# plt.xlabel('iters')  # x轴标签
# plt.ylabel('loss')  # y轴标签
# # 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
# # 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
# plt.plot(x_train_loss, train_loss, linewidth=1, linestyle="solid", label="train loss")
# plt.legend()
# plt.title('Train Loss curve')
# plt.show()
#
# x_loss_test = range(len(test_loss))
# plt.figure()
# # 去除顶部和右边框框
# ax = plt.axes()
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# plt.xlabel('iters')  # x轴标签
# plt.ylabel('acc')  # y轴标签
# # 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
# # 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
# plt.plot(x_loss_test, test_loss, linewidth=1, linestyle="solid", label="train loss")
# plt.legend()
# plt.title('Test Loss curve')
# plt.show()
