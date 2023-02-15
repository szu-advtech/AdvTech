import os

import torch
import torch.utils.data
import time

import config
import sen12ms_cr_dataLoader
from model.DSen2_CR import DSen2_CR
from model.value_metric import carl_error, cloud_psnr, cloud_root_mean_squared_error
from data_process.dataIO import get_cloudy_true_predict_img


def train():
    args = config.get_args()
    print("开始划分数据集")

    dataset = sen12ms_cr_dataLoader.SEN12MSCRDataset(args.dataset_dir)
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataiter = iter(val_dataloader)
    print("数据集划分完毕")
    print("训练集大小：{}\r\n测试集大小：{}\r\n数据集初始化完毕".format(len(train_dataset), len(val_dataset)))

    # 定义网络结构
    net = DSen2_CR()

    # 使用Adam优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=0.00001)

    # 使用云自适应损失函数carl
    carl_loss = carl_error

    # 将数据装入gpu（在有gpu且使用GPU进行训练的情况下）
    cloud_img = torch.FloatTensor(args.batch_size, 15, 256, 256)
    ground_truth = torch.FloatTensor(args.batch_size, 13, 256, 256)
    csm_img = torch.FloatTensor(args.batch_size, 1, 256, 256)

    """# 如果使用GPU 则把这些玩意全部放进显存里面去
    # if config.use_gpu:
    net = net.cuda()
    # CARL_Loss= CARL_Loss.cuda() 这是定义的外部函数 不能放入CUDA
    cloud_img = cloud_img.cuda()
    ground_truth = ground_truth.cuda()
    csm_img = csm_img.cuda()"""

    print("开始训练...")

    for epoch in range(1, args.epoch):
        epoch_start_time = time.time()
        # 数据集的小循环
        for iteration, batch in enumerate(train_dataloader, start=1):
            # 数据操作    numpy 数据转成tensor
            img_cld, img_csm, img_truth = batch

            img_cld = cloud_img.resize_(img_cld.shape).copy_(img_cld)
            img_csm = csm_img.resize_(img_csm.shape).copy_(img_csm)
            img_truth = ground_truth.resize_(img_truth.shape).copy_(img_truth)

            # print(img_cld.shape,img_csm.shape,img_truth.shape)

            # 网络训练
            img_predict = net(img_cld)


            optimizer.zero_grad()
            loss = carl_loss(img_truth, img_predict, img_csm)
            loss.backward()
            optimizer.step()

            if iteration % args.show_result == 0:
                with torch.no_grad():
                    print("epoch[{}]({}/{}): loss: {:.8f}".format(
                        epoch, iteration, len(train_dataloader), loss.item()))

                    input_cloudy, input_csm, input_true = next(val_dataiter)
                    net.eval()
                    # input_cloudy = input_cloudy.cuda()
                    predict_result = net(input_cloudy)
                    net.train()
                    get_cloudy_true_predict_img(input_cloudy, input_true, input_csm, predict_result)

                    # print(cloud_root_mean_squared_error(input_true, predict_result))
                    # print(cloud_psnr(input_true, predict_result))


            # 保存网络
            if iteration % args.save_model == 0 or epoch == args.epoch:
                path = os.path.join(os.getcwd(), 'output', f"net_epoch_{epoch}_iteration_{iteration}.pth")
                if not os.path.exists(os.path.join(os.getcwd(), 'output')):
                    os.makedirs("./output")

                torch.save(net.state_dict(), path)
                print(f"第{epoch}轮训练结果已保存")

        print("第{}轮训练完毕，用时{}S".format(epoch, time.time() - epoch_start_time))



if __name__ == "__main__":
    train()