import os
import argparse

import torch
import torch.optim as optim

from torchvision import transforms

from my_dataset import MyDataSet
from model import swin_tiny_patch4_window7_224 as create_model
from utils import read_split_data, train_one_epoch, evaluate

import wandb

# 使用的是Poly策略调整学习率
def create_lr_scheduler(optimizer,num_step: int,epochs: int,warmup=True,warmup_epochs=1, warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)



def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    #
    wandb.init(project="1000", id="test17")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")


    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    img_size = 224
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes).to(device)
    # weights_dict = torch.load(args.weights, map_location=device)
    #
    # model.load_state_dict(weights_dict, strict=False)

    if args.weights != "":

        weights_dict = torch.load(args.weights, map_location=device)["model"]

        print(model.load_state_dict(weights_dict, strict=False))

    # if True:
    #     for name, para in model.named_parameters():
    #         # 除head外，其他权重全部冻结
    #         if "head" not in name:
    #             para.requires_grad_(False)
    #         else:
    #             print("training {}".format(name))
    #
    # pg = [p for p in model.parameters() if p.requires_grad]
    

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5E-2)

    # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), 100, warmup=True)

    for epoch in range(args.epochs):
        # train
        train_loss, train_acc,lr = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch,lr_scheduler=lr_scheduler,wandb = wandb)


        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        print("val:           ",val_loss  ,"           ",val_acc)

        wandb.log({"val_loss": val_loss,  "val_acc":val_acc})


        torch.save(model.state_dict(), "./weights/modeltest-{}.pth".format(epoch))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0002)


    parser.add_argument('--data-path', type=str,
                        default="../train")

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='weights/SWIN.pth',
                        help='initial weights path')

    parser.add_argument('--device', default='cuda:4', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
