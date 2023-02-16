import random
import jittor as jt
from jittor import nn
import os
import math
from jittor.dataset import ImageFolder
from tensorboardX import SummaryWriter
from jittor import optim
import argparse
import jittor.transform as transforms
from jittor.dataset.cifar import CIFAR10
from model.my_PVT_classification import classification as Model
from utils.util import train_one_epoch
from utils.util import evaluate
from model.config import CONFIGS

config = CONFIGS['PVT_classification_tiny']


# config = CONFIGS['PVT_classification_small']


def main(args):
    global config
    jt.flags.use_cuda = 1
    if os.path.exists("weights") is False:
        os.makedirs("weights")

    tb_writer = SummaryWriter()
    # train_dir = config.train_dir
    # val_dir = config.val_dir
    # batch_size = config.batch_size
    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    # print('Using {} dataloader workers every process'.format(nw))
    # my_transform = transform.Compose([transform.Resize(size=(224, 224))])
    # train_data_loader = ImageFolder(train_dir, transform=my_transform).set_attrs(batch_size=batch_size, num_workers=nw,
    #                                                                              shuffle=True)
    # val_data_loader = ImageFolder(val_dir, transform=my_transform).set_attrs(batch_size=batch_size, num_workers=nw,
    #                                                                          shuffle=True)
    p1 = random.random()
    p2 = random.random()
    transform = transforms.Compose([transforms.RandomGray(),
                                    transforms.RandomHorizontalFlip(p1),
                                    transforms.RandomVerticalFlip(p2),
                                    transforms.RandomRotation(10, resample=False,
                                                              expand=False,
                                                              center=None),
                                    transforms.ColorJitter(brightness=0.5,
                                                           contrast=0.5, hue=0.5)])
    train_data_loader = CIFAR10(train=True, transform=transform).set_attrs(shuffle=True)
    val_data_loader = CIFAR10(train=False, transform=None).set_attrs(shuffle=True)
    model = Model(config=config)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = jt.load(args.weights)
        # 删除不需要的权重(分类头的权重)
        del_keys = ['head.weight', 'head.bias']
        for k in del_keys:
            del weights_dict[k]
        print(model.load_state_dict(weights_dict))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head, pre_logits外，其他权重全部冻结
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    # pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = nn.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=5E-5, nesterov=True)
    # optimizer = jt.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, nesterov=True)    # optimizer = nn.Adam(model.parameters(), lr=config.lr)  # 换个优化器
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / config.epoch)) / 2) * (1 - config.lrf) + config.lrf  # cosine
    scheduler = optim.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(config.epoch):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_data_loader,
                                                epoch=epoch)

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_data_loader,
                                     epoch=epoch)
        scheduler.step(val_loss)
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        # 每5个epoch就保存一次权重参数
        if (epoch + 1) % 5 == 0:
            jt.save(model.state_dict(), "./weights/model-{}.pkl".format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default='')
    parser.add_argument('--model-name', default='', help='create model name')

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()
    main(opt)
