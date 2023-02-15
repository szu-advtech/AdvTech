import jittor as jt
from jittor import nn
from jittor.dataset import ImageFolder

from segment.model.PVT_Unet import PVTUnet as Model
import os
from segment.utils.util import train_one_epoch
from segment.utils.util import evaluate
import math
from tensorboardX import SummaryWriter
from jittor import optim
from jittor import lr_scheduler
import argparse
# import jittor.transform as transform

from utils.dataset import VOCSegDataset as Dataset  # VOC数据集
# from jseg.datasets.ade import ADE20KDataset as Dataset
import jseg.datasets.pipelines.transforms as transforms
from utils.util import Evaluator

from segment.model.config import CONFIGS
config = CONFIGS['PVT_Unet_small']
from jittor.dataset.cifar import CIFAR10


def main(args):
    global config
    jt.flags.use_cuda = 1
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")
    tb_writer = SummaryWriter()

    data_dir = config.data_root
    batch_size = config.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    # piplines = [transforms.RandomCrop((224, 224)),
    #                         transforms.RandomRotate(prob=0.4,degree=10),
    #                         transforms.RGB2Gray(),
    #                         transforms.RandomFlip(),
    #                         transforms.Normalize(0., 1.), transforms.PhotoMetricDistortion()]
    # my_transform = Compose()
    # train_data_loader = Dataset(batch_size=8, shuffle=True,img_dir='K:/dataset/ADE20K/data/training',ann_dir='',pipeline=piplines)
    # val_data_loader = Dataset(batch_size=8, shuffle=True,img_dir='K:/dataset/ADE20K/data/ADE20K_2016_07_26/images/validation',pipeline=[])

    train_data_loader = Dataset(is_train=True,crop_size=(224, 224),voc_root='K:/dataset/VOC2012').set_attrs(batch_size=batch_size,shuffle=True)
    val_data_loader = Dataset(is_train=False,crop_size=(224, 224),voc_root='K:/dataset/VOC2012').set_attrs(batch_size=batch_size,shuffle=True)

    model = Model(config=config)
    evaluator = Evaluator(num_class=config.num_classes)
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = jt.load(args.weights)
        # 删除不需要的权重(分类头的权重)
        del_keys = ['head.weight', 'head.bias']
        for k in del_keys:
            del weights_dict[k]
        # print(model.load_state_dict(weights_dict, strict=False))
        print(model.load_state_dict(weights_dict))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head, pre_logits外，其他权重全部冻结
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    optimizer = optim.RMSprop(model.parameters(), lr=config.lr, eps=1e-8, alpha=0.9)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', patience=2)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    best_miou = 0
    for epoch in range(config.epoch):
        # train
        train_loss = train_one_epoch(model=model,
                                     optimizer=optimizer,
                                     data_loader=train_data_loader,
                                     epoch=epoch)

        # validate
        val_loss, best_miou, mIoU, dice = evaluate(model=model,
                                                   data_loader=val_data_loader,
                                                   evaluator=evaluator,
                                                   best_miou=best_miou,
                                                   epoch=epoch)
        scheduler.step(val_loss)
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        # tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        # 每5个epoch就保存一次权重参数
        if epoch % 5 == 0:
            jt.save(model.state_dict(), "./weights/model-{}.pkl".format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--model-name', default='', help='create model name')
    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()
    main(opt)
