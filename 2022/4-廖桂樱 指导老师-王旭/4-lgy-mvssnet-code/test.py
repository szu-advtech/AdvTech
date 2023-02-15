import os
import torch
from mvssnetdataset import MVSSNetDataset
from torch.utils.data import DataLoader
from tool.averagemeter import AverageMeter
from tool.auc import cal_auc
from tool.iou import cal_iou
from tool.f1 import cal_f1
from tqdm import tqdm
import numpy as np
import cv2
from mvssnet import get_mvss
import matplotlib.pyplot as plt
# from MVSS_Net import get_mvss

def display(image):
    fig = plt.figure(figsize=(4,3))
    ax = fig.add_subplot(111)
    ax.imshow(image)

def test(args):
    if not os.path.exists(os.path.join(args.result_save_dir, args.model+args.suffix)):
        os.makedirs(os.path.join(os.path.join(args.result_save_dir, args.model+args.suffix), 'forgery'))
        os.makedirs(os.path.join(os.path.join(args.result_save_dir, args.model+args.suffix), 'gt'))
        os.makedirs(os.path.join(os.path.join(args.result_save_dir, args.model+args.suffix), 'mask'))
    assert os.path.exists(args.weight_save_dir)
    model = get_mvss(backbone='resnet50', pretrained_base=True, nclass=1, sobel=True, constrain=True, n_input=3, is_plus=False)
    if len(args.multi_gpu) > 1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.cuda()
    model_path = "/data/liaogy/weight/mvssnet_casia2.0/experiment_2_epoch_76.pth"
    model.load_state_dict(torch.load(model_path, map_location='cpu')['state_dict'])

    test_dataset = MVSSNetDataset(root=args.root, txt=args.txt, val_txt=args.val_txt, mode='test', size=args.size, is_used_edge=False,
                                  seed=args.seed)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    print("Now test model2 in ", args.val_txt)
    # ./././ test ./././ #
    model.eval()
    iou = AverageMeter()
    f1 = AverageMeter()
    auc = AverageMeter()
    t = tqdm(test_loader)
    with torch.no_grad():
        for idx, (forgery, mask) in enumerate(t):
            if idx == args.test_num:
                exit()
            forgery, mask = forgery.cuda(), mask.cuda()
            edge, outputs = model(forgery, 1) # mvssnet
            # 像素级别的检测
            trans_mask = mask.detach().cpu().numpy()
            trans_mask = np.expand_dims(trans_mask, axis=1)
            trans_outputs = outputs.detach().cpu().numpy()
            auc.update(cal_auc(trans_outputs, trans_mask))
            f1.update(cal_f1(trans_outputs, trans_mask))
            iou.update(cal_iou(trans_outputs, trans_mask))
            trans_forgery = forgery[0].detach().cpu().numpy() # (c, h, w)
            trans_forgery = np.transpose(trans_forgery, (1, 2, 0)) # (h, w, c)
            trans_forgery = (trans_forgery * 255).astype(np.uint8) # [0, 1]->[0, 255]
            trans_forgery = cv2.cvtColor(trans_forgery, cv2.COLOR_RGB2BGR)
            trans_mask = mask[0].detach().cpu().numpy()  # (h, w)
            trans_mask = (trans_mask * 255).astype(np.uint8)
            trans_outputs = trans_outputs.squeeze()
            trans_outputs = (trans_outputs*255).astype(np.uint8)
            cv2.imwrite(os.path.join(os.path.join(args.result_save_dir, args.model+args.suffix),
                                         'forgery/forgery_{}.png'.format(idx)), trans_forgery)
            cv2.imwrite(os.path.join(os.path.join(args.result_save_dir, args.model+args.suffix),
                                         'gt/gt_{}.png'.format(idx)), trans_mask)
            cv2.imwrite(os.path.join(os.path.join(args.result_save_dir, args.model+args.suffix),
                                         'mask/mask_{}.png'.format(idx)), trans_outputs)
            t.set_description('auc:{:.3f} | f1:{:.3f} | iou:{:.3f} | '.format(auc.avg, f1.avg, iou.avg))


if __name__ == '__main__':
    outputs = np.array([0, 5, 0, 2, 0, 1, 1, 0, 1, 0], dtype=np.int32)
    k = np.sum(outputs)
    # print(k)
    # outputs = torch.tensor(outputs)
    # print(outputs[0])
    # print(max(max(outputs)))
    # print(nn.MaxPool1d(outputs))
    # print(outputs.shape)
    # k = torch.max(outputs[0][1])
    # print(int(k))
    # print(torch.max(outputs[0]))

    # print(torch.max(outputs[0][0]))
    # gt = np.array([[0, 0, 1, 1, 0, 1, 1, 1, 1, 0]], dtype=np.int32)
    # outputs = outputs.reshape((1, 2, 5))
    # gt = gt.reshape((1, 2, 5))
    # print(outputs.shape, gt.shape)
    # print(cal_f1(outputs, gt))

    # filter_x = np.array([
    #     [1, 0, -1],
    #     [2, 0, -2],
    #     [1, 0, -1],
    # ]).astype(np.float32)
    # print(filter_x.shape)
    # filter_x = filter_x.reshape((1, 1, 3, 3))
    # print(filter_x)
    # print(filter_x.shape)
    print(0.9975**(16*16))