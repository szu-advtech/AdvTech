"""Run testing given a trained model."""

import argparse
import time
import torch.backends.cudnn as cudnn
import random

import numpy as np
import torch.nn.parallel
import torch.optim
import torchvision

from dataset import CoviarDataSet
from model import Model
from transforms import GroupCenterCrop
from transforms import GroupOverSample
from transforms import GroupScale
from CAM import show_feature_map
from timesformer.models.vit import TimeSformer
import os
import cv2
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50



os.environ['CUDA_VISIBLE_DEVICES']="6"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
parser = argparse.ArgumentParser(
    description="Standard video-level testing")
parser.add_argument('--data-name', type=str, choices=['ucf101', 'hmdb51','mmi'])
parser.add_argument('--representation', type=str, choices=['iframe', 'residual', 'mv'])
parser.add_argument('--no-accumulation', action='store_true',
                    help='disable accumulation of motion vectors and residuals.')
parser.add_argument('--data-root', type=str)
parser.add_argument('--test-list', type=str)
parser.add_argument('--weights', type=str)
parser.add_argument('--arch', type=str)
parser.add_argument('--save-scores', type=str, default=None)
parser.add_argument('--test_segments', type=int, default=16)
parser.add_argument('--test-crops', type=int, default=1)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of workers for data loader.')
parser.add_argument('--gpus', nargs='+', type=int, default=None)

args = parser.parse_args()

if args.data_name == 'ucf101':
    num_class = 101
elif args.data_name == 'hmdb51':
    num_class = 51
elif args.data_name == 'mmi':
    num_class = 6
else:
    raise ValueError('Unknown dataset '+args.data_name)

def returnCAM(feature_conv, weight_softmax, class_idx):


    bz, nc, h, w = feature_conv.shape        #1,2048,7,7

    output_cam = []

    for idx in class_idx:  #只输出预测概率最大值结果不需要for循环
        feature_conv = feature_conv.reshape((nc, h*w))
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))  #(2048, ) * (2048, 7*7) -> (7*7, ) （n,）是一个数组，既不是行向量也不是列向量

        cam = cam.reshape(h, w)

        cam_img = (cam - cam.min()) / (cam.max() - cam.min())  #Normalize
        cam_img = np.uint8(255 * cam_img)                      #Format as CV_8UC1 (as applyColorMap required)

        #output_cam.append(cv2.resize(cam_img, size_upsample))  # Resize as image size
        output_cam.append(cam_img)
    return output_cam


# 定义获取梯度的函数
def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())

# 定义获取特征图的函数
def farward_hook(module, input, output):
    fmap_block.append(output)

def cam_show_img(img, feature_map, grads, out_dir):
    H, W, _ = img.shape
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)		# 4
    grads = grads.reshape([grads.shape[0],-1])					# 5
    weights = np.mean(grads, axis=1)							# 6
    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]							# 7
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    cam = cv2.resize(cam, (W, H))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam_img = 0.3 * heatmap + 0.7 * img

    path_cam_img = os.path.join(out_dir, "cam.jpg")
    cv2.imwrite(path_cam_img, cam_img)

def compute_loss(logit, index=None):
    if  index:
        index = np.argmax(logit.cpu().data.numpy())
    else:
        index = np.array(index[0])

    index = index[np.newaxis, np.newaxis]
    index = torch.from_numpy(index)

    index = index.long()
    one_hot = torch.zeros(1, 6).scatter_(1, index, 1)
    # one_hot = torch.zeros(batch_size, self.num_cls).scatter_(1, index, 1)
    one_hot.requires_grad = True
    one_hot=one_hot.cuda()
    logit=logit.cuda()
    loss = torch.sum(one_hot * logit)
    return loss


def compute_cam(feature_map, grads):
    """
    feature_map: np.array [C, H, W]
    grads: np.array, [C, H, W]
    return: np.array, [H, W]
    """
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)
    alpha = np.mean(grads, axis=(1, 2))  # GAP
    for k, ak in enumerate(alpha):
        cam += ak * feature_map[k]  # linear combination

    cam = np.maximum(cam, 0)  # relu
    cam = cv2.resize(cam, self.size)
    cam = (cam - np.min(cam)) / np.max(cam)
    return cam

def __show_cam_on_image(self, img: np.ndarray, mask: np.ndarray, if_show=True, if_write=False):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    if if_write:
        cv2.imwrite("camcam.jpg", cam)
    if if_show:
        # 要显示RGB的图片，如果是BGR的 热力图是反过来的
        plt.imshow(cam[:, :, ::-1])
        plt.show()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True

# def main():

if __name__ == '__main__':
    fmap_block = list()
    grad_block = list()

    seed = random.randint(1, 10000)
    # seed = 62
    setup_seed(seed)
    print("seed==",seed)
    # main()

    if args.arch in ['ts']:
        net = TimeSformer(img_size=224, num_classes=num_class, num_frames=args.test_segments,
                          attention_type='divided_space_time',
                          pretrained_model="/data/jlzhang/.cache/torch/hub/checkpoints/jx_vit_base_p16_224-80ecf9dd.pth",
                          representation=args.representation)
    else:
        net = Model(num_class, args.test_segments, args.representation,
                    base_model=args.arch)

    checkpoint = torch.load(args.weights)
    print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))

    base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
    net.load_state_dict(base_dict)


    # 存放梯度和特征图

    fc_weights = net.state_dict()['fc.weight'].cpu().numpy()

    if args.test_crops == 1:
        cropping = torchvision.transforms.Compose([
            GroupScale(net.scale_size),
            GroupCenterCrop(net.crop_size),
        ])
    elif args.test_crops == 10:
        cropping = torchvision.transforms.Compose([
            GroupOverSample(net.crop_size, net.scale_size, is_mv=(args.representation == 'mv'))
        ])
    else:
        raise ValueError("Only 1 and 10 crops are supported, but got {}.".format(args.test_crops))

    data_loader = torch.utils.data.DataLoader(
        CoviarDataSet(
            args.data_root,
            args.data_name,
            video_list=args.test_list,
            num_segments=args.test_segments,
            representation=args.representation,
            transform=cropping,
            is_train=False,
            accumulate=(args.no_accumulation),
            ),
        batch_size=1, shuffle=False,
        num_workers=args.workers * 2, pin_memory=True)

    if args.gpus is not None:
        devices = [args.gpus[i] for i in range(args.workers)]
    else:
        devices = list(range(args.workers))
    # torch.cuda.set_device(2)

    # USE_CUDA = torch.cuda.is_available()
    # device = torch.device("cuda:0" if USE_CUDA  else "cpu")
    # net = torch.nn.DataParallel(net)
    # net.to(device)
    # cudnn.benchmark = True

    # print(net.base_model[-1][2])
    # net.base_model[-1][2].register_forward_hook(farward_hook)  # 9
    # net.base_model[-1][2].register_backward_hook(backward_hook)

    # net = torch.nn.DataParallel(net.cuda(devices[0]), device_ids=devices)

    # net = torch.nn.DataParallel(net)
    net.to(device)
    cudnn.benchmark = True
    net.eval()

    net.base_model[-1][2].register_forward_hook(farward_hook)  # 9
    net.base_model[-1][2].register_backward_hook(backward_hook)
    # print(net.base_model)

    # net['base_model'][-1][2].register_forward_hook(farward_hook)  # 9
    # net['base_model'][-1][2].register_backward_hook(backward_hook)

    # data_gen = enumerate(data_loader)

    total_num = len(data_loader.dataset)
    output = []

    def forward_video(data,den_input):
        with torch.no_grad():
            input_var = torch.autograd.Variable(data)
            den_input_var = torch.autograd.Variable(den_input)
        # input_var = torch.autograd.Variable(data, volatile=True)
        if args.arch in ['ts']:
            input_var = input_var.transpose(1, 2)
        # target_layers=net.module.base_model[0][-1]

        scores,fearture,MI_2 = net(input_var,den_input_var)
        # cam = GradCAM(model=net, target_layers=target_layers, use_cuda=True)
        # targets = [ClassifierOutputTarget(6)]
        # grayscale_cam = cam(input_tensor=input_var, targets=targets)
        # grayscale_cam = grayscale_cam[0, :]
        # visualization = show_cam_on_image(input_var, grayscale_cam, use_rgb=True)


        # scores ,fearture= net(input_var)
        # show_feature_map(data, fearture, 4, 1, 1)


        # scores2=scores[20:30,:]
        # scores2=scores2*0.5
        # scores[20:30,:]=scores2
        # scores2=scores[10:20,:]
        # scores2=scores2*0
        # scores[10:20,:]=scores2
        # if args.arch not in ['ts']:
        #     scores = scores.view((-1, args.test_segments * args.test_crops) + scores.size()[1:])

            # scores = torch.mean(scores, dim=1)

        # scores = torch.sum(scores, dim=1)

        # return scores.data.cpu().numpy().copy()
        return scores,fearture


    proc_start_time = time.time()
    class_ = {0: 'anger', 1: 'Disgust',2:'Fear',3:'Happiness',4:'Sadness',5:'Surprise'}

    for i, (data, label,den_input) in enumerate(data_loader):

        data = data.cuda()
        label=label.cuda()
        den_input=den_input.cuda()
        # data=np.squeeze(data)
        video_scores,feature = forward_video(data,den_input)

        feature=feature.detach().cpu().numpy()

        h_x = torch.nn.functional.softmax(video_scores, dim=1).data.squeeze()

        probs, idx = h_x.sort(0, True)  # 按概率从大到小排列
        probs = probs.cpu().numpy()  # if tensor([0.0019,0.9981]) ->[0.9981, 0.0019]
        idx = idx.cpu().numpy()  # [1, 0]

        CAMs = returnCAM(feature, fc_weights, [idx[0]])

        img = data
        height, width, _ = img.shape  # 读取输入图片的尺寸
        heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)),
                                    cv2.COLORMAP_JET)  # CAM resize match input image size
        result = heatmap * 0.3 + img * 0.5  # 比例可以自己调节

        text = '%s %.2f%%' % (class_[idx[0]], probs[0] * 100)  # 激活图结果上的文字显示
        cv2.putText(result, text, (210, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.9,
                    color=(123, 222, 238), thickness=2, lineType=cv2.LINE_AA)
        CAM_RESULT_PATH = "/data/jlzhang/Py_protect/2022-08-05/pytorch-coviar-master/CAM/"  # CAM结果的存储地址
        if not os.path.exists(CAM_RESULT_PATH):
            os.mkdir(CAM_RESULT_PATH)
        image_name_ = img.split(".")[-2]
        cv2.imwrite(CAM_RESULT_PATH + image_name_ + '_' + 'pred_' + class_[idx[0]] + '.jpg', result)  # 写入存储磁盘

        # idx = np.argmax(video_scores.cpu().data.numpy())



        # net.zero_grad()
        # class_loss = compute_loss(video_scores, label.cuda().cpu().numpy())
        # class_loss.backward()
        #
        # grads_val = grad_block[0].cpu().data.numpy().squeeze()
        # fmap = fmap_block[0].cpu().data.numpy().squeeze()
        # output_dir="/data/jlzhang/Py_protect/2022-08-05/pytorch-coviar-master/CAM/"
        # cam_show_img(data, fmap, grads_val, output_dir)


        output.append((video_scores, label[0]))
        cnt_time = time.time() - proc_start_time
        if (i + 1) % 100 == 0:
            print('video {} done, total {}/{}, average {} sec/video'.format(i, i+1,
                                                                            total_num,
                                                                            float(cnt_time) / (i+1)))

    video_pred = [np.argmax(x[0]) for x in output]
    video_labels = [x[1] for x in output]
    video_true=np.array(video_pred) == np.array(video_labels)
    # x=-1
    # for i in video_true :
    #     x=x+1
    #     if i:
    #         print(x)



    print('Accuracy {:.02f}% ({})'.format(
        float(np.sum(np.array(video_pred) == np.array(video_labels))) / len(video_pred) * 100.0,
        len(video_pred)))


    if args.save_scores is not None:

        name_list = [x.strip().split()[0] for x in open(args.test_list)]
        order_dict = {e:i for i, e in enumerate(sorted(name_list))}

        reorder_output = [None] * len(output)
        reorder_label = [None] * len(output)
        reorder_name = [None] * len(output)

        for i in range(len(output)):
            idx = order_dict[name_list[i]]
            reorder_output[idx] = output[i]
            reorder_label[idx] = video_labels[i]
            reorder_name[idx] = name_list[i]

        np.savez(args.save_scores, scores=reorder_output, labels=reorder_label, names=reorder_name)


# if __name__ == '__main__':
#     fmap_block = list()
#     grad_block = list()
#
#     seed = random.randint(1, 10000)
#     # seed = 62
#     setup_seed(seed)
#     print("seed==",seed)
#     main()
