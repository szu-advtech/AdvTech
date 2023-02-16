import argparse
import sys
from packaging import version
import time
import util
import os
import os.path as osp
import timeit
from collections import OrderedDict
import scipy.io

import torch
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from operator import itemgetter

import scipy
from scipy import ndimage
import math
from PIL import Image
import numpy as np
import shutil
import random

from deeplab.model_advent import Res_Deeplab
from deeplab.datasets_advent import GTA5TestDataSet
from deeplab.datasets_advent import SrcSTDataSet, GTA5StMineDataSet, SoftSrcSTDataSet, SoftGTA5StMineDataSet

### shared ###
# IMG_MEAN = np.array((0.406, 0.456, 0.485), dtype=np.float32) # BGR
# IMG_STD = np.array((0.225, 0.224, 0.229), dtype=np.float32) # BGR
### for advent
IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
IMG_STD = np.array((1.0, 1.0, 1.0), dtype=np.float32)

# data
### source
## gta
DATA_SRC_DIRECTORY = './dataset/gta5'
DATA_SRC_LIST_PATH = './dataset/list/gta5/train.lst'
DATA_SRC = 'gta'
RESTORE_FROM = './src_model/gta5/src_model.pth'
NUM_CLASSES = 19
INIT_SRC_PORT = 0.03 # GTA: 0.03
### target
DATA_TGT_DIRECTORY = './dataset/cityscapes'
DATA_TGT_TRAIN_LIST_PATH = './dataset/list/cityscapes/train_ClsConfSet.lst'
DATA_TGT_TEST_LIST_PATH = './dataset/list/cityscapes/val.lst'
IGNORE_LABEL = 255
# train scales for src and tgt
TRAIN_SCALE_SRC = '0.5,1.5'
TRAIN_SCALE_TGT = '0.5,1.5'
# model
MODEL = 'DeeplabRes'
# gpu
GPU = 0
PIN_MEMORY = False
# log files
# LOG_FILE = 'self_training_log'
LOG_FILE = 'testing_log'

### train ###
BATCH_SIZE = 2
INPUT_SIZE = '512,1024'# 512,1024 for GTA;
RANDSEED = 3
# params for optimizor
LEARNING_RATE =5e-5
POWER = 0.0
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
NUM_ROUNDS = 4
EPR = 2
SRC_SAMPLING_POLICY = 'r'
KC_POLICY = 'cb'
KC_VALUE = 'conf'
INIT_TGT_PORT = 0.2
MAX_TGT_PORT = 0.5
TGT_PORT_STEP = 0.05
# varies but dataset
MAX_SRC_PORT = 0.06 #0.06;
SRC_PORT_STEP = 0.0025 #0.0025:
MRKLD = 0.0
LRENT = 0.0
MRSRC = 0.0
MINE_PORT = 1e-3
RARE_CLS_NUM = 3
MINE_CHANCE = 0.8
### val ###
SAVE_PATH = 'debug'
TEST_IMAGE_SIZE = '1024,2048'
EVAL_SCALE = 0.9
# TEST_SCALE = '0.9,1.0,1.2'
TEST_SCALE = '0.5,0.8,1.0'
DS_RATE = 4

def seed_torch(seed=0):
   random.seed(seed)
   os.environ['PYTHONHASHSEED'] = str(seed)
   np.random.seed(seed)
   torch.manual_seed(seed)
   torch.cuda.manual_seed(seed)
   torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
   torch.backends.cudnn.benchmark = True
   torch.backends.cudnn.enabled = False
   #torch.backends.cudnn.deterministic = True

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    ### shared by train & val
    # data
    parser.add_argument("--data-src", type=str, default=DATA_SRC,
                        help="Name of source dataset.")
    parser.add_argument("--data-src-dir", type=str, default=DATA_SRC_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-src-list", type=str, default=DATA_SRC_LIST_PATH,
                        help="Path to the file listing the images&labels in the source dataset.")
    parser.add_argument("--data-tgt-dir", type=str, default=DATA_TGT_DIRECTORY,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-tgt-train-list", type=str, default=DATA_TGT_TRAIN_LIST_PATH,
                        help="Path to the file listing the images*GT labels in the target train dataset.")
    parser.add_argument("--data-tgt-test-list", type=str, default=DATA_TGT_TEST_LIST_PATH,
                        help="Path to the file listing the images*GT labels in the target test dataset.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    # model
    parser.add_argument("--model", type=str, default=MODEL,
                        help="Model Choice (DeeplabMulti/DeeplabVGG).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    # gpu
    parser.add_argument("--gpu", type=int, default=GPU,
                        help="choose gpu device.")
    parser.add_argument("--pin-memory", type=bool, default=PIN_MEMORY,
                        help="Whether to pin memory in train & eval.")
    # log files
    parser.add_argument("--log-file", type=str, default=LOG_FILE,
                        help="The name of log file.")
    parser.add_argument('--debug',help='True means logging debug info.',
                        default=False, action='store_true')
    ### train ###
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--eval-training", action="store_true",
                        help="Use the saved means and variances, or running means and variances during the evaluation.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--train-scale-src", type=str, default=TRAIN_SCALE_SRC,
                        help="The scale for multi-scale training in source domain.")
    parser.add_argument("--train-scale-tgt", type=str, default=TRAIN_SCALE_TGT,
                    help="The scale for multi-scale training in target domain.")
    # params for optimizor
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    ### val
    parser.add_argument('--test-flipping', dest='test_flipping',
                        help='If average predictions of original and flipped images.',
                        default=False, action='store_true')
    parser.add_argument("--test-image-size", type=str, default=TEST_IMAGE_SIZE,
                        help="The test image size.")
    parser.add_argument("--eval-scale", type=float, default=EVAL_SCALE,
                        help="The test image scale.")
    parser.add_argument("--test-scale", type=str, default=TEST_SCALE,
                        help="The test image scale.")
    ### self-training params
    parser.add_argument("--save", type=str, default=SAVE_PATH,
                        help="Path to save result for self-training.")
    parser.add_argument("--num-rounds", type=int, default=NUM_ROUNDS,
                        help="Number of rounds for self-training.")
    parser.add_argument("--epr", type=int, default=EPR,
                        help="Number of epochs per round for self-training.")
    parser.add_argument('--kc-policy', default=KC_POLICY, type=str, dest='kc_policy',
                        help='The policy to determine kc. "cb" for weighted class-balanced threshold')
    parser.add_argument('--kc-value', default=KC_VALUE, type=str,
                        help='The way to determine kc values, either "conf", or "prob".')
    parser.add_argument('--ds-rate', default=DS_RATE, type=int,
                        help='The downsampling rate in kc calculation.')
    parser.add_argument('--init-tgt-port', default=INIT_TGT_PORT, type=float, dest='init_tgt_port',
                        help='The initial portion of target to determine kc')
    parser.add_argument('--max-tgt-port', default=MAX_TGT_PORT, type=float, dest='max_tgt_port',
                        help='The max portion of target to determine kc')
    parser.add_argument('--tgt-port-step', default=TGT_PORT_STEP, type=float, dest='tgt_port_step',
                        help='The portion step in target domain in every round of self-paced self-trained neural network')
    parser.add_argument('--init-src-port', default=INIT_SRC_PORT, type=float, dest='init_src_port',
                        help='The initial portion of source portion for self-trained neural network')
    parser.add_argument('--max-src-port', default=MAX_SRC_PORT, type=float, dest='max_src_port',
                        help='The max portion of source portion for self-trained neural network')
    parser.add_argument('--src-port-step', default=SRC_PORT_STEP, type=float, dest='src_port_step',
                        help='The portion step in source domain in every round of self-paced self-trained neural network')
    parser.add_argument('--randseed', default=RANDSEED, type=int,
                        help='The random seed to sample the source dataset.')
    parser.add_argument("--src-sampling-policy", type=str, default=SRC_SAMPLING_POLICY,
                        help="The sampling policy on source dataset: 'c' for 'cumulative' and 'r' for replace ")
    parser.add_argument('--mine-port', default=MINE_PORT, type=float,
                        help='If a class has a predication portion lower than the mine_port, then mine the patches including the class in self-training.')
    parser.add_argument('--rare-cls-num', default=RARE_CLS_NUM, type=int,
                        help='The number of classes to be mined.')
    parser.add_argument('--mine-chance', default=MINE_CHANCE, type=float,
                        help='The chance of patch mining.')
    parser.add_argument('--rm-prob',
                        help='If remove the probability maps generated in every round.',
                        default=False, action='store_true')
    parser.add_argument('--mr-weight-kld', default=MRKLD, type=float, dest='mr_weight_kld',
                        help='weight of kld model regularization')
    parser.add_argument('--lr-weight-ent', default=LRENT, type=float, dest='lr_weight_ent',
                        help='weight of negative entropy label regularization')
    parser.add_argument('--mr-weight-src', default=MRSRC, type=float, dest='mr_weight_src',
                        help='weight of regularization in source domain')

    parser.add_argument("--num-epoch", type=int, default=2, dest='num_epoch',
                        help="Number of rounds for self-training.")

    return parser.parse_args()

args = get_arguments()

# palette
if args.data_src == 'gta':
    # gta:
    palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
               220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
               0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]

if args.data_src == 'synthia':
    # synthia:
    palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
               220, 220, 0, 107, 142, 35, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142,
               0, 60, 100, 0, 0, 230, 119, 11, 32]

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

def main():
    randseed = args.randseed
    seed_torch(randseed)
    device = torch.device("cuda:" + str(args.gpu))
    save_path = args.save
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    logger = util.set_logger(args.save, args.log_file, args.debug)
    logger.info('start with arguments %s', args)

    restore_from = args.restore_from
    model = Res_Deeplab(num_classes=args.num_classes)
    loc = "cuda:" + str(args.gpu)
    saved_state_dict = torch.load(restore_from, map_location=loc)
    new_params = saved_state_dict.copy()
    model.load_state_dict(new_params)

    _, _, _, test_num = parse_split_list(args.data_tgt_test_list)

    ## label mapping
    sys.path.insert(0, 'dataset/helpers')
    if args.data_src == 'synthia':
        from labels_cityscapes_synthia import id2label, trainId2label
    elif args.data_src == 'gta':
        from labels import id2label, trainId2label
    label_2_id = 255 * np.ones((256,))
    for l in id2label:
        if l in (-1, 255):
            continue
        label_2_id[l] = id2label[l].trainId
    id_2_label = np.array([trainId2label[_].id for _ in trainId2label if _ not in (-1, 255)])
    valid_labels = sorted(set(id_2_label.ravel()))

    tgt_set = 'test'
    save_eval_path = osp.join(args.save, 'testSet_vis')
    if not os.path.exists(save_eval_path):
        os.makedirs(save_eval_path)
    test(model, device, save_eval_path, tgt_set, test_num, args.data_tgt_test_list, label_2_id,
         valid_labels, args, logger)

def test(model, device, save_eval_path, tgt_set, test_num, test_list, label_2_id, valid_labels, args, logger):
    """Create the model and start the evaluation process."""
    ## scorer
    scorer = ScoreUpdater(valid_labels, args.num_classes, test_num, logger)
    scorer.reset()
    h, w = map(int, args.test_image_size.split(','))
    test_image_size = (h, w)
    test_size = ( h, w )
    test_scales = [float(_) for _ in str(args.test_scale).split(',')]
    num_scales = len(test_scales)

    ## test data loader
    testloader = data.DataLoader(GTA5TestDataSet(args.data_tgt_dir, test_list, test_size=test_size, test_scale=1.0, mean=IMG_MEAN, std=IMG_STD, scale=False, mirror=False),
                                    batch_size=1, shuffle=False, pin_memory=args.pin_memory)
    model.eval()
    model.to(device)

    ## upsampling layer
    if version.parse(torch.__version__) >= version.parse('0.4.0'):
        interp = nn.Upsample(size=test_image_size, mode='bilinear', align_corners=True)
    else:
        interp = nn.Upsample(size=test_image_size, mode='bilinear')

    ## evaluation process
    logger.info('###### Start evaluating in target domain {} set! ######'.format(tgt_set))
    start_eval = time.time()
    with torch.no_grad():
        for index, batch in enumerate(testloader):
            image, label, _, name = batch

            img = image.clone()
            for scale_idx in range(num_scales):
                if version.parse(torch.__version__) > version.parse('0.4.0'):
                    image = F.interpolate(img, scale_factor=test_scales[scale_idx], mode='bilinear', align_corners=True)
                else:
                    test_size = (int(h * test_scales[scale_idx]), int(w * test_scales[scale_idx]))
                    interp_tmp = nn.Upsample(size=test_size, mode='bilinear', align_corners=True)
                    image = interp_tmp(img)
                if args.model == 'DeeplabRes':
                    output2 = model(image.to(device))
                    coutput = interp(output2).cpu().data[0].numpy()

                    import pdb
                    pdb.set_trace()

                if args.test_flipping:
                    output2 = model(torch.from_numpy(image.numpy()[:,:,:,::-1].copy()).to(device))
                    coutput = 0.5 * ( coutput + interp(output2).cpu().data[0].numpy()[:,:,::-1] )
                if scale_idx == 0:
                    output = coutput.copy()
                else:
                    output = output+coutput

                # import pdb
                # pdb.set_trace()

            output = output/num_scales
            output = output.transpose(1,2,0)
            amax_output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
            # score
            pred_label = amax_output.copy()
            label = label_2_id[np.asarray(label.numpy(), dtype=np.uint8)]
            scorer.update(pred_label.flatten(), label.flatten(), index)

            # save visualized seg maps
            amax_output_col = colorize_mask(amax_output)
            name = name[0].split('/')[-1]
            image_name = name.split('.')[0]
            amax_output_col.save('%s/%s_color.png' % (save_eval_path, image_name))

    logger.info('###### Finish evaluating in target domain {} set! Time cost: {:.2f} seconds. ######'.format(tgt_set, time.time()-start_eval))
    return

def parse_split_list(list_name):
    image_list = []
    image_name_list = []
    label_list = []
    file_num = 0
    with open(list_name) as f:
        for item in f.readlines():
            fields = item.strip().split('\t')
            image_name = fields[0].split('/')[-1]
            image_list.append(fields[0])
            image_name_list.append(image_name)
            label_list.append(fields[1])
            file_num += 1
    return image_list, image_name_list, label_list, file_num

class ScoreUpdater(object):
    # only IoU are computed. accu, cls_accu, etc are ignored.
    def __init__(self, valid_labels, c_num, x_num, logger=None, label=None, info=None):
        self._valid_labels = valid_labels

        self._confs = np.zeros((c_num, c_num))
        self._per_cls_iou = np.zeros(c_num)
        self._logger = logger
        self._label = label
        self._info = info
        self._num_class = c_num
        self._num_sample = x_num

    @property
    def info(self):
        return self._info

    def reset(self):
        self._start = time.time()
        self._computed = np.zeros(self._num_sample) # one-dimension
        self._confs[:] = 0

    def fast_hist(self,label, pred_label, n):
        k = (label >= 0) & (label < n)
        return np.bincount(n * label[k].astype(int) + pred_label[k], minlength=n ** 2).reshape(n, n)

    def per_class_iu(self,hist):
        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

    def do_updates(self, conf, i, computed=True):
        if computed:
            self._computed[i] = 1
        self._per_cls_iou = self.per_class_iu(conf)

    def update(self, pred_label, label, i, computed=True):
        conf = self.fast_hist(label, pred_label, self._num_class)
        self._confs += conf
        self.do_updates(self._confs, i, computed)
        self.scores(i)

    def scores(self, i=None, logger=None):
        x_num = self._num_sample
        ious = np.nan_to_num( self._per_cls_iou )

        logger = self._logger if logger is None else logger
        if logger is not None:
            if i is not None:
                speed = 1. * self._computed.sum() / (time.time() - self._start)
                logger.info('Done {}/{} with speed: {:.2f}/s'.format(i + 1, x_num, speed))
            name = '' if self._label is None else '{}, '.format(self._label)
            logger.info('{}mean iou: {:.2f}%'. \
                        format(name, np.mean(ious) * 100))
            with util.np_print_options(formatter={'float': '{:5.2f}'.format}):
                logger.info('\n{}'.format(ious * 100))

        return ious

if __name__ == '__main__':
    main()
