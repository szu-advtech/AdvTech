from __future__ import print_function
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from model.resnet import resnet34
from model.basenet import AlexNetBase, VGGBase, Predictor, Predictor_deep
from utils.utils import weights_init
from utils.lr_schedule import inv_lr_scheduler
from utils.return_dataset import return_dataset
from utils.loss import entropy, adentropy
from model.discriminator import get_fc_discriminator
from utils.faa_masks import fc_fft_target_masks
from utils.gate import GateModule96
import torch.nn.functional as F
import pandas as pd
from torch.cuda.amp import autocast as autocast
# from visdom import  Visdom
# Training settings
# viz=Visdom(env='MME_train')
parser = argparse.ArgumentParser(description='SSDA Classification')
parser.add_argument('--steps', type=int, default=50000, metavar='N',
                    help='maximum number of iterations '
                         'to train (default: 50000)')
parser.add_argument('--method', type=str, default='MME_RDA',
                    choices=['S+T', 'ENT', 'MME','MME_RDA'],
                    help='MME is proposed method, ENT is entropy minimization,'
                         ' S+T is training only on labeled examples')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--multi', type=float, default=0.1, metavar='MLT',
                    help='learning rate multiplication')
parser.add_argument('--T', type=float, default=0.05, metavar='T',
                    help='temperature (default: 0.05)')
parser.add_argument('--lamda', type=float, default=0.1, metavar='LAM',
                    help='value of lamda')
parser.add_argument('--save_check', action='store_true', default=False,
                    help='save checkpoint or not')
parser.add_argument('--checkpath', type=str, default='./save_model_ssda',
                    help='dir to save checkpoint')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging '
                         'training status')
parser.add_argument('--save_interval', type=int, default=500, metavar='N',
                    help='how many batches to wait before saving a model')
parser.add_argument('--net', type=str, default='resnet34',
                    help='which network to use')
parser.add_argument('--source', type=str, default='real',
                    help='source domain')
parser.add_argument('--target', type=str, default='sketch',
                    help='target domain')
parser.add_argument('--dataset', type=str, default='multi',
                    choices=['multi', 'office', 'office_home'],
                    help='the name of dataset')
parser.add_argument('--num', type=int, default=3,
                    help='number of labeled examples in the target')
parser.add_argument('--patience', type=int, default=5, metavar='S',
                    help='early stopping to wait for improvment '
                         'before terminating. (default: 5 (5000 iterations))')
parser.add_argument('--early', action='store_false', default=True,
                    help='early stopping on validation or not')

args = parser.parse_args()
print('Dataset %s Source %s Target %s Labeled num perclass %s Network %s' %
      (args.dataset, args.source, args.target, args.num, args.net))
source_loader, target_loader, target_loader_unl, \
target_loader_val,target_loader_test, class_list = return_dataset(args)
use_gpu = torch.cuda.is_available()
record_dir = 'record/%s/%s' % (args.dataset, args.method)
if not os.path.exists(record_dir):
    os.makedirs(record_dir)
record_file = os.path.join(record_dir,
                           '%s_net_%s_%s_to_%s_num_%s' %
                           (args.method, args.net, args.source,
                            args.target, args.num))

torch.cuda.manual_seed(args.seed)
if args.net == 'resnet34':
    G = resnet34()
    inc = 512
elif args.net == "alexnet":
    G = AlexNetBase()
    inc = 4096
elif args.net == "vgg":
    G = VGGBase()
    inc = 4096
else:
    raise ValueError('Model cannot be recognized.')

params = []
for key, value in dict(G.named_parameters()).items():
    if value.requires_grad:
        if 'classifier' not in key:
            params += [{'params': [value], 'lr': args.multi,
                        'weight_decay': 0.0005}]
        else:
            params += [{'params': [value], 'lr': args.multi * 10,
                        'weight_decay': 0.0005}]

if "resnet" in args.net:
    F1 = Predictor_deep(num_class=len(class_list),
                        inc=inc)
else:
    F1 = Predictor(num_class=len(class_list), inc=inc,
                   temp=args.T)
weights_init(F1)
lr = args.lr
G.cuda()
F1.cuda()
f_attacker_target = GateModule96()
f_attacker_target.cuda()


train_loss=[]
test_loss=[]
val_loss=[]
im_data_s = torch.FloatTensor(1)
im_data_t = torch.FloatTensor(1)
im_data_tu = torch.FloatTensor(1)
gt_labels_s = torch.LongTensor(1)
gt_labels_t = torch.LongTensor(1)
sample_labels_t = torch.LongTensor(1)
sample_labels_s = torch.LongTensor(1)

im_data_s = im_data_s.cuda()
im_data_t = im_data_t.cuda()
im_data_tu = im_data_tu.cuda()
gt_labels_s = gt_labels_s.cuda()
gt_labels_t = gt_labels_t.cuda()
sample_labels_t = sample_labels_t.cuda()
sample_labels_s = sample_labels_s.cuda()

im_data_s = Variable(im_data_s)
im_data_t = Variable(im_data_t)
im_data_tu = Variable(im_data_tu)
gt_labels_s = Variable(gt_labels_s)
gt_labels_t = Variable(gt_labels_t)
sample_labels_t = Variable(sample_labels_t)
sample_labels_s = Variable(sample_labels_s)

if os.path.exists(args.checkpath) == False:
    os.mkdir(args.checkpath)
device = 0
source_label=0
target_label=1


def train():
    # d_main = get_fc_discriminator(num_classes=len(class_list))
    # d_main.train()
    # d_main.to(device)
    G.train()
    F1.train()

    optimizer_g = optim.SGD(params, momentum=0.9,
                            weight_decay=0.0005, nesterov=True)
    optimizer_f = optim.SGD(list(F1.parameters()), lr=1.0, momentum=0.9,
                            weight_decay=0.0005, nesterov=True)

    f_attacker_target.train()
    optimizer_f_attacker_target = optim.SGD(f_attacker_target.parameters(),
                                            lr=2.5e-4,
                                            momentum=0.9,
                                            weight_decay=0.0005)
    # optimizer_d_main = optim.Adam(d_main.parameters(), lr=1e-4,betas=(0.9, 0.99))
    interp_target = nn.Upsample(size=(360, 640), mode='bilinear',
                                align_corners=True)
    def zero_grad_all():
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
        optimizer_f_attacker_target.zero_grad()
        # optimizer_d_main.zero_grad()
    param_lr_g = []
    for param_group in optimizer_g.param_groups:
        param_lr_g.append(param_group["lr"])
    param_lr_f = []
    for param_group in optimizer_f.param_groups:
        param_lr_f.append(param_group["lr"])
    criterion = nn.CrossEntropyLoss().cuda()
    all_step = args.steps
    data_iter_s = iter(source_loader)
    data_iter_t = iter(target_loader)
    data_iter_t_unl = iter(target_loader_unl)
    len_train_source = len(source_loader)
    len_train_target = len(target_loader)
    len_train_target_semi = len(target_loader_unl)
    best_acc = 0
    counter = 0
    for step in range(all_step):
        optimizer_g = inv_lr_scheduler(param_lr_g, optimizer_g, step,
                                       init_lr=args.lr)
        optimizer_f = inv_lr_scheduler(param_lr_f, optimizer_f, step,
                                       init_lr=args.lr)
        lr = optimizer_f.param_groups[0]['lr']
        if step % len_train_target == 0:
            data_iter_t = iter(target_loader)
        if step % len_train_target_semi == 0:
            data_iter_t_unl = iter(target_loader_unl)
        if step % len_train_source == 0:
            data_iter_s = iter(source_loader)
        data_t = next(data_iter_t)
        data_t_unl = next(data_iter_t_unl)
        data_s = next(data_iter_s)
        im_data_s.resize_(data_s[0].size()).copy_(data_s[0])
        gt_labels_s.resize_(data_s[1].size()).copy_(data_s[1])
        im_data_t.resize_(data_t[0].size()).copy_(data_t[0])
        gt_labels_t.resize_(data_t[1].size()).copy_(data_t[1])
        im_data_tu.resize_(data_t_unl[0].size()).copy_(data_t_unl[0])
        zero_grad_all()
        data = torch.cat((im_data_s, im_data_t), 0)
        target = torch.cat((gt_labels_s, gt_labels_t), 0)
        output = G(data)
        out1 = F1(output)
        loss = criterion(out1, target)

        loss.backward(retain_graph=True)
        optimizer_g.step()
        optimizer_f.step()
        zero_grad_all()
        previous_images=data_t_unl[0].detach().clone()
        if not args.method == 'S+T':
            # 在此处加入FAA_T提取特征
            # with torch.no_grad():
            #     pred_trg_aux_pooled_ref, pred_trg_main_pooled_ref = pred_trg_aux_pooled.detach().clone(), pred_trg_main_pooled.detach().clone()
            images_target_faa, loss_faa_target = faa_target(im_data_tu.cuda(), f_attacker_target,previous_images.cuda())
            output = G(im_data_tu)
            ouput_im_data_tu_faa = G(images_target_faa.cuda())
            if args.method == 'ENT':
                loss_t = entropy(F1, output, args.lamda)
                loss_t.backward()
                optimizer_f.step()
                optimizer_g.step()
            elif args.method == 'MME_RDA':
                #用F1对攻击图像进行分类
                pred_trg_main=F1(output).detach().clone()
                out_pred_trg_main=F.softmax(pred_trg_main)
                pred_trg_main_faa = F1(ouput_im_data_tu_faa)
                out_pred_trg_main_faa = F.softmax(pred_trg_main_faa)
                loss_rda_target=l1_loss(out_pred_trg_main_faa,out_pred_trg_main)

                # d_out_main_faa = d_main(prob_2_entropy(F.softmax(out_pred_trg_main_faa)))
                # loss_adv_trg_main_faa = bce_loss(d_out_main_faa, source_label)
                loss_t = adentropy(F1, output, args.lamda)
                # loss = loss_adv_trg_main_faa + loss_rda_target+ loss_faa_target
                loss_FAA =loss_rda_target + loss_faa_target

                loss_FAA.backward()

                # viz.line(X=[step], Y=[-loss_t.data.item()], win=args.net + '_MME ' + 'train_loss', update="append",
                #          opts=dict(title=args.net + '_MME ' + 'train_loss', xlabel='step'))
                loss_t.backward()

                torch.nn.utils.clip_grad_norm_(F1.parameters(), 1)
                optimizer_f.step()
                optimizer_g.step()
                optimizer_f_attacker_target.step()
            elif args.method == 'MME':
                loss_t = adentropy(F1, output, args.lamda)
                loss_t.backward()

                optimizer_f.step()
                optimizer_g.step()
                optimizer_f_attacker_target.step()
            else:
                raise ValueError('Method cannot be recognized.')
            log_train = 'S {} T {} Train Ep: {} lr{} \t ' \
                        'Loss Classification: {:.6f} Loss T {:.6f} ' \
                        'Method {}\n'.format(args.source, args.target,
                                             step, lr, loss.data,
                                             -loss_t.data, args.method)
        else:
            log_train = 'S {} T {} Train Ep: {} lr{} \t ' \
                        'Loss Classification: {:.6f} Method {}\n'.\
                format(args.source, args.target,
                       step, lr, loss.data,
                       args.method)
        G.zero_grad()
        F1.zero_grad()
        f_attacker_target.zero_grad()
        zero_grad_all()
        if step % args.log_interval == 0:
            print(log_train)
        if step % args.save_interval == 0 :
            loss_test, acc_test = test(target_loader_test)
            loss_val, acc_val = test(target_loader_val)
            test_loss.append([step,loss_test.item()])
            val_loss.append([step,loss_val.item()])
            train_loss.append([step,loss.item()])
            print("trainloss:{},testloss:{},valloss{}\n".format(loss.item(),loss_test.item(),loss_val.item()))
            G.train()
            F1.train()
            f_attacker_target.train()
            if acc_val >= best_acc:
                best_acc = acc_val
                best_acc_test = acc_test
                counter = 0
            else:
                counter += 1
            if args.early and step>20000:
                if counter > args.patience:
                    val_loss_data=pd.DataFrame(val_loss)
                    test_loss_data=pd.DataFrame(test_loss)
                    train_loss_data=pd.DataFrame(train_loss)
                    val_loss_data.to_csv(args.method+"_val_loss.csv")
                    test_loss_data.to_csv(args.method + "_test_loss.csv")
                    train_loss_data.to_csv(args.method + "_train_loss.csv")
                    break
            print('best acc test %f best acc val %f' % (best_acc_test,
                                                        acc_val))
            print('record %s' % record_file)
            with open(record_file, 'a') as f:
                f.write('step %d best %f final %f \n' % (step,
                                                         best_acc_test,
                                                         acc_val))

            G.train()
            F1.train()
            f_attacker_target.train()
            if args.save_check:
                print('saving model')
                torch.save(G.state_dict(),
                           os.path.join(args.checkpath,
                                        "G_iter_model_{}_{}_"
                                        "to_{}_step_{}.pth.tar".
                                        format(args.method, args.source,
                                               args.target, step)))
                torch.save(F1.state_dict(),
                           os.path.join(args.checkpath,
                                        "F1_iter_model_{}_{}_"
                                        "to_{}_step_{}.pth.tar".
                                        format(args.method, args.source,
                                               args.target, step)))
    if step==args.steps:
        val_loss_data = pd.DataFrame(val_loss)
        test_loss_data = pd.DataFrame(test_loss)
        train_loss_data = pd.DataFrame(train_loss)
        val_loss_data.to_csv(args.method + "_val_loss.csv")
        test_loss_data.to_csv(args.method + "_test_loss.csv")
        train_loss_data.to_csv(args.method + "_train_loss.csv")

def test(loader):
    G.eval()
    F1.eval()
    test_loss = 0
    correct = 0
    size = 0
    num_class = len(class_list)
    output_all = np.zeros((0, num_class))
    criterion = nn.CrossEntropyLoss().cuda()
    confusion_matrix = torch.zeros(num_class, num_class)
    with torch.no_grad():
        for batch_idx, data_t in enumerate(loader):
            im_data_t.resize_(data_t[0].size()).copy_(data_t[0])
            gt_labels_t.resize_(data_t[1].size()).copy_(data_t[1])
            feat = G(im_data_t)
            output1 = F1(feat)
            output_all = np.r_[output_all, output1.data.cpu().numpy()]
            size += im_data_t.size(0)
            pred1 = output1.data.max(1)[1]
            for t, p in zip(gt_labels_t.view(-1), pred1.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            correct += pred1.eq(gt_labels_t.data).cpu().sum()
            test_loss += criterion(output1, gt_labels_t) / len(loader)
    print('\nTest set: Average loss: {:.4f}, '
          'Accuracy: {}/{} F1 ({:.0f}%)\n'.
          format(test_loss, correct, size,
                 100. * correct / size))
    return test_loss.data, 100. * float(correct) / size

def faa_target(input_images, f_attacker_target, ref_images):
    output_images1, loss_faa = faa_attack(input_images, ref_images,fc_fft_target_masks, f_attacker_target, portion=0.1)
    # scale_ratio = np.random.randint(100.0*0.8, 100.0 * 1.2) / 100.0
    # scaled_size_target = (round(360 * scale_ratio / 8) * 8, round(640 * scale_ratio / 8) * 8)
    # interp_target_sc = nn.Upsample(size=scaled_size_target, mode='bilinear', align_corners=True)
    # output_images2 = interp_target_sc(output_images1)
    # return output_images2, loss_faa
    return output_images1, loss_faa

def faa_attack(input_images, ref_images,fc_fft_masks, gate, portion=0.1):
    n = 32
    #interpolation

    interp_src = nn.Upsample(size = (input_images.shape[-2], input_images.shape[-1]), mode='bilinear', align_corners=True)
    ref_images = interp_src(ref_images)
    #transform
    #fft_input = torch.rfft( input_images.clone(), signal_ndim=2, onesided=False)
    fft_input =torch.fft.ifft2(input_images.clone(),dim=(-2,-1))
    fft_input=torch.stack((fft_input.real,fft_input.imag),-1)

    #fft_ref = torch.rfft( ref_images.clone(), signal_ndim=2, onesided=False )
    fft_ref = torch.fft.ifft2(ref_images.clone(),dim=(-2,-1))
    fft_ref = torch.stack((fft_ref.real,fft_ref.imag),-1)

    b, c, im_h, im_w, _ = fft_input.shape

    # extract amplitude and phase of both ffts (1, 3, h, w)
    amp_src, pha_src = extract_ampl_phase( fft_input.clone())
    amp_trg, pha_trg = extract_ampl_phase( fft_ref.clone())


    #band_pass filter
    amp_trg_32 = torch.unsqueeze(amp_trg, 1) #(1, 1, 3, h, ...)
    amp_trg_32 = amp_trg_32.expand((b, n, c, im_h, im_w)) #(1, 32, 3, h, ...)

    amp_trg_32 = amp_trg_32.cuda() * fc_fft_masks[:, :, :, :, :].cuda() #(1, n, 3, h, w)
    amp_trg_96 = amp_trg_32.view(b, c*n, im_h, im_w)

    amp_src_32 = torch.unsqueeze(amp_src, 1) #(1, 1, 3, h, ...)
    amp_src_32 = amp_src_32.expand((b, n, c, im_h, im_w)) #(1, 32, 3, h, ...)

    amp_src_32 = amp_src_32.cuda() * fc_fft_masks[:, :, :, :, :].cuda() #(1, n, 3, h, w)
    amp_src_96 = amp_src_32.view(b, c*n, im_h, im_w)

    _, gate_scores = gate(amp_src_96)

    gate_scores = gate_scores[:, :, 0]


    topk = torch.topk(gate_scores,dim=1, k=int(np.floor(n*3*portion)))
    gate_index=topk[1]

    gate_scores = gate_scores * 0


    for i in range(len(gate_scores)):
        gate_scores[i][gate_index[i]] = 1.0


    gate_scores = gate_scores.view(b, 96, 1, 1)

    # Gate portion loss -- no need as it directly used portion = 0.1
    # loss_gate_portion = torch.clamp((torch.sum(gate_scores) - int(np.floor(n*3*portion))), min=0)
    #生成攻击图像
    amp_src_96_faa = (amp_src_96 * torch.abs(1 - gate_scores)) + ( amp_trg_96 * gate_scores )

    amp_src_32_faa = amp_src_96_faa.view(b, n, c, im_h, im_w)

    amp_src_ = torch.sum(amp_src_32_faa, dim=1)
    # Reconstruction loss -- enforce the FCs with shape/outline content to be unchanged
    #重建损失，用生成的图像和原图像进行对比
    loss_recon = l1_loss(amp_src_96_faa[:, :, :, :], amp_src_96[:,:, :, :])

    loss_faa = - loss_recon * 0.0002
    # recompose fft of source
    fft_input_ = torch.zeros( fft_input.size(), dtype=torch.float )

    fft_input_[:,:,:,:,0] = torch.cos(pha_src.clone()) * amp_src_.clone()
    fft_input_[:,:,:,:,1] = torch.sin(pha_src.clone()) * amp_src_.clone()

    # get the recomposed image: images perturbed by faa
    _, _, imgH, imgW = input_images.size()
    #output_images_faa = torch.irfft( fft_input_, signal_ndim=2, onesided=False, signal_sizes=[imgH,imgW] )
    output_images_faa = torch.fft.irfft2(fft_input_)
    return output_images_faa[:,:,:,:,0], loss_faa
def extract_ampl_phase(fft_im):
    # fft_im: size should be bx3xhxwx2
    fft_amp = fft_im[:,:,:,:,0]**2 + fft_im[:,:,:,:,1]**2
    fft_amp = torch.sqrt(fft_amp)
    fft_pha = torch.atan2( fft_im[:,:,:,:,1], fft_im[:,:,:,:,0] )
    return fft_amp, fft_pha
def l1_loss(input, target):
    loss = torch.abs(input - target)
    loss = torch.mean(loss)
    return loss
def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)
def bce_loss(y_pred, y_label):
    y_truth_tensor = torch.FloatTensor(y_pred.size())
    y_truth_tensor.fill_(y_label)
    y_truth_tensor = y_truth_tensor.to(y_pred.get_device())
    return nn.BCEWithLogitsLoss()(y_pred, y_truth_tensor)

train()
