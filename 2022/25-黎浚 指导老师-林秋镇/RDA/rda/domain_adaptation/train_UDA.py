import os
import sys
from pathlib import Path

import os.path as osp
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch import nn
from torchvision.utils import make_grid
from tqdm import tqdm

from RDA.rda.model.discriminator import get_fc_discriminator
from RDA.rda.utils.func import adjust_learning_rate, adjust_learning_rate_discriminator
from RDA.rda.utils.func import loss_calc, bce_loss
from RDA.rda.utils.func import prob_2_entropy
from RDA.rda.utils.viz_segmask import colorize_mask

from RDA.rda.utils.gate import GateModule96
from RDA.rda.utils.faa_masks import fc_fft_source_masks, fc_fft_target_masks

import matplotlib.pyplot as plt


def train_domain_adaptation(model, trainloader, targetloader, cfg):
    if cfg.TRAIN.DA_METHOD == 'RDA':
        train_RDA(model, trainloader, targetloader, cfg)
    else:
        raise NotImplementedError(f"Not yet supported DA method {cfg.TRAIN.DA_METHOD}")


def train_RDA(model, trainloader, targetloader, cfg):
    ''' UDA training with RDA and AdaptSeg
    '''
    input_size_source = cfg.TRAIN.INPUT_SIZE_SOURCE
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    device = cfg.GPU_ID
    num_classes = cfg.NUM_CLASSES
    viz_tensorboard = os.path.exists(cfg.TRAIN.TENSORBOARD_LOGDIR)
    if viz_tensorboard:
        writer = SummaryWriter(log_dir=cfg.TRAIN.TENSORBOARD_LOGDIR)

    model.train()
    model.to(device)
    cudnn.benchmark = True
    cudnn.enabled = True

    d_aux = get_fc_discriminator(num_classes=num_classes)
    d_aux.train()
    d_aux.to(device)

    d_main = get_fc_discriminator(num_classes=num_classes)
    d_main.train()
    d_main.to(device)

    f_attacker_source = GateModule96()
    f_attacker_source.train()
    f_attacker_source.to(device)

    f_attacker_target = GateModule96()
    f_attacker_target.train()
    f_attacker_target.to(device)

    optimizer = optim.SGD(model.optim_parameters(cfg.TRAIN.LEARNING_RATE),
                          lr=cfg.TRAIN.LEARNING_RATE,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    optimizer_d_aux = optim.Adam(d_aux.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                 betas=(0.9, 0.99))
    optimizer_d_main = optim.Adam(d_main.parameters(), lr=cfg.TRAIN.LEARNING_RATE_D,
                                  betas=(0.9, 0.99))

    optimizer_f_attacker_source = optim.SGD(f_attacker_source.parameters(),
                                            lr=cfg.TRAIN.LEARNING_RATE_faa,
                                            momentum=cfg.TRAIN.MOMENTUM,
                                            weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    optimizer_f_attacker_target = optim.SGD(f_attacker_target.parameters(),
                                            lr=cfg.TRAIN.LEARNING_RATE_faa,
                                            momentum=cfg.TRAIN.MOMENTUM,
                                            weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    interp = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear',
                         align_corners=True)

    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear',
                                align_corners=True)

    source_label = 0
    target_label = 1
    trainloader_iter = enumerate(trainloader)
    targetloader_iter = enumerate(targetloader)

    _, batch = targetloader_iter.__next__()
    images, _, _, _ = batch
    previous_images = images.detach().clone()

    for i_iter in tqdm(range(cfg.TRAIN.EARLY_STOP + 1)):
        optimizer.zero_grad()
        optimizer_d_aux.zero_grad()
        optimizer_d_main.zero_grad()
        optimizer_f_attacker_source.zero_grad()
        optimizer_f_attacker_target.zero_grad()

        adjust_learning_rate(optimizer, i_iter, cfg)
        adjust_learning_rate(optimizer_f_attacker_source, i_iter, cfg)
        adjust_learning_rate(optimizer_f_attacker_target, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_aux, i_iter, cfg)
        adjust_learning_rate_discriminator(optimizer_d_main, i_iter, cfg)


        for param in d_aux.parameters():
            param.requires_grad = False
        for param in d_main.parameters():
            param.requires_grad = False
#源域训练集数据输入训练
        _, batch = trainloader_iter.__next__()
        images_source, labels, _, _ = batch
        pred_src_aux_pooled, pred_src_main_pooled = model(images_source.cuda(device))
        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux = interp(pred_src_aux_pooled)
            loss_seg_src_aux = loss_calc(pred_src_aux, labels, device)
        else:
            loss_seg_src_aux = 0
        pred_src_main = interp(pred_src_main_pooled)
        loss_seg_src_main = loss_calc(pred_src_main, labels, device)
        loss = (cfg.TRAIN.LAMBDA_SEG_MAIN * loss_seg_src_main
                + cfg.TRAIN.LAMBDA_SEG_AUX * loss_seg_src_aux)
        loss.backward()

        # # RDA flow.
        # with torch.no_grad():
        #     pred_src_aux_pooled_ref, pred_src_main_pooled_ref = pred_src_aux_pooled.detach().clone(), pred_src_main_pooled.detach().clone()
        # images_source_faa, loss_faa_source = faa_source(images_source.cuda(device), cfg, f_attacker_source, previous_images.cuda(device))
        # pred_src_aux_faa, pred_src_main_faa = model(images_source_faa.cuda(device))
        # interp_faa_source = nn.Upsample(size = (pred_src_main_pooled_ref.shape[-2], pred_src_main_pooled_ref.shape[-1]), mode='bilinear', align_corners=True)
        # pred_src_main_faa_pooled = interp_faa_source(pred_src_main_faa)
        # out_src_main_faa_pooled = F.softmax(pred_src_main_faa_pooled)
        # out_src_main_pooled = F.softmax(pred_src_main_pooled_ref)
        # loss_rda_source = l1_loss(out_src_main_faa_pooled, out_src_main_pooled)
        # if cfg.TRAIN.MULTI_LEVEL:
        #     pred_src_aux_faa_pooled = interp_faa_source(pred_src_aux_faa)
        #     out_src_aux_faa_pooled = F.softmax(pred_src_aux_faa_pooled)
        #     out_src_aux_pooled = F.softmax(pred_src_aux_pooled_ref)
        #     loss_rda_aux_source = l1_loss(out_src_aux_faa_pooled, out_src_aux_pooled)
        # else:
        #     loss_rda_aux_source = 0
        # loss = (loss_rda_source + loss_rda_aux_source) + loss_faa_source
        # loss.backward()
        #lijunadd
        # _, batch1 = trainloader_iter.__next__()
        # images_source, labels, _, _ = batch

        _, batch = targetloader_iter.__next__()
        images, _, _, _ = batch
        previous_images = images.detach().clone()
#目标域数据输入预测
        pred_trg_aux_pooled, pred_trg_main_pooled = model(images.cuda(device))
#训练域鉴别器，此处使用目标域输出结果混淆鉴别器
        if cfg.TRAIN.MULTI_LEVEL:
            pred_trg_aux = interp_target(pred_trg_aux_pooled)
            d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_trg_aux)))
            loss_adv_trg_aux = bce_loss(d_out_aux, source_label)
        else:
            loss_adv_trg_aux = 0
        pred_trg_main = interp_target(pred_trg_main_pooled)
        d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main)))
        loss_adv_trg_main = bce_loss(d_out_main, source_label)
        loss = (cfg.TRAIN.LAMBDA_ADV_MAIN * loss_adv_trg_main
                + cfg.TRAIN.LAMBDA_ADV_AUX * loss_adv_trg_aux)
        loss = loss
        loss.backward()

        # RDA flow.
        #梯度清零
        with torch.no_grad():
            pred_trg_aux_pooled_ref, pred_trg_main_pooled_ref = pred_trg_aux_pooled.detach().clone(), pred_trg_main_pooled.detach().clone()
        images_target_faa, loss_faa_target = faa_target(images.cuda(device), cfg, f_attacker_target, previous_images.cuda(device))
        #images_target_faa, loss_faa_target=faa_source(images_source.cuda(device),cfg,f_attacker_source,previous_images.cuda(device))
        #用生成的目标域图输入模型得到预测概率
        pred_trg_aux_faa, pred_trg_main_faa = model(images_target_faa.cuda(device))
        #攻击的目标域图片得出的结果进行采样
        interp_faa_target = nn.Upsample(size = (pred_trg_main_pooled_ref.shape[-2], pred_trg_main_pooled_ref.shape[-1]), mode='bilinear', align_corners=True)
        pred_trg_main_faa_pooled = interp_faa_target(pred_trg_main_faa)
        out_trg_main_faa_pooled = F.softmax(pred_trg_main_faa_pooled)
        out_trg_main_pooled_ref = F.softmax(pred_trg_main_pooled_ref)
        #目标域图像中攻击和未攻击判读结果的损失
        loss_rda_target = l1_loss(out_trg_main_faa_pooled, out_trg_main_pooled_ref)
        if cfg.TRAIN.MULTI_LEVEL:
            pred_trg_aux_faa_pooled = interp_faa_target(pred_trg_aux_faa)
            out_trg_aux_faa_pooled = F.softmax(pred_trg_aux_faa_pooled)
            out_trg_aux_pooled_ref = F.softmax(pred_trg_aux_pooled_ref)
            loss_rda_target_aux = l1_loss(out_trg_aux_faa_pooled, out_trg_aux_pooled_ref)
        else:
            loss_rda_target_aux = 0
        #两个域鉴别器求损失
        if cfg.TRAIN.MULTI_LEVEL:
            pred_trg_aux_faa_ori = interp_target(pred_trg_aux_faa_pooled)
            d_out_aux_faa = d_aux(prob_2_entropy(F.softmax(pred_trg_aux_faa_ori)))
            loss_adv_trg_aux_faa = bce_loss(d_out_aux_faa, source_label)
        else:
            loss_adv_trg_aux_faa = 0
        pred_trg_main_faa_ori = interp_target(pred_trg_main_faa_pooled)
        d_out_main_faa = d_main(prob_2_entropy(F.softmax(pred_trg_main_faa_ori)))
        loss_adv_trg_main_faa = bce_loss(d_out_main_faa, source_label)
        #总的损失为域鉴别器损失+攻击后目标域任务模型损失+傅里叶对抗损失
        loss = (cfg.TRAIN.LAMBDA_ADV_MAIN * loss_adv_trg_main_faa
                + cfg.TRAIN.LAMBDA_ADV_AUX * loss_adv_trg_aux_faa) + (loss_rda_target + loss_rda_target_aux) + loss_faa_target
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer_f_attacker_source.step()
        optimizer_f_attacker_target.step()
        optimizer.step()


        # Train discriminator networks
        # enable training mode on discriminator networks
        for param in d_aux.parameters():
            param.requires_grad = True
        for param in d_main.parameters():
            param.requires_grad = True
        # train with source
        if cfg.TRAIN.MULTI_LEVEL:
            pred_src_aux = pred_src_aux.detach()
            d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_src_aux)))
            loss_d_aux = bce_loss(d_out_aux, source_label)
            loss_d_aux = loss_d_aux / 2
            loss_d_aux.backward()
        pred_src_main = pred_src_main.detach()
        d_out_main = d_main(prob_2_entropy(F.softmax(pred_src_main)))
        loss_d_main = bce_loss(d_out_main, source_label)
        loss_d_main = loss_d_main / 2
        loss_d_main.backward()

        # train with target
        if cfg.TRAIN.MULTI_LEVEL:
            pred_trg_aux = pred_trg_aux.detach()
            d_out_aux = d_aux(prob_2_entropy(F.softmax(pred_trg_aux)))
            loss_d_aux = bce_loss(d_out_aux, target_label)
            loss_d_aux = loss_d_aux / 2
            loss_d_aux.backward()
        else:
            loss_d_aux = 0
        pred_trg_main = pred_trg_main.detach()
        d_out_main = d_main(prob_2_entropy(F.softmax(pred_trg_main)))
        loss_d_main = bce_loss(d_out_main, target_label)
        loss_d_main = loss_d_main / 2
        loss_d_main.backward()
        if cfg.TRAIN.MULTI_LEVEL:
            optimizer_d_aux.step()
        optimizer_d_main.step()

        current_losses = {
            # 'loss_rda_source': (loss_rda_source + loss_rda_aux_source),
                          'loss_rda_target': (cfg.TRAIN.LAMBDA_ADV_MAIN * loss_adv_trg_main_faa + cfg.TRAIN.LAMBDA_ADV_AUX * loss_adv_trg_aux_faa) + (loss_rda_target + loss_rda_target_aux),
                          'loss_d_main': loss_d_main}
        print_losses(current_losses, i_iter)

        if i_iter % cfg.TRAIN.SAVE_PRED_EVERY == 0 and i_iter != 0:
            print('taking snapshot ...')
            #print('exp =', cfg.TRAIN.SNAPSHOT_DIR)
            snapshot_dir = Path('/root/autodl-nas/')
            torch.save(model.state_dict(), snapshot_dir / f'model_{i_iter}.pth')

            if i_iter >= cfg.TRAIN.EARLY_STOP - 1:
                break
        sys.stdout.flush()
        torch.cuda.empty_cache()


def l1_loss(input, target):
    loss = torch.abs(input - target)
    loss = torch.mean(loss)
    return loss


def draw_in_tensorboard(writer, images, i_iter, pred_main, num_classes, type_):
    grid_image = make_grid(images[:3].clone().cpu().data, 3, normalize=True)
    writer.add_image(f'Image - {type_}', grid_image, i_iter)

    grid_image = make_grid(torch.from_numpy(np.array(colorize_mask(np.asarray(
        np.argmax(F.softmax(pred_main).cpu().data[0].numpy().transpose(1, 2, 0),
                  axis=2), dtype=np.uint8)).convert('RGB')).transpose(2, 0, 1)), 3,
                           normalize=False, range=(0, 255))
    writer.add_image(f'Prediction - {type_}', grid_image, i_iter)

    output_sm = F.softmax(pred_main).cpu().data[0].numpy().transpose(1, 2, 0)
    output_ent = np.sum(-np.multiply(output_sm, np.log2(output_sm)), axis=2,
                        keepdims=False)
    grid_image = make_grid(torch.from_numpy(output_ent), 3, normalize=True,
                           range=(0, np.log2(num_classes)))
    writer.add_image(f'Entropy - {type_}', grid_image, i_iter)


def print_losses(current_losses, i_iter):
    list_strings = []
    for loss_name, loss_value in current_losses.items():
        list_strings.append(f'{loss_name} = {to_numpy(loss_value):.3f} ')
    full_string = ' '.join(list_strings)
    tqdm.write(f'iter = {i_iter} {full_string}')


def log_losses_tensorboard(writer, current_losses, i_iter):
    for loss_name, loss_value in current_losses.items():
        writer.add_scalar(f'data/{loss_name}', to_numpy(loss_value), i_iter)


def to_numpy(tensor):
    if isinstance(tensor, (int, float)):
        return tensor
    else:
        return tensor.data.cpu().numpy()


def faa_target(input_images, cfg, f_attacker_target, ref_images):
    output_images1, loss_faa = faa_attack(input_images, ref_images, f_attacker_target, fc_fft_target_masks, cfg, portion=0.1)
    scale_ratio = np.random.randint(100.0*cfg.TRAIN.SCALING_RATIO[0], 100.0 * cfg.TRAIN.SCALING_RATIO[1]) / 100.0
    scaled_size_target = (round(cfg.TRAIN.INPUT_SIZE_TARGET[1] * scale_ratio / 8) * 8, round(cfg.TRAIN.INPUT_SIZE_TARGET[0] * scale_ratio / 8) * 8)
    interp_target_sc = nn.Upsample(size=scaled_size_target, mode='bilinear', align_corners=True)
    output_images2 = interp_target_sc(output_images1)
    return output_images2, loss_faa


def faa_source(input_images, cfg, f_attacker_source, ref_images):
    output_images1, loss_faa = faa_attack(input_images, ref_images, f_attacker_source, fc_fft_source_masks, cfg, portion=0.1)
    scale_ratio = np.random.randint(100.0*cfg.TRAIN.SCALING_RATIO[0], 100.0 * cfg.TRAIN.SCALING_RATIO[1]) / 100.0
    scaled_size_source = (round(cfg.TRAIN.INPUT_SIZE_SOURCE[1] * scale_ratio / 8) * 8, round(cfg.TRAIN.INPUT_SIZE_SOURCE[0] * scale_ratio / 8) * 8)
    interp_source_sc = nn.Upsample(size=scaled_size_source, mode='bilinear', align_corners=True)
    output_images2 = interp_source_sc(output_images1)
    return output_images2, loss_faa


def faa_attack(input_images, ref_images, gate, fc_fft_masks, cfg, portion=0.1):
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
    amp_trg_32 = amp_trg_32 * fc_fft_masks[:, :, :, :, :, 0].cuda(cfg.GPU_ID) #(1, n, 3, h, w)
    amp_trg_96 = amp_trg_32.view(b, c*n, im_h, im_w)

    amp_src_32 = torch.unsqueeze(amp_src, 1) #(1, 1, 3, h, ...)
    amp_src_32 = amp_src_32.expand((b, n, c, im_h, im_w)) #(1, 32, 3, h, ...)
    amp_src_32 = amp_src_32 * fc_fft_masks[:, :, :, :, :, 0].cuda(cfg.GPU_ID) #(1, n, 3, h, w)
    amp_src_96 = amp_src_32.view(b, c*n, im_h, im_w)
    _, gate_scores = gate(amp_src_96)

    gate_scores = gate_scores[0, :, 0]
    _, gate_index = torch.topk(gate_scores, int(np.floor(n*3*portion)))
    gate_scores = gate_scores * 0
    gate_scores[gate_index] = 1.0
    gate_scores = gate_scores.view(1, 96, 1, 1)

    # Gate portion loss -- no need as it directly used portion = 0.1
    # loss_gate_portion = torch.clamp((torch.sum(gate_scores) - int(np.floor(n*3*portion))), min=0)
    #生成攻击图像
    amp_src_96_faa = (amp_src_96 * torch.abs(1 - gate_scores)) + ( amp_trg_96 * gate_scores )
    amp_src_32_faa = amp_src_96_faa.view(b, n, c, im_h, im_w)
    amp_src_ = torch.sum(amp_src_32_faa, dim=1)

    # Reconstruction loss -- enforce the FCs with shape/outline content to be unchanged
    #重建损失，用生成的图像和原图像进行对比
    loss_recon = l1_loss(amp_src_96_faa[:, 6:48, :, :], amp_src_96[:, 6:48, :, :])
    loss_faa = - loss_recon * cfg.TRAIN.LAMBDA_FAA_AUX

    # recompose fft of source
    fft_input_ = torch.zeros( fft_input.size(), dtype=torch.float )
    fft_input_[:,:,:,:,0] = torch.cos(pha_src.clone()) * amp_src_.clone()
    fft_input_[:,:,:,:,1] = torch.sin(pha_src.clone()) * amp_src_.clone()

    # get the recomposed image: images perturbed by faa
    _, _, imgH, imgW = input_images.size()
    #output_images_faa = torch.irfft( fft_input_, signal_ndim=2, onesided=False, signal_sizes=[imgH,imgW] )
    output_images_faa = torch.fft.irfft2(fft_input_, dim=(-2, -1))
    return output_images_faa[...,0], loss_faa


def extract_ampl_phase(fft_im):
    # fft_im: size should be bx3xhxwx2
    fft_amp = fft_im[:,:,:,:,0]**2 + fft_im[:,:,:,:,1]**2
    fft_amp = torch.sqrt(fft_amp)
    fft_pha = torch.atan2( fft_im[:,:,:,:,1], fft_im[:,:,:,:,0] )
    return fft_amp, fft_pha
