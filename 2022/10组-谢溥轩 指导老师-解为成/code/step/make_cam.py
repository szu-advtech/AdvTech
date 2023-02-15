import torch
from torch import multiprocessing, cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn

import numpy as np
import importlib
import os

import voc12.dataloader
from misc import torchutils, imutils

cudnn.enabled = True
from step.gradCAM import GradCAM


def adv_climb(image, epsilon, data_grad):
    sign_data_grad = data_grad / (torch.max(torch.abs(data_grad)) + 1e-12)
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, image.min().data.cpu().float(),
                                  image.max().data.cpu().float())  # min, max from data normalization
    return perturbed_image


def add_discriminative(expanded_mask, regions, score_th):
    region_ = regions / regions.max()
    expanded_mask[region_ > score_th] = 1
    return expanded_mask


def _work(process_id, model, dataset, args):
    databin = dataset[process_id]
    n_gpus = torch.cuda.device_count()
    data_loader = DataLoader(databin, shuffle=False, num_workers=args.num_workers // n_gpus, pin_memory=False)

    with cuda.device(process_id):
        model.cuda()
        gcam = GradCAM(model=model, candidate_layers=["spotlight_stage4", "compensation_stage4"])
        for iter, pack in enumerate(data_loader):
            img_name = pack['name'][0]
            label = pack['label'][0]
            size = pack['size']

            strided_size = imutils.get_strided_size(size, 4)
            strided_up_size = imutils.get_strided_up_size(size, 16)

            spotlight_cams = []
            compensation_cams = []
            num_classes = len(list(torch.nonzero(pack['label'][0])[:, 0]))

            # generate spotlight cam(spotlight branch)
            # every image has 7 scales (1.0, 0.5, 0.75, 1.25, 1.5, 1.75, 2.0)
            for s_count, size_idx in enumerate([1, 2, 0, 3, 4, 5, 6]):
                orig_img = pack['img'][size_idx].clone()
                for c_idx, c in enumerate(list(torch.nonzero(pack['label'][0])[:, 0])):
                    pack['img'][size_idx] = orig_img

                    # To read image before flip
                    single_size_img = pack['img'][size_idx].detach()[0]
                    if size_idx != 1:
                        total_adv_iter = args.adv_iter
                    else:
                        if args.adv_iter > 10:
                            total_adv_iter = args.adv_iter // 2
                            mul_for_scale = 2
                        elif args.adv_iter < 6:
                            total_adv_iter = args.adv_iter
                            mul_for_scale = 1
                        else:
                            total_adv_iter = 5
                            mul_for_scale = float(total_adv_iter) / 5

                    # start to use gradCAM to refine CAM
                    for it in range(total_adv_iter):
                        single_size_img.requires_grad = True

                        outputs = gcam.forward(single_size_img.cuda(non_blocking=True), step=1)

                        # generate initial CAM
                        if c_idx == 0 and it == 0:
                            cam_all_classes = torch.zeros([num_classes, outputs.shape[2], outputs.shape[3]])

                        gcam.backward(ids=c)

                        # generate gradCAM
                        regions = gcam.generate(target_layer="spotlight_stage4")
                        regions = regions[0] + regions[1].flip(-1)

                        if it == 0:
                            init_cam = regions.detach()

                        cam_all_classes[c_idx] += regions[0].data.cpu() * mul_for_scale
                        logit = outputs
                        logit = F.relu(logit)
                        logit = torchutils.gap2d(logit, keepdims=True)[:, :, 0, 0]

                        valid_cat = torch.nonzero(pack['label'][0])[:, 0]
                        logit_loss = - 2 * (logit[:, c]).sum() + torch.sum(logit)

                        expanded_mask = torch.zeros(regions.shape)
                        expanded_mask = add_discriminative(expanded_mask, regions, score_th=args.score_th)

                        # compute the loss between initial CAM and the latest CAM
                        L_AD = torch.sum((torch.abs(regions - init_cam)) * expanded_mask.cuda())

                        # compute total loss
                        loss = - logit_loss - L_AD * args.AD_coeff

                        model.zero_grad()
                        single_size_img.grad.zero_()
                        loss.backward()

                        data_grad = single_size_img.grad.data

                        perturbed_data = adv_climb(single_size_img, args.AD_stepsize, data_grad)
                        single_size_img = perturbed_data.detach()
                # store CAMs of all scales image
                spotlight_cams.append(cam_all_classes)
            # compute final spotlight cam
            strided_spotlight_cam = torch.sum(torch.stack(
                [F.interpolate(torch.unsqueeze(o, 0), strided_size, mode='bilinear', align_corners=False)[0] for o
                 in spotlight_cams]), 0)
            highres_spotlight_cam = [F.interpolate(torch.unsqueeze(o, 1), strided_up_size,
                                          mode='bilinear', align_corners=False) for o in spotlight_cams]

            highres_spotlight_cam = torch.sum(torch.stack(highres_spotlight_cam, 0), 0)[:, 0, :size[0], :size[1]]
            strided_spotlight_cam /= F.adaptive_max_pool2d(strided_spotlight_cam, (1, 1)) + 1e-5
            highres_spotlight_cam /= F.adaptive_max_pool2d(highres_spotlight_cam, (1, 1)) + 1e-5

            # generate compensation cam(compensation branch)
            # every image has 7 scales (1.0, 0.5, 0.75, 1.25, 1.5, 1.75, 2.0)
            for s_count, size_idx in enumerate([1, 2, 0, 3, 4, 5, 6]):
                orig_img = pack['img'][size_idx].clone()
                for c_idx, c in enumerate(list(torch.nonzero(pack['label'][0])[:, 0])):
                    pack['img'][size_idx] = orig_img

                    # To read image before flip
                    single_size_img = pack['img'][size_idx].detach()[0]
                    if size_idx != 1:
                        total_adv_iter = args.adv_iter
                    else:
                        if args.adv_iter > 10:
                            total_adv_iter = args.adv_iter // 2
                            mul_for_scale = 2
                        elif args.adv_iter < 6:
                            total_adv_iter = args.adv_iter
                            mul_for_scale = 1
                        else:
                            total_adv_iter = 5
                            mul_for_scale = float(total_adv_iter) / 5

                    # start to use gradCAM to refine CAM
                    for it in range(total_adv_iter):
                        single_size_img.requires_grad = True

                        outputs = gcam.forward(single_size_img.cuda(non_blocking=True), step=2)

                        # generate initial CAM
                        if c_idx == 0 and it == 0:
                            cam_all_classes = torch.zeros([num_classes, outputs.shape[2], outputs.shape[3]])

                        gcam.backward(ids=c)

                        # generate gradCAM
                        regions = gcam.generate(target_layer="compensation_stage4")
                        regions = regions[0] + regions[1].flip(-1)

                        if it == 0:
                            init_cam = regions.detach()

                        cam_all_classes[c_idx] += regions[0].data.cpu() * mul_for_scale
                        logit = outputs
                        logit = F.relu(logit)
                        logit = torchutils.gap2d(logit, keepdims=True)[:, :, 0, 0]

                        valid_cat = torch.nonzero(pack['label'][0])[:, 0]
                        logit_loss = - 2 * (logit[:, c]).sum() + torch.sum(logit)

                        expanded_mask = torch.zeros(regions.shape)
                        expanded_mask = add_discriminative(expanded_mask, regions, score_th=args.score_th)

                        # compute the loss between initial CAM and the latest CAM
                        L_AD = torch.sum((torch.abs(regions - init_cam)) * expanded_mask.cuda())

                        # compute total loss
                        loss = - logit_loss - L_AD * args.AD_coeff

                        model.zero_grad()
                        single_size_img.grad.zero_()
                        loss.backward()

                        data_grad = single_size_img.grad.data

                        perturbed_data = adv_climb(single_size_img, args.AD_stepsize, data_grad)
                        single_size_img = perturbed_data.detach()
                # store CAMs of all scales image
                compensation_cams.append(cam_all_classes)
            # compute final spotlight cam
            strided_compensation_cam = torch.sum(torch.stack(
                [F.interpolate(torch.unsqueeze(o, 0), strided_size, mode='bilinear', align_corners=False)[0] for o
                 in compensation_cams]), 0)
            highres_compensation_cam = [F.interpolate(torch.unsqueeze(o, 1), strided_up_size,
                                          mode='bilinear', align_corners=False) for o in compensation_cams]

            highres_compensation_cam = torch.sum(torch.stack(highres_compensation_cam, 0), 0)[:, 0, :size[0], :size[1]]
            strided_compensation_cam /= F.adaptive_max_pool2d(strided_compensation_cam, (1, 1)) + 1e-5
            highres_compensation_cam /= F.adaptive_max_pool2d(highres_compensation_cam, (1, 1)) + 1e-5

            # generate final weighted cam
            strided_weighted_cam = strided_spotlight_cam * args.weight + strided_compensation_cam * (1 - args.weight)
            highres_weighted_cam = highres_spotlight_cam * args.weight + highres_compensation_cam * (1 - args.weight)

            # save final weighted cam
            np.save(os.path.join(args.cam_out_dir, img_name + '.npy'),
                    {"keys": valid_cat, "cam": strided_weighted_cam.cpu(), "high_res": highres_weighted_cam.cpu().numpy()})

            if process_id == n_gpus - 1 and iter % (len(databin) // 20) == 0:
                print("%d " % ((5 * iter + 1) // (len(databin) // 20)), end='')

def run(cfg, args):
    model = getattr(importlib.import_module(args.amr_network), 'CAM')()
    model.load_state_dict(torch.load(args.amr_weights_name), strict=True)
    model.eval()
    n_gpus = torch.cuda.device_count()

    dataset = voc12.dataloader.VOC12ClassificationDatasetMSF(args.infer_list, voc12_root=args.voc12_root,
                                                             scales=args.cam_scales)
    dataset = torchutils.split_dataset(dataset, n_gpus)

    print('[ ', end='')
    multiprocessing.spawn(_work, nprocs=n_gpus, args=(model, dataset, args), join=True)
    print(']')

    torch.cuda.empty_cache()
