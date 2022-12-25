"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    test_pub.py
# Abstract       :    Script for inference

# Current Version:    1.0.1
# Date           :    2022-05-12
##################################################################################################
"""

import cv2
import mmcv
import json
import jsonlines
import numpy as np
from tqdm import tqdm
from davarocr.davar_table.utils import TEDS, format_html
from davarocr.davar_common.apis import inference_model, init_model

from mmdet.apis import init_detector
from mmocr.apis.inference import model_inference
from mmocr.datasets.pipelines.crop import crop_img

import re

def remove_html_tag(content):
    """
    This function will remove the html tag, eg:'<b>' and '</b>' of the content.
    :param content: [str]. text content of each cell.
    :return: text content without html tag, eg:'<b>' and '</b>'.
    """
    rmString = '</?[a-zA-Z]*>'
    return re.sub(rmString, '', content)
# def obtain_ocr_results(img_path, model_textdet, model_rcg):
#     """obtrain ocr results of table.
#     """

#     def crop_from_bboxes(img, bboxes, expand_pixels=(0, 0, 0, 0)):
#         """crop images from original images for recognition model
#         """
#         ret_list = []
#         for bbox in bboxes:
#             max_x, max_y = min(img.shape[1], bbox[2] + expand_pixels[3]), min(img.shape[0], bbox[3] + expand_pixels[1])
#             min_x, min_y = max(0, bbox[0] - expand_pixels[2]), max(0, bbox[1] - expand_pixels[0])
#             if len(img.shape) == 2:
#                 crop_img = img[min_y: max_y, min_x: max_x]
#             else:
#                 crop_img = img[min_y: max_y, min_x: max_x, :]
#             ret_list.append(crop_img)

#         return ret_list

#     ocr_result = {'bboxes': [], 'confidence': [], 'texts': []}

#     # single-line text detection
#     text_bbox, text_mask = inference_model(model_textdet, img_path)[0]
#     text_bbox = text_bbox[0]
#     for box_id in range(text_bbox.shape[0]):
#         score = text_bbox[box_id, 4]
#         box = [int(cord) for cord in text_bbox[box_id, :4]]
#         ocr_result['bboxes'].append(box)
#         ocr_result['confidence'].append(score)

#     # single-line text recognition
#     origin_img = mmcv.imread(img_path, cv2.IMREAD_IGNORE_ORIENTATION + cv2.IMREAD_COLOR)
#     origin_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY)
#     cropped_img = crop_from_bboxes(origin_img, ocr_result['bboxes'], expand_pixels=(1, 1, 3, 3))
#     rcg_output = inference_model(model_rcg, cropped_img)
#     ocr_result['texts'] = rcg_output['text']

#     return ocr_result

def obtain_ocr_results(img_path, model_textdet, model_rcg):
    """obtrain ocr results of table.
    """
    expand_pixels=(1, 1, 3, 3)
    ocr_result = {'bboxes': [], 'confidence': [], 'texts': []}

    image = mmcv.imread(img_path)
    det_result = model_inference(model_textdet, image)
    bboxes = det_result['boundary_result']

    for bbox in bboxes:
        score = float(bbox[-1])
        ocr_result['confidence'].append(score)
        box = bbox[:8]
        if len(bbox) > 9:
            min_x = min(bbox[0:-1:2])
            min_y = min(bbox[1:-1:2])
            max_x = max(bbox[0:-1:2])
            max_y = max(bbox[1:-1:2])
            box = [min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y]
        cor_xs = [box[i] for i in range(0,len(box),2)]
        cor_ys = [box[i] for i in range(1,len(box),2)]
        davar_box = [int(min(cor_xs)), int(min(cor_ys)), int(max(cor_xs)), int(max(cor_ys))]
        ocr_result['bboxes'].append(davar_box)

        max_x, max_y = min(image.shape[1], davar_box[2] + expand_pixels[3]), min(image.shape[0], davar_box[3] + expand_pixels[1])
        min_x, min_y = max(0, davar_box[0] - expand_pixels[2]), max(0, davar_box[1] - expand_pixels[0])
        box = [min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y]

        box_img = crop_img(image, box)
        recog_result = model_inference(model_rcg, box_img)
        text = recog_result['text']
        ocr_result['texts'].append(remove_html_tag(text))

    return ocr_result

def obtain_ocr_results_mp(img_path, model_textdet, model_rcg ,batch_size, expand_pixels=(0, 0, 0, 0)):
    """obtrain ocr results of table.
    """
    ocr_result = {'bboxes': [], 'confidence': [], 'texts': []}

    image = mmcv.imread(img_path)
    img = cv2.imread(img_path)
    # i = 0
    det_result = model_inference(model_textdet, image)
    bboxes = det_result['boundary_result']
    box_imgs = []
    
    for bbox in bboxes:
        score = float(bbox[-1])
        ocr_result['confidence'].append(score)
        box = bbox[:8]
        if len(bbox) > 9:
            min_x = min(bbox[0:-1:2])
            min_y = min(bbox[1:-1:2])
            max_x = max(bbox[0:-1:2])
            max_y = max(bbox[1:-1:2])
            box = [min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y]
        cor_xs = [box[i] for i in range(0,len(box),2)]
        cor_ys = [box[i] for i in range(1,len(box),2)]
        min_corx = min(cor_xs)
        min_cory = min(cor_ys)
        max_corx = max(cor_xs)
        max_cory = max(cor_ys)
        davar_box = [int(round(min_corx,0)), int(round(min_cory,0)), int(round(max_corx,0)), int(round(max_cory,0))]
        # davar_box = [int(min(cor_xs)), int(min(cor_ys)), int(max(cor_xs)), int(max(cor_ys))]
        ocr_result['bboxes'].append(davar_box)

        max_x, max_y = min(image.shape[1], max_corx + expand_pixels[3]), min(image.shape[0], max_cory + expand_pixels[1])
        min_x, min_y = max(0, min_corx - expand_pixels[2]), max(0, min_cory - expand_pixels[0])
        box = [min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y]
        box_img = crop_img(image, box)
        box_imgs.append(box_img)
    #     cv2.imwrite('/root/zf/DAVAR-Lab-OCR/output/crop/'+str(i)+'.jpg', box_img)
    #     i += 1
        for j in range(0, len(box), 2):
            cv2.line(img, (int(round(box[j],0)), int(round(box[j+1],0))), (int(round(box[(j + 2) % len(box)],0)), int(round(box[(j + 3) % len(box)],0))), (0, 255, 0), 1)
        
    cv2.imwrite('/root/zf/DAVAR-Lab-OCR/output/crop/result.jpg', img)
    
    for chunk_idx in range(len(box_imgs) // batch_size + 1):
        start_idx = chunk_idx * batch_size
        end_idx = (chunk_idx + 1) * batch_size
        chunk_box_imgs = box_imgs[start_idx:end_idx]
        if len(chunk_box_imgs) == 0:
            continue
        recog_results = model_inference(model_rcg, chunk_box_imgs, batch_mode=True)
        for i, recog_result in enumerate(recog_results):
            text = recog_result['text']
            ocr_result['texts'].append(remove_html_tag(text))

    return ocr_result


# test setting
# do_visualize = 0  # whether to visualize
do_visualize = 0  # whether to visualize
evaluation_structure_only = False  # only evaluate structure or evaluate structure with ocr results
# vis_dir = "/path/to/save/visualization"
# savepath = "/path/to/save/prediction"
vis_dir = "/root/zf/DAVAR-Lab-OCR/output/images/"
pred_result_savepath = "/root/zf/DAVAR-Lab-OCR/output/pred_result.json"

# LGPMA setting
# config_lgpma = '/path/to/configs/lgpma_pub.py'
# checkpoint_lgpma = 'path/to/lgpma_checkpoint'
config_lgpma = '/root/zf/DAVAR-Lab-OCR/demo/table_recognition/lgpma/configs/lgpma_pub.py'
checkpoint_lgpma = '/root/zf/DAVAR-Lab-OCR/model/maskrcnn-lgpma-pub-e12-pub.pth'
model_lgpma = init_model(config_lgpma, checkpoint_lgpma, device='cuda:2')

# OCR model setting.
# config_det = '/path/to/configs/ocr_models/det_mask_rcnn_r50_fpn_pubtabnet.py'
# checkpoint_det = '/path/to/text_detection_checkpoint'
# config_rcg = '/path/to/configs/ocr_models/rcg_res32_bilstm_attn_pubtabnet_sensitive.py'
# checkpoint_rcg = '/path/to/text_recognition_checkpoint'
config_det = '/root/zf/TableMASTER-mmocr/configs/textdet/psenet/psenet_r50_fpnf_600e_pubtabnet.py'
checkpoint_det = '/root/zf/TableMASTER-mmocr/work_dir/1210_PseNet_textdet/latest.pth'
config_rcg = '/root/zf/TableMASTER-mmocr/configs/textrecog/master/master_lmdb_ResnetExtra_tableRec_dataset_dynamic_mmfp16.py'
checkpoint_rcg = '/root/zf/TableMASTER-mmocr/work_dir/1123_MASTER_textline/latest.pth'
# model_det = init_model(config_det, checkpoint_det)
# if 'postprocess' in model_det.cfg['model']['test_cfg']:
#     model_det.cfg['model']['test_cfg'].pop('postprocess')
# model_rcg = init_model(config_rcg, checkpoint_rcg)

batch_size = 512

# build detect model
detect_model = init_detector(config_det, checkpoint_det, device='cuda:3')
if hasattr(detect_model, 'module'):
    detect_model = detect_model.module
if detect_model.cfg.data.test['type'] == 'ConcatDataset':
    detect_model.cfg.data.test.pipeline = \
        detect_model.cfg.data.test['datasets'][0].pipeline

# build recog model
recog_model = init_detector(config_rcg, checkpoint_rcg, device='cuda:4')
if hasattr(recog_model, 'module'):
    recog_model = recog_model.module
if recog_model.cfg.data.test['type'] == 'ConcatDataset':
    recog_model.cfg.data.test.pipeline = \
        recog_model.cfg.data.test['datasets'][0].pipeline

# getting image prefix and test dataset from config file
img_prefix = model_lgpma.cfg["data"]["test"]["img_prefix"]
test_dataset = model_lgpma.cfg["data"]["test"]["ann_file"]
with jsonlines.open(test_dataset, 'r') as fp:
    test_file = list(fp)

# generate prediction of html and save result to savepath
pred_result = dict()
pred_html = dict()
count = 0
end = 3
for sample in tqdm(test_file):
    if count < end:
        count += 1
        continue
    # print("deal with file {}".format(sample['filename']))
    img_path = img_prefix + sample['filename']
    cv2.imwrite(vis_dir + "origonal.jpg", cv2.imread(img_path))
    # The ocr results used here can be replaced with your results.
    result_ocr = obtain_ocr_results_mp(img_path, detect_model, recog_model, batch_size, expand_pixels=(2, 0.8, 3, 3))

    # predict result of table, including bboxes, labels, texts of each cell and html representing the table
    model_lgpma.cfg['model']['test_cfg']['postprocess']['ocr_result'] = [result_ocr]
    result_table = inference_model(model_lgpma, img_path)[0]
    pred_result[sample['filename']] = result_table
    pred_html[sample['filename']] = result_table['html']
    with open(pred_result_savepath, "w", encoding="utf-8") as writer:
        json.dump(pred_result, writer, ensure_ascii=False)

    # detection results visualization
    if do_visualize:
        img = cv2.imread(img_path)
        img_name = img_path.split("/")[-1]
        bboxes_non = [b for b in result_table['content_ann']['bboxes'] if len(b)]
        bboxes = [[b[0], b[1], b[2], b[1], b[2], b[3], b[0], b[3]] if len(b) == 4 else b for b in bboxes_non]
        for box in bboxes:
            for j in range(0, len(box), 2):
                cv2.line(img, (box[j], box[j + 1]), (box[(j + 2) % len(box)], box[(j + 3) % len(box)]), (0, 0, 255), 1)
        cv2.imwrite(vis_dir + img_name, img)
    # count += 1
    if count == end:
        break

with open(pred_result_savepath, "w", encoding="utf-8") as writer:
    json.dump(pred_result, writer, ensure_ascii=False)

# generate ground-truth of html from pubtabnet annotation of test dataset.
gt_dict = dict()
for data in test_file:
    if data['filename'] not in pred_html.keys():
        continue
    if evaluation_structure_only is False:
        tokens = data['html']['cells']
        for ind, item in enumerate(tokens):
            tok_nofont = [tok for tok in item['tokens'] if len(tok) <= 1]
            tok_valid = [tok for tok in tok_nofont if tok != ' ']
            tokens[ind]['tokens'] = tok_nofont if len(tok_valid) else []
        data['html']['cells'] = tokens
    gt_dict[data['filename']] = {'html': format_html(data)}

# with open("/root/zf/DAVAR-Lab-OCR/output/gt.json", "w", encoding="utf-8") as writer:
#     json.dump(gt_dict, writer, ensure_ascii=False)

# evaluation using script from PubTabNet
teds = TEDS(structure_only=evaluation_structure_only, n_jobs=16)
scores = teds.batch_evaluate(pred_html, gt_dict)
print(np.array(list(scores.values())).mean())
