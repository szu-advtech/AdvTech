import os
import cv2
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from predictor import VisualizationDemo
# from projects.PointRend.point_rend import add_pointrend_config
from point_sup import add_point_sup_config
# constants
WINDOW_NAME = "COCO detections"


def setup_cfg():
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_point_sup_config(cfg)
    cfg.merge_from_file(
        "./configs/mask_rcnn_R_50_FPN_3x_point_sup_point_aug_coco.yaml")
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    cfg.MODEL.WEIGHTS = os.path.join("./model_0199999.pth")  # 你下载的模型地址和名称
    cfg.freeze()
    return cfg


if __name__ == '__main__':
    cfg = setup_cfg()
    detectron_out = VisualizationDemo(cfg)
    path = r'../../ceshi/test3.jpg'  # 对应测试图片的地址
    outpath = r'../../outputs1'  # 对应保存测试结果的地址
    img = read_image(path, format="BGR")
    predictions, visualized_output = detectron_out.run_on_image(img)
    out_img = visualized_output.get_image()[:, :, ::-1]
    out_path = os.path.join(outpath, os.path.basename(path))
    visualized_output.save(out_path)