import argparse
import os
import torch
from config import cfg
from misc.misc import mkdir
from net import resnet50_amr

if __name__ == "__main__":

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')


    parser = argparse.ArgumentParser(description='ResNet50_AMR Training With Pytorch')

    # os.environ['CUDA_VISIBLE_DEVICES'] = '5'

    # Configure file
    parser.add_argument(
        "--config-file",
        default="/data2/xiepuxuan/code/AMR/config/amr_voc2012.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )

    # Train step
    parser.add_argument('--log_step', default=10, type=int, help='Print logs every log_step')
    parser.add_argument('--save_step', default=1000, type=int, help='Save checkpoint every save_step')
    parser.add_argument('--eval_step', default=1000, type=int,
                        help='Evaluate dataset every eval_step, disabled when eval_step < 0')

    # Environment
    parser.add_argument('--num_epochs', default=20, type=int, help='the number of epochs of training')
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    # Dataset
    parser.add_argument("--train_list", default="/data2/xiepuxuan/dataset/VOC2012/ImageSets/Main/train_aug.txt", type=str)
    parser.add_argument("--val_list", default="/data2/xiepuxuan/dataset/VOC2012/ImageSets/Main/val.txt", type=str)
    parser.add_argument("--visual_cam_list", default="/data2/xiepuxuan/code/AMR/try_cam.txt", type=str)
    parser.add_argument("--infer_list", default="/data2/xiepuxuan/dataset/VOC2012/ImageSets/Main/val.txt", type=str,
                        help="voc12/train_aug.txt to train a fully supervised model, "
                             "voc12/train.txt or voc12/val.txt to quickly check the quality of the labels.")
    parser.add_argument("--eval_cam_list", default="/data2/xiepuxuan/dataset/VOC2012/ImageSets/Main/eval_cam_list.txt", type=str)
    parser.add_argument("--chainer_eval_set", default="train", type=str)
    parser.add_argument("--voc12_root", default='/data2/xiepuxuan/dataset/VOC2012/', type=str,
                        help="Path to VOC 2012 Devkit, must contain ./JPEGImages as subdirectory.")
    parser.add_argument("--batch_size", default=16, type=int, help="the train batch size")
    parser.add_argument("--num_workers", default=32, type=int)

    # CAM
    parser.add_argument("--amr_network", default="net.resnet50_amr", type=str)
    parser.add_argument("--cam_network", default="net.resnet50_cam", type=str)
    parser.add_argument("--cam_weights_name", default="/data2/xiepuxuan/code/AMR/res50_cam.pth", type=str)
    parser.add_argument("--cam_scales", default=(1.0, 0.5, 0.75, 1.25, 1.5, 1.75, 2.0),
                        help="Multi-scale inferences")
    parser.add_argument("--cam_out_dir", default="/data2/xiepuxuan/code/AMR/outputs/amr_voc2012_2/cam_outputs", type=str)
    parser.add_argument("--ir_label_out_dir", default="/data2/xiepuxuan/code/AMR/outputs/amr_voc2012/ir_label_outputs", type=str)
    parser.add_argument("--amr_weights_name", default="/data2/xiepuxuan/code/AMR/final_model2.pth", type=str)
    parser.add_argument("--target_layer", default="spotlight_stage4")
    parser.add_argument("--adv_iter", default=27, type=int)
    parser.add_argument("--AD_coeff", default=7, type=int)
    parser.add_argument("--AD_stepsize", default=0.08, type=float)
    parser.add_argument("--score_th", default=0.5, type=float)
    parser.add_argument("--weight", default=0.5, type=float)
    parser.add_argument("--cam_eval_thres", default=0.20, type=float)

    parser.add_argument("--conf_fg_thres", default=0.60, type=float)
    parser.add_argument("--conf_bg_thres", default=0.33, type=float)

    # Step
    parser.add_argument("--train_amr", type=str2bool, default=False)
    parser.add_argument("--train_cam", type=str2bool, default=False)
    parser.add_argument("--make_cam", type=str2bool, default=False)
    parser.add_argument("--eval_cam", type=str2bool, default=False)
    parser.add_argument("--cam_to_ir_label", type=str2bool, default=False)

    args = parser.parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.num_gpus = num_gpus
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # make output directory
    if cfg.OUTPUT_DIR:
        mkdir(cfg.OUTPUT_DIR)

    # Step
    if args.train_amr:
        import step.train_amr
        step.train_amr.run(cfg, args)

    if args.make_cam:
        import step.make_cam
        step.make_cam.run(cfg, args)

    if args.eval_cam:
        import step.eval_cam
        step.eval_cam.run(args)

    if args.cam_to_ir_label:
        import step.cam_to_ir_label
        step.cam_to_ir_label.run(args)
