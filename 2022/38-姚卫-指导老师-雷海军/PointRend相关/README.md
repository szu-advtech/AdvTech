PointRend运行指令： 

```
python PointRend相关/projects/PointRend/train_net.py --num-gpus 1 --config-file PointRend相关/projects/PointRend/configs/InstanceSegmentation/implicit_pointrend_R_50_FPN_1x_coco.yaml SOLVER.IMS_PER_BATCH 1 SOLVER.BASE_LR 0.0012
```
PointSup运行指令：
```
python PointRend相关/projects/PointSup/train_net.py --num-gpus 1 --config-file PointRend相关/projects/PointSup/configs/mask_rcnn_R_50_FPN_3x_point_sup_coco.yaml SOLVER.IMS_PER_BATCH 1 SOLVER.BASE_LR 0.0012
```
预测直接运行PointRend中的demo.py即可

训练时需要自行下载COCO2017数据集，并放在dataset下

部分引用见下
```BibTeX
@misc{wu2019detectron2,
  author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                  Wan-Yen Lo and Ross Girshick},
  title =        {Detectron2},
  howpublished = {\url{https://github.com/facebookresearch/detectron2}},
  year =         {2019}
}
```