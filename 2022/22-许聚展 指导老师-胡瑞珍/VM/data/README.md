
## 数据集

来自 LVIS
> https://www.lvisdataset.org/dataset

## split.py

来自网上，这个数据集的类别根据频率（frequency）分为3种：common、frequency、rare，这个代码就是把json文件中的3个给分开
> https://github.com/ucbdrive/few-shot-object-detection/blob/master/datasets/split_lvis_annotation.py


## organize_by_cat.py

因为我想随机采样特定类物体的图片，就将json的数据按照类别整合，结构大概如下
'''
    {
        'xxxx cat_id': {
            name:
            images: [
                xxx,
                xxx
            ]
            'bbox': [
                xxx,
                xxx
            ]
        }
    }
'''

然后我只把 common 类的物体取出来保存，都在 `split` 内