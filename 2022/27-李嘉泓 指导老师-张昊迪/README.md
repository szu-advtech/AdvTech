# 原型病人



## 使用方法

Install requirements: `pip install -r requirements.txt`

为了训练PtP，我们要把达到最好变现的参数列在了下面，分别为Proto和Bert的两个模型：

因为数据不能开源的问题，此处不附带数据集，可以按照论文提到的方法自制。

```
python training.py 
            --model_type PROTO
            --train_file {TRAIN.csv}
            --val_file {VAL.csv}
            --test_file {TEST.csv}
            --num_warmup_steps 5000
            --num_training_steps 5000
            --lr_features 0.000005
            --lr_prototypes 0.001
            --lr_others 0.001
            --use_attention True
            --reduce_hidden_size 256
            --all_labels_path {ALL_LABELS.txt}
```


```
python training.py 
            --model_type BERT
            --train_file {TRAIN.csv}
            --val_file {VAL.csv}
            --test_file {TEST.csv}
            --num_warmup_steps 1000
            --num_training_steps 5000
            --lr_features 0.00005
            --all_labels_path {ALL_LABELS.txt}
```
