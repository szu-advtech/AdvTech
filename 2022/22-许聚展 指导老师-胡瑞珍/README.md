
## 计算机前沿技术课程 复现工作

## 复现论文：Semantically Grounded Object Matching for Robust Robotic Scene Rearrangement [🧷](https://arxiv.org/abs/2111.07975)


### 只进行物体匹配：

> 环境依赖

    python >= 3.6
    pytorch >= 1.7.1
    clip (https://github.com/openai/CLIP)

> 核心代码

物体匹配的复现代码在`VM`（visual matching）文件夹下，其中[`matcher.py`](./VM/matcher.py)里面实现了物体匹配的算法，你可以参考`evaluate_lvis.py`。

```python
from VM.matcher import VisualMatcher

matcher = VisualMatcher()

source_list = xxx # sourse images
target_list = xxx # goal images
label_list = xxx # object labels
use_text = True

source_ids, target_ids = matcher.match_images( source_list, target_list, label_list, use_text )
```

> 测试数据集

请参考`VM/data/README.md`下载数据集后进行处理。


### 进行机器人重排列实验：

> 环境依赖

    python >= 3.6
    pyvista
    pytorch >= 1.7.1
    clip (https://github.com/openai/CLIP)
    
> 安装模拟环境（要求RTX显卡）：

omiverse isaac sim (https://developer.nvidia.com/isaac-sim)

> 核心代码

`robot`：内是isaac sim的模拟环境控制代码

`UOC`：修改自[https://github.com/NVlabs/UnseenObjectClustering](https://github.com/NVlabs/UnseenObjectClustering)的代码，我在里面写了个`app.py`，从中提取出了一个`Segmenter`类作为实例分割模块。使用前请下载好他们的训练权重，放到`UOC/data/checkpoints`下

`main.py`：机器人重排列的主代码

`ui.py`：设置isaac sim的界面代码

`run.bat`：执行`main.py`的命令


