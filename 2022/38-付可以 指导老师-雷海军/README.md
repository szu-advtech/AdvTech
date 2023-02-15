# 一、实验环境

- matlab：9.13.0
- matlab-Image Processing Toolbox：11.6
- matplotlib：3.3.4
- opencv：4.6.0
- numpy：1.16.6
- PIL：8.4.0
- tensorflow：1.15.0
- tensorflow-gpu：1.15.0
- vgg：19
- cuda：10.0
- cudnn：7.6
- nvidia-driver：516.94
- python：3.6.5

# 二、训练与测试

### 1、Test

- 1、克隆我的项目到你的本地，并解压缩打开
- 2、通过 Readme.md 文件中提供的路径，下载百度网盘中的 checkpoint 参数压缩文件，并将 其解压缩到项目根目录下
- 3、将你需要增强的图像放在一个 test-real 文件夹中
- 4、打开 maintest.py 文件，将 is-train 设置为 false，并根据你的具体路径对 test-datadir 和 sample-dir 进行修改
- 5、运行 python maintest.py，完成后就可以在 sample-dir 看到结果了

### 2、Train

- 1、克隆我的项目到你的本地，并解压缩打开
- 2、通过 Readme.md 文件中提供的路径，下载百度网盘中的 vgg-pretrained 参数压缩文件、 UIEB-890 数据集，并将其解压缩到合适的地方
- 3、打开 maintrain.py 文件，对参数（训练批次 epoch、批量大小 batch-size、原始图像大小 image-height 和 image-width、参考图像大小 label-height 进而 label-width、学习率）进行修改
- 4、通过我提供的 matlab/python 文件，修改相关参数（如 resize 后的大小、三种增强后的存 储路径等），对 UIEB-890 或你自己的数据集进行预处理，得到四个文件夹分别保存 resize 后的 原始、WBP、CLAHE、GC，然后将对应的参考图像也做 resize 处理。需要特别注意的是，经过 我的测试，由于该网络层数较少且数据集较小，若设置较大的训练批次 epoch 会产生极大的过 拟合，这里如果你使用的是 UIEB 进行训练，我建议 epoch 设置为 100 就好。
- 5、将得到的四种不同的图像分别放在根目录的 input-train、input-wb-train、input-ce-train、 input-gc-train 和 gt-train【需要注意的是：如果你有验证集，也需要进行同样的处理，并分别放到 input-test、input-wb-test、input-ce-test、input-gc-test-test；如果没有验证集，也需要建立这五个文 件夹为空即可】
- 6、运行 python maintrain.py，完成后就可以在 checkpoint/coarse-112 中看到训练得到的参数 文件了

# 三、一些文件下载

- UIEB：https://pan.baidu.com/s/19kVNE7P9K89QTvKoa2HG5A	提取码：kyuy
- RUIE：https://pan.baidu.com/s/1wz5Ba7Wb4CDmk6TY5SMJrg       提取码：tp9m
- checkpoint：https://pan.baidu.com/s/1Bpe5hfQaf6IVbhhL6BH3lQ     提取码：aqhr
- vgg_pretrained：https://pan.baidu.com/s/1uCFDfDIPfbr2qByXwNaQDQ     提取码：esh3
- input_train：https://pan.baidu.com/s/1Zisgfq3TW9nPBN1y6ZOiAA     提取码：0dup
