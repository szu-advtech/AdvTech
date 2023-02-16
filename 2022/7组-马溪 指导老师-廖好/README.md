# SoftProto
 Code and datasets of our paper "[Enhancing Aspect Term Extraction with Soft Prototypes](https://www.aclweb.org/anthology/2020.emnlp-main.164.pdf)" accepted by EMNLP 2020.


## 1. Requirements
* python 3.6.7
* pytorch 1.5.0
* pytorch-pretrained-bert 0.4.0
* numpy 1.19.1

## 2. Usage
 We incorporate the training and evaluation of SoftProto in **train_softproto.py**. Just run it as below. The ```--lm``` argument is used to specify the type of pre-trained language models.

```
CUDA_VISIBLE_DEVICES=0 python train_softproto.py --dataset res14 --lm external --seed 123
```

 Or you can run the shell script to get all results of a certain dataset.

```
sh res14_run.sh
```

## 3. Results
 Since we randomly split the training/validation datasets in the running process, the experimental results could vary on different machines. But if you run the shell script and collect the results of all settings, you'll find that the improvements brought by SoftProto are rather stable.

 We re-run the experiments on another machine which is different from the one used in our paper, and list the results as below. The corresponding log files are contained in the ```./log/``` folder.

 ![image](https://github.com/NLPWM-WHU/SoftProto/blob/master/result.jpg)

## 4. Data Details
 A separate set consists of the following files:

* **sentence.txt** contains the tokenized review sentences.
* **target.txt** contains the aspect term tag sequences. **0=O, 1=B, 2=I**.
* **internal\_forward/backward\_top10.txt** contains the top-10 oracle words in **SoftProtoI**.
* **external\_forward/backward\_top10.txt** contains the top-10 oracle words in **SoftProtoE**.
* **bert\_base\_top10.txt** contains the top-10 oracle words in **SoftProtoB (BASE)**.
* **bert\_pt\_top10.txt** contains the top-10 oracle words in **SoftProtoB (PT)**.

## 5. Generation of Oracle Words
 For generating oracle words using LM/MLM, we will release a script in a few days.

## 6. Citation
 If you find our code and datasets useful, please cite our paper.


```
@inproceedings{chen2020softproto,
  author    = {Zhuang Chen and Tieyun Qian},
  title     = {Enhancing Aspect Term Extraction with Soft Prototypes},
  booktitle = {EMNLP},
  pages     = {2107--2117},
  year      = {2020},
  url       = {https://www.aclweb.org/anthology/2020.emnlp-main.164/}
}
```

 :checkered_flag: 
