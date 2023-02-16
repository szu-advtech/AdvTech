# coding=utf-8
import argparse
import numpy as np
import rstr
from random import shuffle
from main import learn_model
import string
from numpy.random import choice
from random import choice as rand_choice
import re
import matplotlib.pyplot as plt

def random_filler(N):
    """
    它生成一个长度为N的随机字符串，其中每个字符都是从所有小写字母集合中随机选择的，全是大写字母，全是数字

:param N:字符串中的字符数
:return:长度为N的字符串，由字符串中的随机字符组成。ascii_lowercase,字符串。ascii_uppercase,string.digits。
    """
    return ''.join(rand_choice(string.ascii_lowercase + string.ascii_uppercase + string.digits) for _ in range(N))

def randomly_mutate(string, n_bits):
    """
    它接受一个字符串和一些位来进行突变，并返回一个具有指定突变位数的新字符串

:param string:要修改的字符串
:param n_bits:要突变的比特数
:返回:长度为n_bits的字符串，其中n_bits被随机替换为随机位。
    """
    filler = random_filler(n_bits)
    indices_to_replace = set(choice(len(string), n_bits, replace=False))
    new_string = ""
    j = 0
    for i in range(len(string)):
        if i in indices_to_replace:
            new_string += filler[j]
            j+=1
        else:
            new_string += string[i]
    return new_string

def error_detection_experiment(n, base, r, m=2):
    """
    它需要一个正则表达式、一些要生成的字符串、一些要改变的字符串和一个阈值，并返回
模型在检测突变字符串时的准确性

:param n:要生成的正则表达式的个数
:param base:要学习的正则表达式
:param r:突变字符串与非突变字符串的比例
:param m:正则表达式的突变次数，默认为2(可选)
:return:模型的精度。
    """
    unshuffled_regex = [rstr.xeger(base) for i in range(n-int(n*r))] + [randomly_mutate(rstr.xeger(base), 2) for i in range(int(n*r))]
    shuffle(unshuffled_regex)
    m = learn_model(unshuffled_regex)["model"]
    re_pattern = re.compile(base)
    tp = 0.0
    tn = 0.0
    fp = 0.0
    fn = 0.0
    scores = [m.score_strings([s]) for s in unshuffled_regex]
    for i in range(len(unshuffled_regex)):
        if not re_pattern.match(unshuffled_regex[i]):
            if scores[i] > 0.4:
                tp += 1
            else:
                fn += 1
        else:
            if scores[i] > 0.4:
                fp += 1
            else:
                tn += 1
    return 2*tp/(2*tp+fn+fp)

def full_error_detect(n, base):
    """
    它对给定的位数、基数和错误数运行错误检测实验，并绘制结果

:param n:消息的比特数
:param base:编号系统的基数
    """
    for m in range(5):
        x = np.array(range(19))/20.0 + 0.05
        y = [error_detection_experiment(n, base, r,m+1) for r in x]
        plt.plot(x, y)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generates accuracy plots')
    parser.add_argument('base_regex', metavar="b", type=str, help="Baseline regex (hope to capture this)")
    parser.add_argument('--num_samples', metavar="n", type=int, help="Number of samples to generate")
    args = parser.parse_args()
    n = args.num_samples if args.num_samples else 100
    full_error_detect(n, args.base_regex)
