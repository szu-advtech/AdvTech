# coding=utf-8
import string
from copy import deepcopy

import numpy.random as rand

import rl_config
from dafsa import DAFSA
from utils import Utils

import itertools

class StructuredTextLearner:
#lines第一个字符串
    def __init__(self, lines):
        self.lines = lines
        #铰链 ds [, , , , , , , ] regex [':', ':', ':', ':', ':', ':', ':', '']
        self.hinges = self.find_hinges() #返回[':', ':', ':', ':', ':', ':', ':']
        # Creating a list of DAFSA objects.
        self.ds = [DAFSA() for _ in range(len(self.hinges) + 1)]
        self.last_state = None
        self.regex = None
        self.learned_lines = deepcopy(lines)

    # Used for performance plots.
    def get_num_hinges(self):
        return len(self.hinges)

    def random_learned_string(self):
        """
        它从已学习的行列表中返回一个随机行
:返回:learned_lines列表的第一行。
        """
        return self.learned_lines[0]

    def generate_random_strings(self, n):
        """
        对于我们想要生成的n个字符串中的每一个，我们遍历分布列表，生成一个字母从每个分布，然后添加适当的铰链到字符串
:param n:要生成的字符串的个数
        """
        for _ in range(n):
            cur_string = ""
            for j in range(len(self.ds)):
                cur_string += self.ds[j].generate_letters()
                cur_string += (self.hinges + [""])[j]
            yield cur_string

    def generate_mh_strings(self):
        """
        对于每个字典中有效单词的每个可能组合，我们通过连接单词生成一个字符串，中间夹着铰链
        """
        list_of_valids = []
        for i in range(len(self.ds)):
            list_of_valids.append(self.ds[i].all_valid_words())
        for valid in itertools.product(*list_of_valids):
            curr_string = ""
            for j in range(len(valid)):
                curr_string += valid[j]
                curr_string += (self.hinges + [""])[j]
            yield curr_string

#返回给定字符串生成的模式
    def learn_string(self, line):
        """
        对于每个铰链，将字符串拆分为两个部分，并将第一部分添加到相应的数据结构中
:参数行:要学习的字符串
        """
        self.learned_lines.append(line)
        current_line = line
        split_string = []
        for h in self.hinges:
            split_line = current_line.split(h)
            split_string.append(split_line[0])
            current_line = current_line[len(split_line[0]) + 1:]
        split_string.append(current_line)
        #split_string为除去分隔符的所有字符数组【beb8,d4e3,e3w3】
        [self.ds[i].add_word(split_string[i]) for i in range(len(self.ds))]

    def open_for_learning(self):
        for d in self.ds:
            d.allow_adding()

    def compute_regex(self):
        """
        它获取数字列表和铰链列表，并将它们组合成字符串列表
        """
        new_hinges = self.hinges + ['']
        self.regex = [str(self.ds[i]) + new_hinges[i]
                      for i in range(len(self.ds))]

    def learn_deep_structure(self):
        """
        它接受一个字符串列表，并返回与这些字符串匹配的正则表达式列表
:return:正则表达式正在返回。
        """
        #['.', '.', '.']
        hinges = self.hinges
        #'252.241.217.144' regex['cDDc:', 'DcDD:', 'ccDc:', 'DDDD:', 'cDDc:', 'cDcD:', 'DDcD:', 'DcDc']
        for line in self.lines:
            self.learn_string(line)
        new_hinges = hinges + ['']
        self.regex = [str(self.ds[i]) + new_hinges[i]
                      for i in range(len(self.ds))]
        return True
#返回[':', ':', ':', ':', ':', ':', ':']
    def find_hinges(self):
        """
        它有一个字符串列表，并返回这些字符串中最常见的标点字符的列表
:return:字符串列表。
        """
        n = rl_config.get("default_sample_size")(self.lines) + 1
        #rand.choice从序列中获取一个随机元素  所有的空白字符和标点字符 hinge_listjoin所有标点符号
        doc_strings = list(rand.choice(self.lines, n, replace=False))
        candidates = list(string.whitespace + string.punctuation)
        hinge_list = [''.join([x for x in ds if x in candidates])
                      for ds in doc_strings] + ['']
        hinges = list(Utils.mode(hinge_list))#获取list中出现频数最多的元素 使用集合 众数
        return hinges

#期望的分数
    def feed_forward_score(self, lines=None, num_lines=None):
        """
        它接受标记列表并将其转换为字符串。对于每一行，使用铰链将该行拆分为字符串列表，然后将每个字符串输入到相应的字符串中
判别器，并对结果求和
:param lines:字符串列表
:param num_lines:从文件中读取的行数
:return:行平均得分。
:return:正则表达式的字符串表示形式。
        """
        hinges = self.hinges
        s = 0
        avg_line_length = 0
        for line in lines:
            line = str(line)
            avg_line_length += len(line)
            current_line = line
            split_string = []
            for h in hinges:
                split_line = current_line.split(h)
                split_string.append(split_line[0])
                current_line = current_line[len(split_line[0]) + 1:]
            split_string.append(current_line)
            s += float(sum([self.ds[i].feed_forward(split_string[i])
                            for i in range(len(self.ds))]))
        s /= avg_line_length
        return s

    # String representation字符串表示形式
    def __str__(self):
        """
        它接受令牌列表并将其转换为字符串
:return:正则表达式的字符串表示形式。
        """
        self.compute_regex()
        return ''.join(self.regex)
