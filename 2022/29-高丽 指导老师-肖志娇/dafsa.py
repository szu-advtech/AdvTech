# coding=utf-8
import operator
import string

import numpy.random as rand
import numpy as np
from scipy.stats import chisquare

import rl_config
from utils import Utils

import itertools


class Layer:

    def __init__(self, node):
        self.compression_accuracy = rl_config.get("capture_percentage") #压缩准确性
        self.nodes = {node: 1}
        self.symbols = {Utils.get_class(node): [node]} #根据ASII返回代码字符 8-57 D 65-90 C 97-122 c
        self.total_count = 1
        self.optional = False
        self.compressed = None

    def compress(self):
        """
        如果最常见的符号是单个字符，而该字符不是特殊字符，则返回该字符
的性格。否则，返回通配符
:return:字符串中最常见的字符。
        """
        # Finding the most common symbol in the layer.
        symbol_pcts = {}
        # Compute percentages 计算百分比
        max_symbol = None
        max_symbol_pct = 0
        for symbol in self.symbols:
            symbol_pcts[symbol] = float(
                len(self.symbols[symbol])) / self.total_count
            if symbol_pcts[symbol] > max_symbol_pct:
                max_symbol = symbol
                max_symbol_pct = symbol_pcts[symbol]

        # Case 1: all the same character 情况1:都是相同的字符
        # Does the character itself matter? 字符本身重要吗?
        if max_symbol_pct > 0.9:
            is_invalid_character = (
                ord(max_symbol) < 48 or ord(max_symbol) > 122)
            if (is_invalid_character or
                    ord(max_symbol) == Utils.get_ord(max_symbol)):
                return max_symbol
            else:
                alphabet = [0] * 26
                if max_symbol == 'D':
                    alphabet = [0] * 10
                alphabet_dict = {}
                for c in self.symbols[max_symbol]:
                    try:
                        alphabet[Utils.get_ord(c)] += 1
                        alphabet_dict[c] = alphabet_dict.get(c, 0) + 1
                    except IndexError:
                        continue
                # Computing the p-value of the chi-squared test.
                p_val = chisquare(sorted(alphabet)).pvalue
                if not p_val <= 0.05:#10**(-10):
                    return max_symbol
                else:
                    comp_acc = 0.0
                    my_regex = ""
                    while comp_acc < self.compression_accuracy:
                        this_symb = max(alphabet_dict.items(),
                                        key=operator.itemgetter(1))[0]
                        my_regex += this_symb + "|"
                        comp_acc += float(alphabet_dict[this_symb]
                                          ) / float(sum(alphabet))
                        alphabet_dict[this_symb] = 0
                    re = my_regex[:-1]
                    if "|" in re:
                        return "(" + re + ")"
                    return re
        else:
            alphabet = [0] * 256
            for c in self.symbols[max_symbol]:
                    alphabet[Utils.get_ord(c)] += 1
            p_val = chisquare(alphabet).pvalue
            if p_val > 0.01:
                return max_symbol
            else:
                return "*"

    def acceptable_letters(self):
        """
        如果压缩字符串是'*'，则可接受的字母都是字母和数字。如果压缩字符串是
“D”，那么可接受的字母都是数字。如果压缩字符串为'C'，则可接受的字母为
全是大写字母。如果压缩字符串是'c'，那么可接受的字母都是小写字母。如果
压缩字符串包含'|'，那么可接受的字母是'|'之间的字符串。否则,
可接受的字母是压缩后的字符串本身
:return:可接受的字母列表。
        """
        compressed = self.compress()
        if compressed == '*':
            acceptable_letters = list(string.letters + string.digits)
        elif compressed == 'D':
            acceptable_letters = list(string.digits)
        elif compressed == 'C':
            acceptable_letters = list(string.ascii_uppercase)
        elif compressed == 'c':
            acceptable_letters = list(string.ascii_lowercase)
        elif "|" in compressed:
            acceptable_letters = compressed[1:-1].split("|")
        else:
            acceptable_letters = [compressed]
        return acceptable_letters


    def generate_letter(self):
        """
        它从可接受的字母列表中返回一个随机字母
:return: acceptable_letters列表中的随机字母。
        """
        return rand.choice(self.acceptable_letters())
#添加结点
    def add_node(self, node, dry_run=False):
        """
        它向图中添加一个节点，并返回一个值，指示该节点是否是新节点

:param node:要添加到树中的节点
:param dry_run:如果为True，则节点不被添加到树中，默认为False(可选)
:返回:添加到树中的新节点数。
        """
        added_new = 0
        if not dry_run:
            self.nodes[node] = self.nodes.get(node, 0) + 1
        if not Utils.get_class(node) in self.symbols:
            added_new = 1
        elif node not in self.nodes and not self.compress() in {'*', 'D', 'C', 'c'}:
            added_new = 0.5
        if not dry_run:
            self.symbols[Utils.get_class(node)] = self.symbols.get(
                Utils.get_class(node), []) + [node]
            self.total_count += 1
        return added_new

    def letter_reps(self):
        """
       它返回由正则表达式表示的字母列表
:return:字符串列表。
        """
        compressed = self.compress()
        if "|" in compressed:
            return compressed[1:-1].split("|")
        else: # compressed in {"*", "D", "C", "c"} or single char
            return [compressed]

    def __repr__(self):
        """
        如果正则表达式是可选的，则返回方括号中的正则表达式，否则返回正则表达式
:return:对象的字符串表示形式。
        """
        if self.optional:
            return "[" + self.compress() + "]"
        else:
            return self.compress()


class DAFSA():

    def __init__(self):
        """
        The function initializes the layers, score, historical score, done_adding, inc, std_score, n, needed_sample_size,
        mean_score, and old_hist_score
        """
        self.layers = []
        self.score = 0
        self.historical_score = [0] #历史分数
        self.done_adding = False
        self.inc = 10
        self.std_score = 100
        self.n = rl_config.config["clt_sample_size"]
        self.needed_sample_size = lambda x: (1.96 * x/0.5)**2
        self.mean_score = None
        self.old_hist_score = [0]


    def generate_letters(self):
        """
        它为每一层生成一个字母，然后将它们连接到一个字符串中
:return:字符串
        """
        letters_arr = [l.generate_letter() for l in self.layers]
        return ''.join(letters_arr)

#beb8 返回给定字符串的模式 layers里存转换后的值
    def add_word(self, word):
        """
        如果我们还没有完成添加单词，那么我们就将单词添加到层中，如果我们已经完成添加单词，那么就添加
跟各层人说一声，我们就完事了

:param word:要加入网络的单词
        """
        if not self.done_adding:
            counter = 0
            while counter < len(word):
                if len(self.layers) < counter + 1:
                    new_layer = Layer(word[counter])
                    self.layers.append(new_layer)
                else:
                    self.score += self.layers[counter].add_node(word[counter])
                counter += 1
            self.historical_score.append(self.score)
            if len(self.historical_score) % self.inc == 0:
                self.std_score = np.std(self.historical_score)#返回数组元素的标准差
            if self.needed_sample_size(self.std_score) < len(self.historical_score):
                self.done_adding = True
                self.historical_score = self.historical_score + self.old_hist_score
                self.mean_score = np.mean(self.historical_score)

    def allow_adding(self):
        """
        它将历史得分设置为0，并将done_adding变量设置为False
        """
        self.old_hist_score = self.historical_score
        self.historical_score = [0]
        self.done_adding = False

    def make_representation(self):
        """
        它接受一个神经网络，并返回该神经网络的字符串表示
:返回:压缩层的字符串。
        """
        my_string = ""
        for layer in self.layers:
            my_string += layer.compress()
        return my_string

    def __repr__(self):
        return self.make_representation()

    def __str__(self):
        return self.make_representation()

    def to_expression(self):
        return self.make_representation()

    def feed_forward(self, word):
        """
        >如果单词的长度小于层数，则每一层未使用的加1分

:参数word:要评分的单词
:return:单词的分数。
        """
        score = 0
        counter = 0
        while counter < len(word):
            if len(self.layers) < counter + 1:
                score += 1
            else:
                score += self.layers[counter].add_node(
                    word[counter], dry_run=True)
            counter += 1
        score += (1 if len(self.layers) > counter else 0)
        #score += max(len(self.layers) - counter, 0)
        return score

    def all_valid_words(self):
        """
        对于每一层，得到所有有效的字母表示，然后得到这些字母的所有可能的组合，
并返回由这些组合组成的所有单词的列表
:return:由层中的字母组成的所有可能的单词的列表。
        """
        valid_per_letter = [l.letter_reps() for l in self.layers]
        words = []
        for prod_string in itertools.product(*valid_per_letter):
            words.append(''.join(prod_string))
        return words
