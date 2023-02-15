# coding=utf-8
import string
from random import shuffle

import distance #计算任意序列之间的相似性
import editdistance #编辑距离
import numpy as np
from scipy.cluster import hierarchy

import rl_config


class Utils:

    #获取list中出现频数最多的元素 使用集合
    @staticmethod
    def mode(lst):
        return max(set(lst), key=lst.count)

    @staticmethod
    def get_ord(symbol):
        """
        返回给定字符的数值。
        返回的值是字符在其所属类中的偏移量
        属于。具体地说:
        [:lower:]中的字符-> a = 0，…， z = 25
        character in [:upper:] -> A = 0，…， z = 25
        [:digit:]中的字符-> 1 = 1，…， 9 = 9
        任何其他字符->十进制ASCII码
        """
        #ord()方法作用是获取Unicode中字符的编码，ASCII编码中仅支持英文和特定的一些字符，并不支持中文，而Unicode编码中支持中文和很多的特殊字符，并且在Unicode编码中，中文和英文字母一样占一个字节
        ascii_code = ord(symbol)
        # digits
        if 48 <= ascii_code <= 57:
            return int(symbol)
        # uppercase alpha characters
        elif 65 <= ascii_code <= 90:
            return ascii_code - 65
        # lowercase alpha characters
        elif 97 <= ascii_code <= 122:
            return ascii_code - 97
        return ord(symbol)

    @staticmethod
    def get_char_class(char):
        """
        对象的简短文本表示形式
        字符类，使用以下映射:

       [:lower:]中的字符->字符串"(c*)"
       [:upper:]中的字符->字符串"(C*)"
       字符在[:digit:] ->字符串"(D*)"
       [:punct:]中的字符->原始字符
       [:space:]中的字符->字符串" "
       任何其他字符->原始字符
        """
        if char in string.ascii_lowercase:
            return "(c*)"
        elif char in string.ascii_uppercase:
            return "(C*)"
        elif char in string.digits:
            return "(D*)"
        elif char in string.whitespace: #字符串空白是在Python3的字符串模块中预定义的。如果包含空格，制表符，换行符，返回符，换页符和垂直制表符
            return " "
        elif char in string.punctuation: #返回所有标点符号。
            return char

#根据ASII返回代码字符 8-57 D 65-90 C 97-122 c
    @staticmethod
    def get_class(symbol):
        ascii_code = ord(symbol)
        if 8 <= ascii_code <= 57:
            return 'D'
        elif 65 <= ascii_code <= 90:
            return 'C'
        elif 97 <= ascii_code <= 122:
            return 'c'
        return symbol

    @staticmethod
    def get_char_class_ed(char):
        if char in string.ascii_lowercase:
            return "@"
        elif char in string.ascii_uppercase:
            return "#"
        elif char in string.digits:
            return "$"
        elif char in string.whitespace:
            return "^"
        elif char in string.punctuation:
            return char

            """
            它接受一个字符串并返回一个具有相同字符的字符串，但每个字符类之间有一个空格
    :param string:要处理的字符串
    :return:输入字符串的字符类的字符串。
            """
    @staticmethod
    def to_char_blocks(string):
        new_string = ""
        last_char = ""
        for x in string:
            char_type = Utils.get_char_class(x)
            if not char_type == last_char:
                last_char = char_type
                new_string += last_char
        return new_string

        """
        它接受一个字符串并返回一个相同长度的字符串，其中每个字符都被替换为一个字符表示原始字符的类
:param string:要转换的字符串
:return:输入字符串的字符类型。
        """
    @staticmethod
    def to_char_types(string):
        #print (Utils.get_char_class_ed(x) for x in string)
        #print ("".join([Utils.get_char_class_ed(x) for x in string]))
        return "".join([Utils.get_char_class_ed(x) for x in string])

        """
        它接受两个字符串并返回它们之间的编辑距离
str1:第一个要比较的字符串
str2:要比较的字符串
:return:两个字符串之间的编辑距离。
        """
    @staticmethod
    def edit_dist_mod(str1, str2):
        new_str1 = Utils.to_char_types(str1)
        new_str2 = Utils.to_char_types(str2)
        # print (str1)
        # print (str2)
        # print (new_str1)
        # print (new_str2)
        # print (editdistance.eval(new_str1, new_str2))
        return editdistance.eval(new_str1, new_str2)

        """
        它接受一个字符串列表，对它们进行洗牌，然后计算相邻字符串之间的平均编辑距离
:param column:字符串列表
:return:列中相邻元素之间的平均编辑距离。
        """
    @staticmethod
    def col_distance(column):
        shuffle(column) #序列的所有元素随机排序
        #print (float(len(column) - 1))
        return sum([Utils.edit_dist_mod(column[i], column[i + 1]) for i in range(len(column) - 1)]) / float(len(column) - 1)

        """
        列中单词之间的平均距离小于列中单词平均长度的一半列，则列有结构
:param column:字符串列表
:return:列与列平均长度之间的距离。
        """
    @staticmethod
    def has_structure(column):
        return Utils.col_distance(column) < 0.5 * sum([len(x) for x in column]) / float(len(column))

        """
        它接受一个字符串列表，并返回列表中最常见的字符串
:param str_list:字符串列表
:return:列表的模式。
        """
    @staticmethod
    def mode(str_list):
        max_element = None
        elements = {}
        for item in str_list:
            elements[item] = elements.get(item, 0) + 1
            if (not max_element) or (elements[item] > elements[max_element]):
                max_element = item
        return max_element

    """
          它接受两个字符串，并返回0到1之间的数字，其中0表示字符串相同，1表示它们完全不同
  str1:第一个要比较的字符串
  str2:要比较的字符串
  :返回:两个字符串之间的加权jaccard距离。
          """
    @staticmethod
    def weighted_jaccard(str1, str2):
        d1 = distance.jaccard(str1, str2)
        d2 = 4 * \
            distance.jaccard(Utils.to_char_blocks(
                str1), Utils.to_char_blocks(str2))
        return d1 + d2

        """
        它接受一个字符串列表，并返回一个整数列表，其中每个整数都是类的集群号相应的字符串
:param strings:字符串列表
:return:每个字符串的集群号。
        """
    @staticmethod
    def cluster_strings(strings):
        def d(coord):
            i, j = coord
            return Utils.weighted_jaccard(strings[i], strings[j])
        coords = np.triu_indices(len(strings), 1)
        dist_mat = np.apply_along_axis(d, 0, coords)
        linkage_mat = hierarchy.linkage(dist_mat)
        for i in [x / 10.0 for x in range(51)]:
            fcl = hierarchy.fcluster(linkage_mat, i, "distance")
            if max(fcl) <= 5:
                return fcl

    """
            合并对给定函数具有相同值的分支
    :param branches:一个分支列表
    :param func:接受一个分支并为其返回一个标记的函数
    :返回:一个分支列表。
            """
    @staticmethod
    def merge_branches(branches, func=None):
        if func is None:
            def func(x): return x
        seen = {}
        result = []
        for item in branches:
            marker = func(item)
            if marker in seen:
                for s in item.learned_lines:
                    seen[marker].learn_string(s)
            else:
                seen[marker] = item
                result.append(item)
        return result

        """
        函数接受一个分支列表和一个索引列表，并将第一个索引处的分支与在第二个指标处分支
:param branches:一个分支列表
:param indexes:待合并分支的索引
:返回:合并后的分支被移除的分支列表。
        """
    @staticmethod
    def perform_merge(branches, indices):
        for s in branches[indices[1]].learned_lines:
            branches[indices[0]].open_for_learning()
            branches[indices[0]].learn_string(s)
        del branches[indices[1]]
        return branches

        """
        它从两个DAFSA生成随机字符串，并计算两组字符串之间的平均编辑距离
:参数d1:第一个DFA
:参数d2:我们要比较的DFA
:param mod:如果为True，则使用修改后的编辑距离(见下文)，默认为False(可选)
:参数精度:结果的期望精度，默认为1(可选)
:return:两个DAFSA之间编辑距离的平均值。
        """
    @staticmethod
    def compare_dafsas_flat(d1, d2, mod=False, accuracy=1):
        n = rl_config.config["clt_sample_size"]
        curr_stdev = None
        inc = 10
        all_dists = []
        # pep8: use function instead
        needed_sample_size = lambda x: (1.96 * x / float(accuracy))**2
        while (curr_stdev is None or len(all_dists) < needed_sample_size(curr_stdev)):
            for i in range(inc):
                g1 = d1.generate_random_strings(n)
                g2 = d2.generate_random_strings(n)
                if not mod:
                    all_dists.append(
                        np.mean([editdistance.eval(g1.next(), g2.next()) for i in range(n)]))
                else:
                    all_dists.append(
                        np.mean([Utils.edit_dist_mod(g1.next(), g2.next()) for i in range(n)]))
            curr_stdev = np.std(all_dists)
        return np.mean(all_dists)

        """
        它接受两个DAFSA对象，并在第二个DAFSA生成的字符串上返回第一个DAFSA的平均分
:参数d1:第一个DAFSA
:参数d2:我们正在比较的DAFSA
:return: d2生成的字符串前馈分数的平均值。
        """
    @staticmethod
    def compare_dafsas_nested(d1, d2):
        n = rl_config.config["clt_sample_size"]
        curr_stdev = 5
        all_dists = []
        needed_sample_size = lambda x: (1.96 * x / 0.5)**2
        while len(all_dists) < needed_sample_size(curr_stdev):
            all_dists += [d1.feed_forward_score([s]) for s in d2.generate_random_strings(n)]
            curr_stdev = np.std(all_dists)
        return np.mean(all_dists)
        """
        它在2D数组中查找最小值并返回该值的索引。它接受两个字典，并对它们进行比较。
函数返回一个分数字典。
分数是通过比较两个字典的值计算出来的。
字典的值是字典的列表。
列表中的字典通过比较它们的值来进行比较。
:参数d1:第一个dafsa
:参数d2:要合并到d1中的dafsa
:返回:两个大法的分数。
:param mat:二维数字数组
:return:矩阵中最小值的索引。
        """
#比较合并
    @staticmethod
    def compare_dafas_merging(d1, d2):
        scores = Utils.compare_dafsas_nested(d1, d2)
        return scores

        """
        它查找二维数组中的最小值并返回该值的索引
:param mat:二维数字数组
:return:矩阵中最小值的索引。
        """
    @staticmethod
    def two_dimensional_min(mat):
        min_index = (0, 0)
        min_val = -1
        for i in range(len(mat)):
            for j in range(len(mat[i])):
                if mat[i][j] < min_val or min_val == -1:
                    min_index = (i, j)
                    min_val = mat[i][j]
        return min_index

        """
        它接受一个分支和一个索引的列表，并从该索引处的分支返回一个随机字符串
:param branches:一个Branch对象列表
:param i:要生成字符串的分支索引
:return:来自第i个分支的随机字符串。
        """
    @staticmethod
    def g(branches, i):
        return branches[i].generate_random_strings(1).next()

        """
        它获取一个分支列表和一个集中列表，并合并过于相似的分支，并添加合并分支的集中度
         :param branches:列表的列表
         :param min_distance:两个分支合并的最小距离
         :参数浓度:每个分支的浓度列表
          :return:正在返回分支和浓度。
        """
    @staticmethod
    def condense_branches(branches, min_distance, concentrations):
        def distance_func((x, y)):
            return Utils.compare_dafas_merging(branches[x], branches[y])
        #editdistance.eval(Utils.g(branches, x), Utils.g(branches, y))
        flag = True
        while flag:
            flag = False
            for i in range(len(branches)):
                for j in range(len(branches)):
                    dist = distance_func((i, j))
                    if not i == j and dist < min_distance:
                        branches = Utils.perform_merge(branches, (i, j))
                        concentrations[i] += concentrations[j]
                        del concentrations[j]
                        flag = True
                        break
                if flag:
                    break
        return branches, concentrations

    """
            它接受一个分支列表和一个浓度列表，并返回一个新的分支列表，一个新的浓度，以及任意两个分支之间的最小距离
            It takes a list of branches and a list of concentrations, and returns a new list of branches, a new list of
            concentrations, and the minimum distance between any two branches        
            :param branches: a list of lists 
            :param concentrations: a list of concentrations of each branch
            :return: the merged branches, the minimum distance, and the concentrations.
            """
#动态合并分支
    @staticmethod
    def merge_branches_dynamic(branches, concentrations):
        #计算分支之间的成对距离
        def distance_func((x, y)):
            return Utils.compare_dafas_merging(branches[x], branches[y])
        #editdistance.eval(Utils.g(branches, x), Utils.g(branches, y))
        distance_matrix = [
            [distance_func((i, j)) if not i == j else 10 for i in range(len(branches))]
            for j in range(len(branches))
        ]
        min_index = Utils.two_dimensional_min(distance_matrix)
        min_distance = distance_func(min_index)
        branches = Utils.perform_merge(branches, min_index)
        concentrations[min_index[0]] += concentrations[min_index[1]]
        del concentrations[min_index[1]]
        return branches, min_distance, concentrations
