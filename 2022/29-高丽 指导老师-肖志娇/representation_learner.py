# coding=utf-8
from random import random

import numpy as np
import numpy.random as rand

import rl_config
from structure_learner import StructuredTextLearner
from utils import Utils


class RepresentationLearner:
#lines就是所有的列和对于字符串
    def __init__(self, lines):
        """
        该函数接受一个行列表，对它们进行洗牌，然后创建一个StructuredTextLearner对象列表。

第一个StructuredTextLearner对象由列表中的第一行初始化。

StructuredTextLearner对象的其余部分使用列表中的其余行进行初始化。

然后函数返回StructuredTextLearner对象的列表。

:param lines:我们将要集群的文本行
        """
        self.lines = lines
        rand.shuffle(self.lines)
        if rl_config.config["should_branch"]:
            #print(self.lines[0])
            #print(StructuredTextLearner([self.lines[0]])) 簇 clusters就是个StructuredTextLearner类
            self.clusters = [StructuredTextLearner([self.lines[0]])]
            self.branching_threshold = rl_config.config[
                "branching_seed_threshold"]
        else:
            self.string_clusters = self.cluster()
            self.clusters = [StructuredTextLearner(
                c) for c in self.string_clusters]
        self.cluster_concentrations = [1]

    # 用于性能图。
    def get_num_branches(self):
        return len(self.clusters)

    # Used for performance plots.
    def get_total_num_hinges(self):
        num_hinges = 0
        for cluster in self.clusters:
            num_hinges += cluster.get_num_hinges()
        return num_hinges

    # Used for performance plots.
    def get_max_num_hinges(self):
        max_num_hinges = 0
        for cluster in self.clusters:
            max_num_hinges = max(max_num_hinges, cluster.get_num_hinges())
        return max_num_hinges

    def cluster(self):
        """
        它接受一个字符串列表，并返回一个字符串列表的列表，其中每个字符串列表都是
类似的字符串
:return:字符串列表的列表。
        """
        sample_size = rl_config.get("default_sample_size")(self.lines)
        seeds = Utils.cluster_strings(self.lines[:sample_size])
        clusters = [[] for _ in range(max(seeds))]
        for i in range(len(seeds)):
            clusters[seeds[i] - 1].append(self.lines[i])
        return clusters

    def get_right_cluster(self, line):
        """
        如果我们应该进行分支，那么就获得正确的或分支集群。否则，获得正确的无分支集群

:param line:我们当前所在文件的行
:return:返回正确的集群。
        """
        if rl_config.config["should_branch"]:
            return self.get_right_or_branch(line)
        return self.get_right_no_branch(line)

    def get_right_no_branch(self, line):
        """
        它返回给定行前馈评分最低的群集的索引

:参数line:要分类的行
:return:评分最低的集群索引。
        """
        min_s = None
        min_c = None
        for i in range(len(self.clusters)):
            s = self.clusters[i].feed_forward_score([line])
            if s == 0:
                return i
            if min_s is None or s < min_s:
                min_s = s
                min_c = i
        return min_c
#如果大于最大分支就合并
    def get_right_or_branch(self, line):
        """
        如果分支数大于最大分支数，请合并分支。否则，如果
该行的Score为0，返回集群的索引。否则，如果该线的分数小于
分支阈值，返回集群的索引。否则，创建一个新的集群并返回
集群

:参数line:要分类的文本行
:return:该行所属集群的索引。
        """
        min_s = None
        min_c = None
        #如果分支的数量超过了指定的最大值
        if len(self.clusters) > rl_config.config["max_branches"]:
            #合并分支
            self.clusters, self.branching_threshold, self.cluster_concentrations = Utils.merge_branches_dynamic(
                self.clusters, self.cluster_concentrations)
        for i in range(len(self.clusters)):
            s = self.clusters[i].feed_forward_score([line])
            if s == 0:
                return i
            if (min_s is None or s < min_s) and s < self.branching_threshold:
                min_s = s
                min_c = i
        if min_c is None:
            self.clusters.append(StructuredTextLearner([line]))
            self.cluster_concentrations.append(0)
            return len(self.clusters) - 1
        else:
            return min_c

#返回学习后的模式
    def learn(self):
        """
        对于每个集群，了解集群的深层结构。然后，对每一行，为这一行找到正确的集群，
学习线的弦。最后，压缩分支
        """
        [s.learn_deep_structure() for s in self.clusters]
        for line in self.lines:
            x = self.get_right_cluster(line)
            self.clusters[x].learn_string(line)
            self.cluster_concentrations[x] += 1
        self.clusters, self.cluster_concentrations = Utils.condense_branches(self.clusters, 0.1, self.cluster_concentrations)

    def __str__(self):
        """
        该函数接受一个聚类列表和一个聚类浓度列表，并返回一个字符串
用管道符号分隔的簇的串联
:return:对象的字符串表示形式。
        """
        assert len(self.clusters) == len(self.cluster_concentrations)
        x = list(set([str(x) for x in self.clusters]))
        return "|".join(x)

    def generate_strings(self, sample_size):
        """
        对于每个集群，从该集群生成一个随机字符串
:param sample_size:从每个集群生成的字符串的数量
        """
        for i in range(len(self.clusters)):
            for gen_string in self.clusters[int(random() * len(self.clusters))].generate_random_strings(sample_size):
                yield gen_string
    
    def generate_mh_strings(self, num_strings=None):
        """
        它接受一个生成器列表，并返回一个生成器，该生成器依次从每个生成器生成下一项
:param num_strings:要生成的字符串数量。如果为None，则生成器将继续生成字符串
        """
        k = 0
        generators = [c.generate_mh_strings() for c in self.clusters]
        while num_strings is None or k < num_strings:
            yield generators[k%len(generators)].next()
            k += 1

    def word_outlier_score(self, word):
        """
        对于每一个聚类，我们计算单词的原始得分，然后我们调整原始得分的浓度聚类，然后将调整后的分数加到总分中
:参数word:要评分的单词
:return:每个聚类中单词得分的平均值。
        """
        #right_cluster = self.get_right_no_branch(word)
        total_score = 0
        for right_cluster in range(len(self.clusters)):
            raw_score = self.clusters[right_cluster].feed_forward_score([word])
            adjustment = self.cluster_concentrations[right_cluster]/float(sum(self.cluster_concentrations))
            adjusted_score = adjustment*raw_score
            print(word, raw_score, adjustment)
            total_score += adjusted_score
        print(word, total_score/float(len(self.clusters)))
        return total_score/float(len(self.clusters))

    def score_strings(self, strings):
        """
        它接受一个字符串列表，并返回这些字符串中正确分类的字符的百分比 通过网络
:param strings:字符串列表
:return:字符串的值。
        """
        strings_list = [string for string in strings]
        x = [self.clusters[self.get_right_no_branch(string)].feed_forward_score([
            string]) for string in strings_list]
        return 100 * float(sum(x)) / sum([len(string) for string in strings_list])

    def score_strings_from_model(self, model):
        """
        我们从模型中生成一串字符串，然后计算字符串和数据。
这个函数要比这个复杂一点，因为我们要确保有足够的样本来得到a，很好地估计了平均距离。
:参数model:要评估的模型
:return:模型生成的字符串得分的平均值。
        """
        n = rl_config.config["clt_sample_size"]
        curr_stdev = 5
        all_dists = []
        needed_sample_size = lambda x: (1.96 * x / 0.5)**2
        while len(all_dists) < needed_sample_size(curr_stdev):
            all_dists += [self.score_strings(model.generate_strings(n))]
            curr_stdev = np.std(all_dists)
        return np.mean(all_dists)
