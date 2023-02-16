import math
import random
import numpy as np
import re
import sys

#python默认的递归深度是很有限的，大致为900。setrecursionlimit用于手动设置递归深度
sys.setrecursionlimit(99999)
SPLIT_CACHE = {}

#规则类
class Rule:
    def __init__(self, priority, ranges):
        # each range is left inclusive and right exclusive, i.e., [left, right)
        #每个区间都是左包含右排除的，即[左，右]
        #ranges是规则里每个字段的区间。ranges数组大小是10，下标0和1分别是src_ip的左右界，以此类推
        self.priority = priority
        self.ranges = ranges
        self.names = ["src_ip", "dst_ip", "src_port", "dst_port", "proto"]
    
    #判断给的左右界是否和当前Rule对象的某个维度的区间相交
    def is_intersect(self, dimension, left, right):
        #若left大于右界，或right小于左界，则返回false，即不相交。否则返回true，即相交
        return not (left >= self.ranges[dimension*2+1] or \
            right <= self.ranges[dimension*2])
    
    #判断给的范围是否和当前Rule对象在多维度上相交
    def is_intersect_multi_dimension(self, ranges):
        #若ranges有任意一个维度出现“左边大于当前Rule对象的右界，或右边小于左界”，则返回false，即不相交。否则返回true，即相交
        for i in range(5):
            if ranges[i*2] >= self.ranges[i*2+1] or \
                    ranges[i*2+1] <= self.ranges[i*2]:
                return False
        return True

    def sample_packet(self):
        #random.randint(x,y)用来生成随机数，参数x和y代表生成随机数的区间范围
        #此处根据每个维度的区间范围来给每个字段生成随机值
        src_ip = random.randint(self.ranges[0], self.ranges[1] - 1)
        dst_ip = random.randint(self.ranges[2], self.ranges[3] - 1)
        src_port = random.randint(self.ranges[4], self.ranges[5] - 1)
        dst_port = random.randint(self.ranges[6], self.ranges[7] - 1)
        protocol = random.randint(self.ranges[8], self.ranges[9] - 1)
        packet = (src_ip, dst_ip, src_port, dst_port, protocol)
        #assert（断言）用于判断一个表达式，在表达式条件为 false 的时候触发异常
        #matches()函数是Rule类的自定义函数。此处用于检查当前Rule对象和生成的样本是否匹配，即样本是否在Rule对象的区间内
        assert self.matches(packet), packet
        return packet

    #检查当前Rule对象和生成的样本是否匹配，判断packet中字段的值是否都在正确区间内
    def matches(self, packet):
        assert len(packet) == 5, packet
        return self.is_intersect_multi_dimension([
            packet[0] + 0,  # src ip
            packet[0] + 1,
            packet[1] + 0,  # dst ip
            packet[1] + 1,
            packet[2] + 0,  # src port
            packet[2] + 1,
            packet[3] + 0,  # dst port
            packet[3] + 1,
            packet[4] + 0,  # protocol
            packet[4] + 1
        ])
    
    #参数other指的是另一个Rule对象。该函数判断当前Rule对象的区间是否被other的区间所覆盖或包含（？）
    #不知道参数ranges是什么……
    def is_covered_by(self, other, ranges):
        for i in range(5):
            if (max(self.ranges[i*2], ranges[i*2]) < max(other.ranges[i*2], ranges[i*2]))or \
                    (min(self.ranges[i*2+1], ranges[i*2+1]) > min(other.ranges[i*2+1], ranges[i*2+1])):
                return False
        return True

    #打印出当前Rule对象的每个字段的区间
    def __str__(self):
        result = ""
        for i in range(len(self.names)):
            result += "%s:[%d, %d) " % (self.names[i], self.ranges[i * 2],
                                        self.ranges[i * 2 + 1])
        return result


def load_rules_from_file(file_name):
    rules = []
    #以acl1_1k为例，一条规则的格式为：
    #@176.19.181.33/32    90.145.23.162/32	0 : 65535	1550 : 1550	0x06/0xFF	0x0000/0x0200	
    rule_fmt = re.compile(r'^@(\d+).(\d+).(\d+).(\d+)/(\d+) '\
        r'(\d+).(\d+).(\d+).(\d+)/(\d+) ' \
        r'(\d+) : (\d+) ' \
        r'(\d+) : (\d+) ' \
        r'(0x[\da-fA-F]+)/(0x[\da-fA-F]+) ' \
        r'(.*?)')
    for idx, line in enumerate(open(file_name)):
        #\t的意思是 ：水平制表符。相当于按了键盘上的TAB按键
        elements = line[1:-1].split('\t')
        line = line.replace('\t', ' ')

        sip0, sip1, sip2, sip3, sip_mask_len, \
        dip0, dip1, dip2, dip3, dip_mask_len, \
        sport_begin, sport_end, \
        dport_begin, dport_end, \
        proto, proto_mask = \
        (eval(rule_fmt.match(line).group(i)) for i in range(1, 17))
        #group()函数用来提出分组截获的字符串，正则表达式里一对括号就是一个分组
        
        #<<左移，|按位或，~按位取反，&按位与
        
        #用一个整数表示源ip地址，并得到左右界。因为规则中的srcIP是指的是从某一个网络来的IP地址，而不是指某一个host，所以是有范围的
        sip0 = (sip0 << 24) | (sip1 << 16) | (sip2 << 8) | sip3
        sip_begin = sip0 & (~((1 << (32 - sip_mask_len)) - 1))
        sip_end = sip0 | ((1 << (32 - sip_mask_len)) - 1)
        #用一个整数表示目的ip地址，并得到左右界
        dip0 = (dip0 << 24) | (dip1 << 16) | (dip2 << 8) | dip3
        dip_begin = dip0 & (~((1 << (32 - dip_mask_len)) - 1))
        dip_end = dip0 | ((1 << (32 - dip_mask_len)) - 1)

        if proto_mask == 0xff:
            proto_begin = proto
            proto_end = proto
        else:
            proto_begin = 0
            proto_end = 0xff

        rules.append(
            Rule(idx, [
                sip_begin, sip_end + 1, dip_begin, dip_end + 1, sport_begin,
                sport_end + 1, dport_begin, dport_end + 1, proto_begin,
                proto_end + 1
            ]))
    return rules


#将value转换为n为的二进制表示
def to_bits(value, n):
    #运算符**指指数，2**n即计算2^n
    if value >= 2**n:
        print("WARNING: clamping value", value, "to", 2**n - 1)
        value = 2**n - 1
    assert value == int(value)
    #bin()函数返回一个整数或者长整数的二进制表示
    b = list(bin(int(value))[2:])
    assert len(b) <= n, (value, b, n)
    return [0.0] * (n - len(b)) + [float(i) for i in b]

#独热编码。为arr中的每一个值都进行n为独热编码
def onehot_encode(arr, n):
    out = []
    for a in arr:
        #建立了一个n为数组，初始化值为0
        x = [0] * n
        for i in range(a):
            x[i] = 1
        #extend()函数，用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）
        out.extend(x)
    return out

#节点类
class Node:
    def __init__(self, id, ranges, rules, depth, partitions, manual_partition):
        self.id = id
        #partitions的值可能是None或[smaller, part_dim, part_size]，smaller是一个布尔值。
        #True，表示该结点中的规则的某个维度的区间大小小于一个阈值，且该结点是父节点的左孩子。False表示大于那个阈值，且该结点是父节点的右孩子
        self.partitions = list(partitions or [])
        self.manual_partition = manual_partition
        self.ranges = ranges
        self.rules = rules
        self.depth = depth
        self.children = []
        self.action = None
        self.pushup_rules = None
        self.num_rules = len(self.rules)

    #判断节点是否已分区
    def is_partition(self):
        """Returns if node was partitioned."""
        if not self.action:
            return False
        elif self.action[0] == "partition":
            return True
        elif self.action[0] == "cut":
            return False
        else:
            return False

    def match(self, packet):
        if self.is_partition():
            matches = []
            for c in self.children:
                match = c.match(packet)
                if match:#若match不为空，则将match加入matches
                    matches.append(match)
            if matches:#若matches不为空，则按照该结点中的规则的优先级排序（？）
                matches.sort(key=lambda r: self.rules.index(r))
                return matches[0]
            return None
        elif self.children:
            for n in self.children:
                #contains()是Node类的自定义函数，判断当前的范围
                if n.contains(packet):
                    return n.match(packet)
            return None
        else:
            for r in self.rules:
                if r.matches(packet):
                    return r
    
    #判断当前Node对象的范围和给定的范围是否相交
    def is_intersect_multi_dimension(self, ranges):
        for i in range(5):
            if ranges[i*2] >= self.ranges[i*2+1] or \
                    ranges[i*2+1] <= self.ranges[i*2]:
                return False
        return True

    #判断当前Node对象的范围是否包含所给的packet的范围
    def contains(self, packet):
        assert len(packet) == 5, packet
        return self.is_intersect_multi_dimension([
            packet[0] + 0,  # src ip
            packet[0] + 1,
            packet[1] + 0,  # dst ip
            packet[1] + 1,
            packet[2] + 0,  # src port
            packet[2] + 1,
            packet[3] + 0,  # dst port
            packet[3] + 1,
            packet[4] + 0,  # protocol
            packet[4] + 1
        ])

    def is_useless(self):
        if not self.children:#若孩子节点为空，则返回false
            return False
        #若孩子结点不为空，则判断自身规则的数量和有最多规则的孩子结点的规则数量是否相等，相等则返回true，不等则返回false
        return max(len(c.rules) for c in self.children) == len(self.rules)
    
    #修剪当前Node对象中的规则
    def pruned_rules(self):
        new_rules = []
        for i in range(len(self.rules) - 1):#从最后一条规则开始遍历
            rule = self.rules[len(self.rules) - 1 - i]
            flag = False
            for j in range(0, len(self.rules) - 1 - i):#遍历当前规则前面的所有规则，并一一与其对比，如果当前规则被它前面的某个规则包含，则结束这个遍历过程
                high_priority_rule = self.rules[j]
                if rule.is_covered_by(high_priority_rule, self.ranges):
                    flag = True
                    break
            if not flag: #若当前规则不被它前面的任何规则包含，则将该规则加入到新的规则集合中
                new_rules.append(rule)
        new_rules.append(self.rules[0])
        new_rules.reverse()
        return new_rules

    def get_state(self):
        state = []
        #将当前Node对象范围里的每个值转换为二进制并放入state集合中
        state.extend(to_bits(self.ranges[0], 32))
        state.extend(to_bits(self.ranges[1] - 1, 32))
        state.extend(to_bits(self.ranges[2], 32))
        state.extend(to_bits(self.ranges[3] - 1, 32))
        assert len(state) == 128, len(state)
        state.extend(to_bits(self.ranges[4], 16))
        state.extend(to_bits(self.ranges[5] - 1, 16))
        state.extend(to_bits(self.ranges[6], 16))
        state.extend(to_bits(self.ranges[7] - 1, 16))
        assert len(state) == 192, len(state)
        state.extend(to_bits(self.ranges[8], 8))
        state.extend(to_bits(self.ranges[9] - 1, 8))
        assert len(state) == 208, len(state)
        
        #若不需要人工调整
        if self.manual_partition is None:
            # 0, 6 -> 0-64%
            # 6, 7 -> 64-100%
            #即0~7分别表示0%, 2%, 4%, 8%, 16%, 32%, 64%, 100%
            #初始化每个字段的切割状态都是0%~100%
            partition_state = [
                0,
                7,  # [>=min, <max) -- 0%, 2%, 4%, 8%, 16%, 32%, 64%, 100%
                0,
                7,
                0,
                7,
                0,
                7,
                0,
                7,
            ]
            #遍历当前Node对象的partitions数组，对partition_state进行更新
            for (smaller, part_dim, part_size) in self.partitions:
                if smaller:
                    partition_state[part_dim * 2 + 1] = min(
                        partition_state[part_dim * 2 + 1], part_size + 1)
                else:
                    partition_state[part_dim * 2] = max(
                        partition_state[part_dim * 2], part_size + 1)
            state.extend(onehot_encode(partition_state, 7))
        else:
            partition_state = [0] * 70
            partition_state[self.manual_partition] = 1
            state.extend(partition_state)
        state.append(self.num_rules)
        return np.array(state)

    #输出当前Node对象的信息
    #ID Acion Depth Range Children的ID 当前结点的规则 当前结点的pushup规则
    def __str__(self):
        result = "ID:%d\tAction:%s\tDepth:%d\tRange:\t%s\nChildren: " % (
            self.id, str(self.action), self.depth, str(self.ranges))
        for child in self.children:
            result += str(child.id) + " "
        result += "\nRules:\n"
        for rule in self.rules:
            result += str(rule) + "\n"
        if self.pushup_rules != None:
            result += "Pushup Rules:\n"
            for rule in self.pushup_rules:
                result += str(rule) + "\n"
        return result

#决策树类
class Tree:
    def __init__(
            self,
            rules,
            leaf_threshold,
            refinements={
                "node_merging": False,
                "rule_overlay": False,
                "region_compaction": False,
                "rule_pushup": False,
                "equi_dense": False
            }):
        # hyperparameters
        self.leaf_threshold = leaf_threshold
        self.refinements = refinements

        self.rules = rules
        #创建一个根结点，id=0，五个字段的范围[0~2^32,0~2^32,0~2^16,0~2^16,0~2^8]
        #根结点规则即为树的规则，深度是1，partition和manual_partition都为None
        self.root = self.create_node(
            0, [0, 2**32, 0, 2**32, 0, 2**16, 0, 2**16, 0, 2**8], rules, 1,
            None, None)
        if (self.refinements["region_compaction"]):
            self.refinement_region_compaction(self.root)
        self.current_node = self.root
        self.nodes_to_cut = [self.root] #需要切割的结点的集合
        self.depth = 1
        self.node_count = 1

    def create_node(self, id, ranges, rules, depth, partitions,
                    manual_partition):
        node = Node(id, ranges, rules, depth, partitions, manual_partition)

        if self.refinements["rule_overlay"]:
            self.refinement_rule_overlay(node)

        return node

    def match(self, packet):
        return self.root.match(packet)

    def get_depth(self):
        return self.depth

    def get_current_node(self):
        return self.current_node

    #终端叶子节点是一个规则数量低于给定阈值的节点
    def is_leaf(self, node):
        return len(node.rules) <= self.leaf_threshold
    
    #判断该决策树是否完成建立，即判断是否还有需要切割的结点
    def is_finish(self):
        return len(self.nodes_to_cut) == 0
    
    #当对结点进行分区后，会产生孩子结点，此时要对树进行更新
    def update_tree(self, node, children):
        if self.refinements["node_merging"]:
            children = self.refinement_node_merging(children)

        if self.refinements["equi_dense"]:
            children = self.refinement_equi_dense(children)

        if (self.refinements["region_compaction"]):
            for child in children:
                self.refinement_region_compaction(child)

        node.children.extend(children)
        children.reverse()#以分区操作为例，原本children=[left,right]，reverse后为[right,left]，由此看出是DFS方式构造树
        #pop()函数用于随机移除列表中的一个元素(默认最后一个元素),并且返回该元素的值。
        self.nodes_to_cut.pop()
        self.nodes_to_cut.extend(children)#将新结点也就是当前结点的孩子结点放入nodes_to_cut列表
        self.current_node = self.nodes_to_cut[-1] #将“当前结点”设置为nodes_to_cut列表的最后一个结点

    #利用CutSplit的分区器
    def partition_cutsplit(self):
        assert self.current_node is self.root
        from cutsplit import CutSplit
        self._split(self.root, CutSplit(self.rules), "cutsplit")

    #利用EffiCuts的分区器
    def partition_efficuts(self):
        assert self.current_node is self.root
        from efficuts import EffiCuts
        self._split(self.root, EffiCuts(self.rules), "efficuts")

    def _split(self, node, splitter, name):
        #tuple()函数用于将列表、区间(range)等转换为元组
        key = (name, tuple(str(r) for r in self.rules))
        if key not in SPLIT_CACHE:
            print("Split not cached, recomputing") #未缓存拆分，正在重新计算（？）
            SPLIT_CACHE[key] = [
                p for p in splitter.separate_rules(self.rules) if len(p) > 0  #在EffiCuts和CutSplit里找这个函数
            ]
        parts = SPLIT_CACHE[key]

        parts.sort(key=lambda x: -len(x)) #x指代parts中每一个元素
        assert len(self.rules) == sum(len(s) for s in parts)
        print(splitter, [len(s) for s in parts])

        children = []
        for i, p in enumerate(parts):
            c = self.create_node(self.node_count, node.ranges, p,
                                 node.depth + 1, [], i)
            self.node_count += 1
            children.append(c)
        node.action = ("partition", 0, 0)
        self.update_tree(node, children)
    
    #对当前结点进行分区操作
    def partition_current_node(self, part_dim, part_size):
        return self.partition_node(self.current_node, part_dim,
                                   part_size)
    
    #对树中的某结点进行分区操作
    #不懂这个part_size是啥，作用是什么！！！！！！！！！！！！！！！！！！！！
    def partition_node(self, node, part_dim, part_size):
        assert part_dim in [0, 1, 2, 3, 4], part_dim #srcip,dstip,srcport.dstport,protocol
        assert part_size in [0, 1, 2, 3, 4, 5], part_size #2%, 4%, 8%, 16%, 32%, 64%
        self.depth = max(self.depth, node.depth + 1)
        #给结点做标记，表示已被分区
        node.action = ("partition", part_dim, part_size)
        
        #partition_node()函数的内部函数，判断一条规则的某个指定维度的区间是否低于一个阈值，若是则被判为small rule，否则被判为big rule
        def fits(rule, threshold):
            span = rule.ranges[part_dim * 2 + 1] - rule.ranges[part_dim * 2]
            assert span >= 0, rule
            return span < threshold
        
        #将当前结点中的规则根据某个维度的区间大小划分为small或big，small的将会被放在当前结点的左孩子，big的
        small_rules = []
        big_rules = []
        #相当于arr=[2**32, 2**32, 2**16, 2**16, 2**8],max_size=arr[part_dim]
        max_size = [2**32, 2**32, 2**16, 2**16, 2**8][part_dim]
        threshold = max_size * 0.02 * 2**part_size  # 2% ... 64%
        for rule in node.rules:
            if fits(rule, threshold):
                small_rules.append(rule)
            else:
                big_rules.append(rule)
        
        #创建当前结点的左孩子
        left_part = list(node.partitions)
        left_part.append((True, part_dim, part_size))
        left = self.create_node(self.node_count, node.ranges, small_rules,
                                node.depth + 1, left_part, None)
        self.node_count += 1
        #创建当前结点的右孩子
        right_part = list(node.partitions)
        right_part.append((False, part_dim, part_size))
        right = self.create_node(self.node_count, node.ranges, big_rules,
                                 node.depth + 1, right_part, None)
        self.node_count += 1
        
        #给当前结点对象的children属性赋值
        children = [left, right]
        self.update_tree(node, children)
        return children
    
    #对当前结点进行切割操作
    def cut_current_node(self, cut_dimension, cut_num):
        return self.cut_node(self.current_node, cut_dimension, cut_num)
    
    #对树中的某结点进行切割操作。cut_dimension表示沿哪个维度切割，cut_num表示切割后当前结点会有几个孩子，即切了cut_num-1次
    def cut_node(self, node, cut_dimension, cut_num):
        self.depth = max(self.depth, node.depth + 1)
        node.action = ("cut", cut_dimension, cut_num)
        range_left = node.ranges[cut_dimension * 2]#当前结点在某个维度的区间的下限
        range_right = node.ranges[cut_dimension * 2 + 1]#当前结点在某个维度的区间的上限
        range_per_cut = math.ceil((range_right - range_left) / cut_num)#均匀切割后每个区间的大小

        children = []
        assert cut_num > 0, (cut_dimension, cut_num)
        for i in range(cut_num):#为当前结点创建cut_num个孩子
            child_ranges = list(node.ranges)#初始化要创建的孩子的区间=父节点的区间。然后下面再修改在切割维度上的区间
            child_ranges[cut_dimension * 2] = range_left + i * range_per_cut
            child_ranges[cut_dimension * 2 + 1] = min(range_right, range_left + (i + 1) * range_per_cut)
            
            #为要创建的孩子结点放入规则，当父节点中的规则的被切割维度的区间和此时正在创建的孩子结点中的被分配的区间相交，则这个规则会被分配给该孩子
            child_rules = []
            for rule in node.rules:
                if rule.is_intersect(cut_dimension,
                                     child_ranges[cut_dimension * 2],
                                     child_ranges[cut_dimension * 2 + 1]):
                    child_rules.append(rule)
            
            #创建孩子结点对象
            child = self.create_node(self.node_count, child_ranges,
                                     child_rules, node.depth + 1,
                                     node.partitions, node.manual_partition)
            children.append(child)
            self.node_count += 1
        
        #当所有孩子结点创建好后，则更新这个树
        self.update_tree(node, children)
        return children
    
    #多个维度切割当前结点，cut_dimensions是一个数组，存储要切割的维度。cut_nums也是数组，存储每个维度切割后得到的区间数，即每个维度的切割次数是cut_nums[i]-1次
    def cut_current_node_multi_dimension(self, cut_dimensions, cut_nums):
        self.depth = max(self.depth, self.current_node.depth + 1)
        node = self.current_node
        node.action = (cut_dimensions, cut_nums)
        
        #把每个维度分别切割cut_num次后，得到的区间大小存入range_per_cut
        range_per_cut = []
        for i in range(len(cut_dimensions)):
            range_left = node.ranges[cut_dimensions[i] * 2]
            range_right = node.ranges[cut_dimensions[i] * 2 + 1]
            cut_num = cut_nums[i]
            range_per_cut.append(
                math.ceil((range_right - range_left) / cut_num))
        
        #切割索引，反正就是用来巧妙地数数，来生成孩子。（自己定一下参数，写一下这个函数的过程就理解了）
        cut_index = [0 for i in range(len(cut_dimensions))]
        children = []
        while True:
            # compute child ranges
            #初始化当前孩子结点的区间等于父节点的区间
            child_ranges = list(node.ranges)
            #需要切割的每个维度都一一切割
            for i in range(len(cut_dimensions)):
                dimension = cut_dimensions[i]
                child_ranges[dimension*2] = node.ranges[dimension*2] + \
                    cut_index[i] * range_per_cut[i]
                child_ranges[dimension * 2 + 1] = min(
                    node.ranges[dimension * 2 + 1], node.ranges[dimension * 2]
                    + (cut_index[i] + 1) * range_per_cut[i])

            # compute child rules
            #给当前的孩子结点放入规则
            child_rules = []
            for rule in node.rules:
                if rule.is_intersect_multi_dimension(child_ranges):
                    child_rules.append(rule)

            # create new child
            child = self.create_node(self.node_count, child_ranges,
                                     child_rules, node.depth + 1,
                                     node.partitions, node.manual_partition)
            children.append(child)
            self.node_count += 1

            # update cut index
            cut_index[0] += 1
            i = 0
            while cut_index[i] == cut_nums[i]:
                cut_index[i] = 0
                i += 1
                if i < len(cut_nums):
                    cut_index[i] += 1
                else:
                    break

            if i == len(cut_nums):
                break

        self.update_tree(node, children)
        return children
    
    #不是均匀切割而是指定从某个位置切割
    def cut_current_node_split(self, cut_dimension, cut_position):
        self.depth = max(self.depth, self.current_node.depth + 1)
        node = self.current_node
        node.action = (cut_dimension, cut_position)
        range_left = node.ranges[cut_dimension * 2]
        range_right = node.ranges[cut_dimension * 2 + 1]
        range_per_cut = cut_position - range_left

        children = []
        for i in range(2):
            child_ranges = node.ranges.copy()
            child_ranges[cut_dimension * 2] = range_left + i * range_per_cut
            child_ranges[cut_dimension * 2 + 1] = min(
                range_right, range_left + (i + 1) * range_per_cut)

            child_rules = []
            for rule in node.rules:
                if rule.is_intersect(cut_dimension,
                                     child_ranges[cut_dimension * 2],
                                     child_ranges[cut_dimension * 2 + 1]):
                    child_rules.append(rule)

            child = self.create_node(self.node_count, child_ranges,
                                     child_rules, node.depth + 1,
                                     node.partitions, node.manual_partition)
            children.append(child)
            self.node_count += 1

        self.update_tree(node, children)
        return children
    
    #获取下一个需要切割的结点
    def get_next_node(self):
        self.nodes_to_cut.pop()
        if len(self.nodes_to_cut) > 0:
            self.current_node = self.nodes_to_cut[-1]
        else:
            self.current_node = None
        return self.current_node

    #检查两个结点是否相邻。此处相邻的定义是，有且仅有一个维度的区间相邻，其他区间【相等】
    def check_contiguous_region(self, node1, node2):
        count = 0
        for i in range(5):
            if node1.ranges[i*2+1] == node2.ranges[i*2] or \
                    node2.ranges[i*2+1] == node1.ranges[i*2]:
                if count == 1:
                    return False
                else:
                    count = 1
            elif node1.ranges[i*2] != node2.ranges[i*2] or \
                    node1.ranges[i*2+1] != node2.ranges[i*2+1]:
                return False
        if count == 0:
            return False
        return True
    
    #合并区域，让node1的区间等于node1和node2区间的并集
    def merge_region(self, node1, node2):
        for i in range(5):
            node1.ranges[i * 2] = min(node1.ranges[i * 2], node2.ranges[i * 2])
            node1.ranges[i * 2 + 1] = max(node1.ranges[i * 2 + 1],
                                          node2.ranges[i * 2 + 1])
    
    #不是很懂这个函数的作用
    #nodes是一个数组。不是把两个结点合并成一个结点，而是合并相邻结点在那个维度的区域
    def refinement_node_merging(self, nodes):
        while True:
            flag = True
            merged_nodes = [nodes[0]]
            last_node = nodes[0] #last_node指向了node[0]，所以下面改变了last_node相当于改变了它所指的对象
            for i in range(1, len(nodes)):
                #如果当前遍历到的结点和last_node相邻且两个结点中的规则是一样的
                #则将last_node的区间设置为自己和当前遍历到的结点的并集
                if self.check_contiguous_region(last_node, nodes[i]):
                    if set(last_node.rules) == set(nodes[i].rules):
                        self.merge_region(last_node, nodes[i])
                        flag = False
                        continue

                merged_nodes.append(nodes[i])
                last_node = nodes[i]

            nodes = merged_nodes
            if flag:
                break

        return nodes
    
    #当结点中没有规则或规则数量大于500，则返回空。否则修建结点中的规则
    def refinement_rule_overlay(self, node):
        if len(node.rules) == 0 or len(node.rules) > 500:
            return
        node.rules = node.pruned_rules()
    
    #区域压紧，调整某个结点的区域。这个函数在干啥呜呜
    def refinement_region_compaction(self, node):
        if len(node.rules) == 0:
            return
        
        #令new_ranges中每个维度的区间都为该结点所有规则中在某个维度的最大区间
        new_ranges = list(node.rules[0].ranges)
        for rule in node.rules[1:]:
            for i in range(5):
                new_ranges[i * 2] = min(new_ranges[i * 2], rule.ranges[i * 2])
                new_ranges[i * 2 + 1] = max(new_ranges[i * 2 + 1],
                                            rule.ranges[i * 2 + 1])
        #该结点的新区间为，在每个维度上，老区间和new_ranges的交集
        for i in range(5):
            node.ranges[i * 2] = max(new_ranges[i * 2], node.ranges[i * 2])
            node.ranges[i * 2 + 1] = min(new_ranges[i * 2 + 1],
                                         node.ranges[i * 2 + 1])
    
    #很难说这个函数是在干嘛……设置每个结点的pushup_rules吧……
    def refinement_rule_pushup(self):
        #初始化列表nodes_by_layer，值全为None
        nodes_by_layer = [None for i in range(self.depth)]

        #层序遍历该树，并将结点存入列表nodes_by_layer，nodes_by_layer[i]即为第i+1层的所有结点的列表
        current_layer_nodes = [self.root]
        nodes_by_layer[0] = current_layer_nodes
        for i in range(self.depth - 1):
            next_layer_nodes = []
            for node in current_layer_nodes:
                next_layer_nodes.extend(node.children)
            nodes_by_layer[i + 1] = next_layer_nodes
            current_layer_nodes = next_layer_nodes
        
        #从树的底部往上遍历，并更新每个结点的pushup_rules
        for i in reversed(range(self.depth)):
            for node in nodes_by_layer[i]:
                if len(node.children) == 0: #若当前遍历到的结点是叶子结点，则它的pushup_rules是它自己的规则的集合
                    node.pushup_rules = set(node.rules)
                else: 
                    #若当前遍历到的是非叶子结点，它的pushup_rules是所有孩子结点的pushup_rules的交集
                    node.pushup_rules = set(node.children[0].pushup_rules)
                    for j in range(1, len(node.children)):
                        #intersection() 方法用于返回两个或更多集合中都包含的元素，即交集
                        node.pushup_rules = node.pushup_rules.intersection(
                            node.children[j].pushup_rules)
                    #更新完当前结点，再更新当前结点的所有孩子。
                    #孩子结点的pushup_rules为原来自己的pushup_rules和父节点的pushup_rules的差集
                    for child in node.children:
                        #difference()方法用于返回两个集合的差集
                        child.pushup_rules = child.pushup_rules.difference(
                            node.pushup_rules)

    #这个函数是在干嘛！救命
    def refinement_equi_dense(self, nodes):
        # try to merge
        nodes_copy = []
        max_rule_count = -1
        #将nodes列表里的结点对象一一放入nodes_copy列表中，并让max_rule_count为含有最多规则的结点中的规则数
        for node in nodes:
            nodes_copy.append(
                Node(node.id, list(node.ranges), list(node.rules), node.depth,
                     node.partitions, node.manual_partition))
            max_rule_count = max(max_rule_count, len(node.rules))
        while True:
            flag = True
            merged_nodes = [nodes_copy[0]]
            last_node = nodes_copy[0]
            #遍历nodes_copy中的结点，若last_node和当前遍历到的结点相邻，则令rules为last_node的规则和当前遍历到的结点的并集
            for i in range(1, len(nodes_copy)):
                if self.check_contiguous_region(last_node, nodes_copy[i]):
                    rules = set(last_node.rules).union( #union()函数取多个集合的并集
                        set(nodes_copy[i].rules))
                    #若rules中的规则数小于last_node和当前遍历到的结点的规则数之和（即两个结点中有同样的规则）
                    #且rules中的规则数小于含有最多规则的结点中的规则数
                    #则将rules中的规则按照优先级排序，并令last_node所指的结点的规则等于rules，然后将当前遍历到的结点的区域合并到last_node
                    if len(rules) < len(last_node.rules) + len(nodes_copy[i].rules) and \
                        len(rules) < max_rule_count:
                        rules = list(rules)
                        rules.sort(key=lambda i: i.priority)
                        last_node.rules = rules
                        self.merge_region(last_node, nodes_copy[i])
                        flag = False
                        continue

                merged_nodes.append(nodes_copy[i])
                last_node = nodes_copy[i]

            nodes_copy = merged_nodes
            if flag:
                break

        # check condition
        if len(nodes_copy) <= 8:
            nodes = nodes_copy
        return nodes

    #计算结果，包括这棵树的所有规则所占的内存空间、内存访问次数（？）、叶子结点数量、非叶结点数量、结点总数
    def compute_result(self):
        if self.refinements["rule_pushup"]:
            self.refinement_rule_pushup()

        # memory space
        # non-leaf: 2 + 16 + 4 * child num
        # leaf: 2 + 16 * rule num
        # details:
        #     header: 2 bytes
        #     region boundary for non-leaf: 16 bytes
        #     each child pointer: 4 bytes
        #     each rule: 16 bytes
        result = {"bytes_per_rule": 0, "memory_access": 0, \
            "num_leaf_node": 0, "num_nonleaf_node": 0, "num_node": 0}
        nodes = [self.root]
        while len(nodes) != 0:
            next_layer_nodes = []
            for node in nodes:
                next_layer_nodes.extend(node.children)

                # compute bytes per rule
                if self.is_leaf(node):
                    result["bytes_per_rule"] += 2 + 16 * len(node.rules)
                    result["num_leaf_node"] += 1
                else:
                    resul-t["bytes_per_rule"] += 2 + 16 + 4 * len(node.children)
                    result["num_nonleaf_node"] += 1

            nodes = next_layer_nodes

        result["memory_access"] = self._compute_memory_access(self.root)
        result["bytes_per_rule"] = result["bytes_per_rule"] / len(self.rules)
        result[
            "num_node"] = result["num_leaf_node"] + result["num_nonleaf_node"]
        return result
    
    #不知道这个函数是干嘛的，或许是计算内存访问次数吧……
    def _compute_memory_access(self, node):
        #若当前结点是叶子结点或孩子结点为空，则返回1
        if self.is_leaf(node) or not node.children:
            return 1
        
        #若当前结点已被分区，则返回所有孩子结点的内存访问次数的和
        #否则返回1+孩子结点中内存访问次数的最大值
        if node.is_partition():
            return sum(self._compute_memory_access(n) for n in node.children)
        else:
            return 1 + max(
                self._compute_memory_access(n) for n in node.children)
    
    #widths存储每一层的宽度即每一层的结点数量
    #dim_stats的元素是列表，元素个数是层数。每个列表里的元素表示5个维度，每个元素的值是每一层里在这个维度被切割的结点的数量
    def get_stats(self):
        widths = []
        dim_stats = []
        nodes = [self.root]
        while len(nodes) != 0 and len(widths) < 30:
            dim = [0] * 5
            next_layer_nodes = []
            for node in nodes:
                next_layer_nodes.extend(node.children)
                if node.action and node.action[0] == "cut":
                    dim[node.action[1]] += 1
            widths.append(len(nodes))
            dim_stats.append(dim)
            nodes = next_layer_nodes
        return {
            "widths": widths,
            "dim_stats": dim_stats,
        }
    
    #将stats整理成字符串
    def stats_str(self):
        stats = self.get_stats()
        out = "widths" + "," + ",".join(map(str, stats["widths"]))
        out += "\n"
        for i in range(len(stats["dim_stats"][0])):
            out += "dim{}".format(i) + "," + ",".join(
                str(d[i]) for d in stats["dim_stats"])
            out += "\n"
        return out
    
    #输出stats
    def print_stats(self):
        print(self.stats_str())
    
    #打印出每一层的结点
    def print_layers(self, layer_num=5):
        nodes = [self.root]
        for i in range(layer_num):
            if len(nodes) == 0:
                return

            print("Layer", i)
            next_layer_nodes = []
            for node in nodes:
                print(node)
                next_layer_nodes.extend(node.children)
            nodes = next_layer_nodes
    
    #层序遍历树，输出每个结点的id、action、区间和所有孩子结点的id
    def __str__(self):
        result = ""
        nodes = [self.root]
        while len(nodes) != 0:
            next_layer_nodes = []
            for node in nodes:
                result += "%d; %s; %s; [" % (node.id, str(node.action),
                                             str(node.ranges))
                for child in node.children:
                    result += str(child.id) + " "
                result += "]\n"
                next_layer_nodes.extend(node.children)
            nodes = next_layer_nodes
        return result
