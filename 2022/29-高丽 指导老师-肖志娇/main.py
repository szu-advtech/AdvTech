# coding=utf-8
import argparse
import time
import traceback
from functools import wraps
from os import listdir

import numpy as np
import pandas as pd

from representation_learner import RepresentationLearner
from utils import Utils


def fn_timer(function):
    """
它打印出一个CSV行，其中包含函数名和运行时，以及可选的附加统计信息
:param function:要计时的函数
:return:正在返回function_timer函数。    """
    """
    打印执行的总运行时间的计时器装饰器。
由此装饰的函数也可以通过提供额外的统计信息
    a "additional_timer_stats" kwarg.
    """
    # 将函数属性复制到包装器函数的装饰器。
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()

        # 第一列是时间.
        stats_csv_header = 'function,runtime_seconds'
        stats_csv_data = '%s,%s' % (function.func_name, str(t1 - t0))

        # 剩余的列.
        if 'additional_timer_stats' in kwargs:
            for k, v in kwargs['additional_timer_stats'].iteritems():
                stats_csv_header += ',%s' % k
                stats_csv_data += ',%s' % str(v)

        if 'additional_timer_stats_from_result' in kwargs:
            for stat in kwargs['additional_timer_stats_from_result']:
                stats_csv_header += ',%s' % stat
                stats_csv_data += ',%s' % str(result[stat])

        print(stats_csv_header)
        print(stats_csv_data)

        return result
    return function_timer

#从路径中检索所有非隐藏文件。
def get_files_from_path(path):
    visible_files = []
    for f in listdir(path): #该方法返回不带根目录的文件名或子目录名
        if not f.startswith('.'):  # filter out hidden files
            path_to_file = path + "/" + f
            visible_files.append(path_to_file)
    return visible_files

#给定文件的绝对路径，它将检索列名。 它假设它们出现在第一行(CSV, TSV)。
def get_columns_from_file(filename, dataframe=None):
    """
    它读取CSV文件并返回列名列表

:param filename:你想从中获取列的文件名
:param dataframe:获取列名的数据框架。如果为None，则使用filename参数
:return:列名列表
    """
    column_names = []
    if dataframe is None:
        csv = pd.read_csv(filename, dtype=str, error_bad_lines=False) \
                .replace([None], [""], regex=True)
        column_names = list(csv)
    else:
        column_names = list(dataframe)
    return column_names


def get_col_from_file(fname, col):
    """
    它读取一个文件并返回一个字符串列表

:param fname:要读取的文件名
:param col:要从文件中读取的列。如果为None，则读取整个文件
:return:字符串列表
    """
    if col is None:
        l = list([x.replace('\n', '') for x in open(fname).readlines()])
    else:
        l = list(pd.read_csv(fname, dtype=str, error_bad_lines=True)
                 [col].replace([None], [""], regex=True))
    return [str(x) for x in l]


def get_regex_from_file(fname, col=None):
    """
    它接受文件名和列号，并返回与该列中的字符串匹配的正则表达式
:param fname:要读取的文件名
:param col:要从文件中读取的列号。如果为None，则读取整个文件
:return:正则表达式模型的字符串表示形式。
    """
    l = get_col_from_file(fname, col)
    model = learn_model(l)
    return str(model['model'])


def learn_model(data):
    """
    它接受一个数据值列表，并返回一个表示这些值的模型

:param data:数字列表的列表。每个数字列表代表一个数据点
:return:一个包含模型、分支数、最大铰链数和铰链平均数的字典。
    """
    """
    给定一个数据值的可迭代对象，它学习一个表示这些值的模型
    """
    #print(data)
    r = RepresentationLearner(data)
    r.learn()

    num_branches = r.get_num_branches()
    total_num_hinges = r.get_total_num_hinges()
    max_num_hinges = r.get_max_num_hinges()
    avg_num_hinges = total_num_hinges / num_branches

    return {'model': r,
            'num_branches': num_branches,
            'max_num_hinges': max_num_hinges,
            'avg_num_hinges': avg_num_hinges}


def compare_regex_from_file(fname1, fname2=None, col1=None, col2=None):
    """
    它接受两个文件，并返回第一个文件的正则表达式在第二个文件的字符串上的分数，反之亦然

:param fname1:包含第一个字符串列表的文件名
:参数fname2:要比较的文件
:param col1:第一个要比较的文件的列号
:param col2:要比较的文件的列号
:return:第一个值是第一个模型在第二个模型上的得分，第二个值是
第一个型号上的第二个型号。
    """
    l1 = get_col_from_file(fname1, col1)
    l2 = get_col_from_file(fname2, col2)

    r1 = RepresentationLearner(l1)
    r1.learn()

    r2 = RepresentationLearner(l2)
    r2.learn()

    return r1.score_strings_from_model(r2), r2.score_strings_from_model(r1)


def compare_dafsas_flat(d1, d2, mod=False, accuracy=0.1):
    """
    它比较两个DAFSA对象，并返回它们之间的差异列表

:参数d1:第一个要比较的DAFSA
:参数d2:要比较的DAFSA
:param mod:如果为True，则如果两个dafsas对精度参数取模相等，则函数将返回True。
默认为False(可选)
:参数精度:给定状态下两个dafsas值之间的最大差值
:返回:布尔值。
    """
    return Utils.compare_dafsas_flat(d1, d2, mod, accuracy)


def compare_representation_flat(fname1, fname2, col1, col2, mod=False):
    """
    它接受两个文件、两个列和一个布尔值，并返回两个列之间的平均相似度

:param fname1:第一个文件的名称
:param fname2:包含第二种表示的数据的文件的名称
:param col1:第一个要比较的文件中的列
:param col2:第二个文件中要与第一个文件进行比较的列
:param mod:如果为true，则按字母大小进行比较，默认为False(可选)
:return:两种表示的簇之间最小距离的平均值。
    """
    l1 = get_col_from_file(fname1, col1)
    l2 = get_col_from_file(fname2, col2)

    r1 = RepresentationLearner(l1)
    r1.learn()

    r2 = RepresentationLearner(l2)
    r2.learn()
    rep1results = []
    for cluster in r1.clusters:
        rep1results.append(
            min([compare_dafsas_flat(cluster, x, mod) for x in r2.clusters]))

    rep2results = []
    for cluster in r2.clusters:
        rep2results.append(
            min([compare_dafsas_flat(cluster, x, mod) for x in r1.clusters]))

    return np.mean(rep1results), np.mean(rep2results)


def format_for_output(filename, colname, structuredness_score, model_string, samples):
    """
    它接受一个文件名、一个列名、一个结构评分、一个模型字符串和一个示例列表，并返回一个
字符串，它以良好的格式包含所有信息

:param filename:正在分析的文件名
:param colname:要分析的列的名称
:param structuredness_score:列的结构评分
:param model_string:描述用于生成样本的模型的字符串
:param samples:用于计算结构评分的样本列表
:return:包含文件名、列名、结构评分、模型字符串和示例的字符串。
    """
    sample_string = ','.join(str(v) for v in samples)
    # string = (filename + " - " + colname + '\n' + "STRUCT. SCORE: " +
    #           str(structuredness_score) + '\n' + " MODEL: " + model_string +
    #           '\n' + " Samples: " + sample_string + "\n\n")
    string = (filename + " - " + colname + '\n' + "STRUCT. SCORE: " +
              str(structuredness_score) + '\n' + " MODEL: " + model_string +
              '\n')
    return string

    """
        它接受一列数据，并返回该列结构的模型

:param column_data:字符串列表，每个字符串都是列中的值
:return:包含以下键的字典:
'model':学习模型的字符串表示形式。
'avg_num_hinge ':学习模型中铰链的平均数量。
'max_num_hinge ':学习模型中铰链的最大数量。
'num_branches':学习模型中的分支数量。
    """
@fn_timer
def learn_structures_for_single_column(column_data, *args, **kwargs):
    """
    学习单个列的结构模型。
    这里的**kwargs只用于向fn_timer装饰器传递additional_timer_stats。
    """
    try:
        model = learn_model(column_data)
    except BaseException as e:
        traceback.print_exc()
        model['model'] = "ERROR: " + str(e)
        model['avg_num_hinges'] = 0
        model['max_num_hinges'] = 0
        model['num_branches'] = 0
    return model


def learn_structures_for_all_columns_in_file(filename, should_log=True):
    """
    它接受一个文件名，读取文件，然后学习文件中每个列的结构

:param filename:要学习其结构的文件名
:param should_log:是否输出到控制台，默认为True(可选)
:return:返回值是由两个元素组成的元组。第一个元素是模型列表，第二个元素是一个
dataframe。
    """
    """
    学习文件中所有列的结构模型。
    """
    dataframe = pd.read_csv(filename, dtype=str, error_bad_lines=False).replace(
        [None], ["None"], regex=True)
    colnames = get_columns_from_file(filename, dataframe)
    models = []
    for column in colnames:
        column_data = dataframe[column]
        average_row_length = get_average_row_length(column_data)
        structuredness_score = Utils.has_structure(column_data[:200])
        model = learn_structures_for_single_column(
            column_data,
            additional_timer_stats={
                'average_row_length': average_row_length,
                'column_length': len(column_data)
            },
            additional_timer_stats_from_result=[
                'avg_num_hinges', 'max_num_hinges', 'num_branches']
        )
        string = format_for_output(filename,
                                   column,
                                   structuredness_score,
                                   str(model['model']),
                                   column_data[:5])
        models.append(model['model'])
        # 学习文件中所有列的结构模型。考虑在外部函数计时时将此设置为可选的，例如I/O
        # 影响总执行时间。
        #
        # TODO: Use a logger instead.
        # 快速修复
        if should_log:
            print(str(string))
    return models, dataframe

def error_detect_for_filename(column_filename, output_filename, threshold=0.5):
    """
   它接受一个列文件，学习每个列的模型，然后输出一个包含每个列的分数的行文件

:param column_filename:包含要学习的列的文件名
:param output_filename:要写入输出的文件名
:param threshold:异常值阈值
    """
    """
    给定一个列文件名，我们学习一个列的模型，然后将每个条目放入模型中，
和输出行检测到是列的离群值
    """
    models, _ = learn_structures_for_all_columns_in_file(column_filename, should_log=False)
    dataframe = pd.read_csv("sample_outlier_indexed.csv", dtype=str)
    dataframe["Indexer"] = dataframe["Indexer"].astype(int)
    dataframe = dataframe.set_index("Indexer")
    dataframe = dataframe.sort_index()
    #dataframe = dataframe.reset_index()
    #dataframe = dataframe.drop("Index")
    score_vector = []
    all_scores = []
    raw_scores = []
    print(map(str, models))
    for index, row in dataframe.iterrows():
        row_scores = []
        for i in range(len(row)):
            elem = row[i]
            model = models[i]
            score = model.word_outlier_score(elem)
            row_scores.append(score)
        if len(all_scores) > 0:
            all_scores.append(row_scores)
#            all_scores.append(row_scores-np.min(raw_scores, axis=0))
#            all_scores.append(row_scores-np.array(raw_scores[-1]))
        else:
            all_scores.append(np.array([0]*len(row_scores)))
        raw_scores.append(row_scores)
    all_scores_transposed = zip(*all_scores)
    for i in range(len(all_scores_transposed)):
        dataframe["col" + str(i+1) + "-score"] = all_scores_transposed[i]

    print(len(zip(*all_scores)))
    all_scores_thresholded = np.array(all_scores)
    #all_scores_thresholded[all_scores_thresholded <= threshold] = 0
    #all_scores_thresholded[all_scores_thresholded > threshold] = 1
    generated_labeled = pd.DataFrame(all_scores_thresholded)
    generated_labeled.to_csv(output_filename + ".labeled", header=False, index=False)
    dataframe.to_csv(output_filename)

@fn_timer #要测试函数的使用时间时，只需要@fn_timer装饰器即可。
def learn_structures_for_all_files_in_path(path):
    """
    给定一个输入路径(path)，它读取目录中的所有文件和所有列
    学习每一列的结构。
    """
    filenames = get_files_from_path(path)
    for f in filenames:
        learn_structures_for_all_columns_in_file(f)


def get_average_row_length(column):
    """
    它接受一个文本列，并返回该列中行的平均长度

:param column:要获取的平均行长度的列
:return:列中行的平均长度。
    """
    row_lengths = [len(row) for row in column] #每个字符串的长度，一个数组
    return 0 if len(row_lengths) == 0 else (float(sum(row_lengths)) / len(row_lengths)) #所有字符串的总长度/字符串个数


#
if __name__ == "__main__":

    #argparse是python用于解析命令行参数和选项的标准模块,建立解析对象
    #了解单个文件或目录中所有文件的结构模型。
    parser = argparse.ArgumentParser(
        description="Learn structures model for a single file, or all files in a directory.")
    group = parser.add_mutually_exclusive_group(required=True)
    #增加属性 要处理的输入文件名的相对路径
    group.add_argument('-f', '--filename',
                       help='Relative path of input filename to be processed.')
    #给属性名之前加上“- -”，就能将之变为可选参数。 包含所有待处理文件的输入目录的相对路径
    group.add_argument(
        '-d', '--directory', help='Relative path of input directory containing all files to be processed.')
    group.add_argument(
        '-e', '--errordetect', help='Print out an error vector of guessed errors')
    #打印出猜测误差的误差向量 在哪里输出错误检测日志(csv优先)
    parser.add_argument('-o', '--output', help='Where to output error detection log (csv preferred)')
    #将列称为错误的阈值(忽略，除非与-e一起使用)
    parser.add_argument('-t', '--threshold', help='Threshold for calling a column an error (ignored unless used with -e)')
    #属性给与args实例 把parser中设置的所有“add_argument"给返回到args子类实例当中，那么parser中增加的属性内容都会在args实例中，使用即可。
    args = parser.parse_args()
    print(args)

    # 两者中的任何一个总是被提供的
    #
    if args.directory is not None:
        learn_structures_for_all_files_in_path(args.directory)
    elif args.filename is not None:
        learn_structures_for_all_columns_in_file(args.filename)
    else:
        error_detect_for_filename(args.errordetect, args.output, threshold=float(args.threshold))
