# coding=utf-8
import argparse
import rstr
from main import learn_model
import numpy as np

def first_summarization_test(base_regex, n):
    """
    It generates a bunch of strings from a regex, then checks how many of those strings are summarized by the learned model

    :param base_regex: the regex to generate strings from
    :param n: the number of strings to generate
    :return: The proportion of strings that are summarized by the column
    """
    '''First summarization test: generate strings from the base regex
    Then check how much of the regex is summarized by the column
    第一个摘要测试:从基正则表达式生成字符串
然后检查列汇总了多少正则表达式'''
    regex_strs = [rstr.xeger(base_regex) for i in range(n)]
    m = learn_model(regex_strs)['model']
    scores = np.array([m.score_strings([s]) for s in regex_strs])
    return len(scores[scores > 0.4])/len(scores)

def second_summarization_test(base_regex, n, num_cols=10):
    '''Second summarization test: generate strings from the base regex for
    10 length-n columns. Result is number of wrong merges, basically equals
    (number of pairs that would not be merged)/(num_cols**2)
    第二个摘要测试:从基本正则表达式生成10个长度为n的列的字符串。结果是错误合并的数量，基本相等
(未合并的对数)/(num_cols**2)'''
    branches = [
                    [rstr.xeger(base_regex) for i in range(n)]
                for i in range(num_cols)]

    results = np.ones((num_cols, num_cols))
    for i in range(num_cols):
        for j in range(i):
            m1 = learn_model(branches[i])['model']
            m2 = learn_model(branches[j])['model']
            results[i][j] = m1.score_strings_from_model(m2) < 0.4
            results[j][i] = m2.score_strings_from_model(m1) < 0.4
    return 1.0-np.sum(results)/results.size

def third_summarization_test(regexes, n):
    '''Third summarization test: generate strings from regexes. Print
    a table indicating pairwise regex containment
    第三个摘要测试:从正则表达式生成字符串。打印一个表，指示成对的正则表达式包含'''
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate accuracy plots for summarization task')
    parser.add_argument('base_regex', metavar="b", type=str, help="Baseline regex (hope to capture this)")
    parser.add_argument('--num_samples', metavar="n", type=int, help="Number of samples to generate")
    parser.add_argument('--test_num', metavar="t", type=int, help="Test number to do---0 for simple summary, 1 for uniform comparison, 2 for heterogenous comparison")
    args = parser.parse_args()
    n = args.num_samples if args.num_samples else 100
    t = args.test_num if args.test_num else -1
    tests = [first_summarization_test, second_summarization_test, third_summarization_test]
    if t == -1:
        for i in range(len(tests)):
            print "Error ", i+1, ": ", tests[i](args.base_regex, n)
    else:
        print "Error for selected test ", t, ": ", tests[t](args.base_regex, n)
