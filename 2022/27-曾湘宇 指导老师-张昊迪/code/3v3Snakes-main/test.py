#
# legal_actions = [[1 for i in range(4)] for j in range(6)]
# print(legal_actions)
# import numpy as np
# l = np.asarray([[1,2,3],[4,5,6]])
# print(f'l:{l}')
#     t = [[0]]
#     print(t)


# key = 'abcde'
# value = range(1, 6)
# res = dict(zip(key, value))
# {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
# print(res)
# import argparse
# def basic_options():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_mode', type=str, default= 'unaligned', help='chooses how datasets are loaded')
#     parser.add_argument('--mode', type=str, default='test', help='test mode')
#     return parser
#
# def data_options(parser):
#     parser.add_argument('--lr', type=str, default='0.0001', help='learning rate')
#     return parser
#
# if __name__ == '__main__':
#     parser = basic_options()
#     opt, unparsed = parser.parse_known_args()
#     print(opt)
#     print(unparsed)
#     parser = data_options(parser)
#     opt = parser.parse_args()
#     print(opt)
#
# list_ = [(5,2),(6,5)]
# arg1 = zip(*list_)
# print(arg1)
def train(arg1,arg2,arg3):
    print(f"arg1:{arg1},arg2:{arg2},arg3:{arg3}")
list_ = [[1,2,3],[4,5,6],[7,8,9]]
train(*list_)
#res:  arg1:[1, 2, 3],arg2:[4, 5, 6],arg3:[7, 8, 9]