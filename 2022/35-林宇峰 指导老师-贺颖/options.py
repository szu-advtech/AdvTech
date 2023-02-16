import argparse
def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('epochs',type = int,default = 10,help = '训练的周期数')
    return parser.parse_args()
