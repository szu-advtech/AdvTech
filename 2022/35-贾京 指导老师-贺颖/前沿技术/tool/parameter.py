import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs',type=int,default=1000,help="rounds of training")
    parser.add_argument('--num_users',type=int,default=50)
    parser.add_argument('--frac',type=float,default=0.2,help="the fraction of clients:C")
    parser.add_argument('--local_bs',type=int,default=40)
    parser.add_argument('--local_step',type=int,default=10)
    parser.add_argument('--test_bs',type=int,default=128)
    parser.add_argument('--lr',type=float,default=0.01)
    parser.add_argument('--beta',type=float,default=0.001)


    parser.add_argument('--iid',action='store_true',default=False)
    parser.add_argument('--dataset' , type=str , default='Mnist')
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--func',type=str,default='HF')
    parser.add_argument('--delta',type=float,default=0.055)


    args = parser.parse_args()
    return args