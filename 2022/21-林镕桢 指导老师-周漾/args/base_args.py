import argparse

def get_base_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=12, metavar='N')
    parser.add_argument('--epochs', type=int, default=100, metavar='N')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR')
    parser.add_argument('--gamma', type=float, default=0.5, metavar='M')
    parser.add_argument('--seed', type=int, default=6, metavar='S')
    parser.add_argument('--is_save', action='store_false')
    parser.add_argument('--save_frequence', type=int, default=5)

    return parser.parse_args()