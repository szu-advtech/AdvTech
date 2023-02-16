
import argparse

def args_node_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--gpu', type=int, default=-1, help="GPU ID, -1 for CPU")
    parser.add_argument('--epoch', type=int, default=1, help="rounds of training")
    parser.add_argument('--dataset', type=str, default="mnist", help="dataset")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--rpc_url', type=str, default="http://localhost:8546", help="rpc_url")
    parser.add_argument('--start_train_index', type=int, default=0, help="num_users")
    parser.add_argument('--end_train_index', type=int, default=0, help="num_users")
    parser.add_argument('--start_test_index', type=int, default=0, help="num_users")
    parser.add_argument('--end_test_index', type=int, default=0, help="num_users")
    parser.add_argument('--user_id', type=int, default=0, help="user_id")
    parser.add_argument('--iid', type=bool, default=True, help='iid')
    parser.add_argument('--address', type=str, default='', help='')
    args = parser.parse_args()
    return args


def args_server_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='cnn', help="cnn")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of imges")
    parser.add_argument('--eth_rpc', type=str, default='http://localhost:8545', help='')
    parser.add_argument('--port', type=int, default=8000, help='')
    args = parser.parse_args()
    return args

def args_debug_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epoch', type=int, default=1, help="rounds of training")
    parser.add_argument('--gpu', type=int, default=-1, help="GPU ID, -1 for CPU")
    parser.add_argument('--dataset', type=str, default="mnist", help="dataset")
    parser.add_argument('--rpc_url', type=str, default="http://localhost:8545", help="rpc_url")
    args = parser.parse_args()
    return args