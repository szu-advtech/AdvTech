import copy, sys
import time
import numpy as np
import torch
import random
import torch.utils.model_zoo as model_zoo
from pathlib import Path

lib_dir = (Path(__file__).parent / ".." / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))
mod_dir = (Path(__file__).parent / ".." / "lib" / "models").resolve()
if str(mod_dir) not in sys.path:
    sys.path.insert(0, str(mod_dir))

from resnet import resnet18
from options import args_parser
from models import CNNMnist, CNNFemnist,  Lenet
from utils import get_dataset, exp_details, proto_aggregation, agg_func, average_weights
from method import FedProto, FedAvg

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

if __name__ == '__main__':
    start_time = time.time()

    args = args_parser()
    exp_details(args)

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda':
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    n_list = np.random.randint(max(2, args.ways - args.stdev), min(args.num_classes, args.ways + args.stdev + 1), args.num_users)
    if args.dataset == 'mnist':
        k_list = np.random.randint(args.shots - args.stdev + 1 , args.shots + args.stdev - 1, args.num_users)
    elif args.dataset == 'cifar10':
        k_list = np.random.randint(args.shots - args.stdev + 1 , args.shots + args.stdev + 1, args.num_users)
    elif args.dataset == 'femnist':
        k_list = np.random.randint(args.shots - args.stdev + 1 , args.shots + args.stdev + 1, args.num_users)

    train_dataset, test_dataset, user_groups, user_groups_lt, classes_list, classes_list_gt = get_dataset(args, n_list, k_list)
    if args.alg == 'fedproto':
        local_model_list = []
        for i in range(args.num_users):
            if args.dataset == 'mnist':
                if args.mode == 'model_heter':
                    args.out_channels = 20
                    if i < 10:
                        local_model = CNNMnist(args = args)
                    else:
                        local_model = Lenet(args = args)
                else:
                    args.out_channels = 20

                    local_model = CNNMnist(args=args)

            elif args.dataset == 'femnist':
                if args.mode == 'model_heter':
                    args.out_channels = 20
                    if i < 10:
                        local_model = CNNFemnist(args=args)
                    else:
                        local_model = Lenet(args=args)
                else:
                    args.out_channels = 20

                local_model = CNNFemnist(args=args)

            elif args.dataset == 'cifar100' or args.dataset == 'cifar10':
                if args.mode == 'model_heter':
                    if i<10:
                        args.stride = [1,4]
                    else:
                        args.stride = [2,2]
                else:
                    args.stride = [2, 2]
                resnet = resnet18(args, pretrained=False, num_classes=args.num_classes)
                initial_weight = model_zoo.load_url(model_urls['resnet18'])
                local_model = resnet
                initial_weight_1 = local_model.state_dict()
                for key in initial_weight.keys():
                     if key[0:3] == 'fc.' or key[0:5] == 'conv1' or key[0:3] == 'bn1':
                         initial_weight[key] = initial_weight_1[key]

                local_model.load_state_dict(initial_weight)

            local_model.to(args.device)
            local_model.train()
            local_model_list.append(local_model)
        FedProto(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list)

    elif args.alg == 'fedavg':
        args.out_channels = 20
        if args.dataset == 'mnist':
            local_model = CNNMnist(args=args)
        elif args.dataset == 'femnist':
            local_model = CNNFemnist(args=args)
        elif args.dataset == 'cifar10':
            args.stride = [2, 2]
            resnet = resnet18(args, pretrained=False, num_classes=args.num_classes)
            initial_weight = model_zoo.load_url(model_urls['resnet18'])
            local_model = resnet
            initial_weight_1 = local_model.state_dict()
            for key in initial_weight.keys():
                if key[0:3] == 'fc.' or key[0:5] == 'conv1' or key[0:3] == 'bn1':
                    initial_weight[key] = initial_weight_1[key]

            local_model.load_state_dict(initial_weight)

        local_model.to(args.device)
        local_model.train()
        global_weights = local_model.state_dict()
        FedAvg(args,train_dataset, test_dataset, user_groups, local_model, user_groups_lt)