import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal, mnist_noniid_lt
from sampling import femnist_iid, femnist_noniid, femnist_noniid_unequal, femnist_noniid_lt
from sampling import cifar_iid, cifar10_noniid, cifar10_noniid_lt
import femnist

trans_cifar10_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
trans_cifar10_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
trans_cifar100_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                               std=[0.267, 0.256, 0.276])])
trans_cifar100_val = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                              std=[0.267, 0.256, 0.276])])

def get_dataset(args, n_list, k_list):
    data_dir = args.data_dir + args.dataset
    if args.dataset == 'mnist':
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)
        if args.iid:
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            if args.unequal:
                user_groups = mnist_noniid_unequal(args, train_dataset, args.num_users)
            else:
                user_groups, classes_list = mnist_noniid(args, train_dataset, args.num_users, n_list, k_list)
                user_groups_lt = mnist_noniid_lt(args, test_dataset, args.num_users, n_list, k_list, classes_list)
                classes_list_gt = classes_list

    elif args.dataset == 'femnist':
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = femnist.FEMNIST(args, data_dir, train=True, download=True,
                                        transform=apply_transform)
        test_dataset = femnist.FEMNIST(args, data_dir, train=False, download=True,
                                       transform=apply_transform)

        if args.iid:
            user_groups = femnist_iid(train_dataset, args.num_users)
        else:
            if args.unequal:
                user_groups = femnist_noniid_unequal(args, train_dataset, args.num_users)
            else:
                user_groups, classes_list, classes_list_gt = femnist_noniid(args, args.num_users, n_list, k_list)
                user_groups_lt = femnist_noniid_lt(args, args.num_users, classes_list)

    elif args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=trans_cifar10_train)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=trans_cifar10_val)

        if args.iid:
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            if args.unequal:
                raise NotImplementedError()
            else:
                user_groups, classes_list, classes_list_gt = cifar10_noniid(args, train_dataset, args.num_users, n_list, k_list)
                user_groups_lt = cifar10_noniid_lt(args, test_dataset, args.num_users, n_list, k_list, classes_list)

    return train_dataset, test_dataset, user_groups, user_groups_lt, classes_list, classes_list_gt

def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def average_weights_het(w):
    w_avg = copy.deepcopy(w)
    for key in w[0].keys():
        if key[0:4] != 'fc2.':
            for i in range(1, len(w)):
                w_avg[0][key] += w[i][key]
            w_avg[0][key] = torch.div(w_avg[0][key], len(w))
            for i in range(1, len(w)):
                w_avg[i][key] = w_avg[0][key]
    return w_avg

def agg_func(protos):
    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos

def proto_aggregation(local_protos_list):
    agg_protos_label = dict()
    for idx in local_protos_list:
        local_protos = local_protos_list[idx]
        for label in local_protos.keys():
            if label in agg_protos_label:
                agg_protos_label[label].append(local_protos[label])
            else:
                agg_protos_label[label] = [local_protos[label]]

    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos_label[label] = [proto / len(proto_list)]
        else:
            agg_protos_label[label] = [proto_list[0].data]

    return agg_protos_label


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.rounds}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.train_ep}\n')
    return