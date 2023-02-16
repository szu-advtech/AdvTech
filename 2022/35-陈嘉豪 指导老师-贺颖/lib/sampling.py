import numpy as np
import random

def mnist_iid(dataset, num_users):
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def mnist_noniid(args, dataset, num_users, n_list, k_list):
    num_shards, num_imgs = 10, 6000
    idx_shard = [i for i in range(num_shards)]
    dict_users = {}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    label_begin = {}
    cnt=0

    for i in idxs_labels[1,:]:
        if i not in label_begin:
                label_begin[i] = cnt
        cnt+=1

    classes_list = []
    for i in range(num_users):
        n = n_list[i]
        k = k_list[i]
        k_len = args.train_shots_max

        classes = random.sample(range(0,args.num_classes), n)
        classes = np.sort(classes)
        print("user {:d}: {:d}-way {:d}-shot".format(i + 1, n, k))
        print("classes:", classes)
        user_data = np.array([])
        for each_class in classes:
            begin = i * k_len + label_begin[each_class.item()]
            user_data = np.concatenate((user_data, idxs[begin : begin+k]),axis=0)
        dict_users[i] = user_data
        classes_list.append(classes)

    return dict_users, classes_list

def mnist_noniid_lt(args, test_dataset, num_users, n_list, k_list, classes_list):
    num_shards, num_imgs = 10, 1000
    idx_shard = [i for i in range(num_shards)]
    dict_users = {}
    idxs = np.arange(num_shards*num_imgs)
    labels = test_dataset.train_labels.numpy()
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    label_begin = {}
    cnt=0
    for i in idxs_labels[1,:]:
        if i not in label_begin:
                label_begin[i] = cnt
        cnt+=1

    for i in range(num_users):
        k = 40
        classes = classes_list[i]
        print("local test classes:", classes)
        user_data = np.array([])
        for each_class in classes:
            begin = i*40 + label_begin[each_class.item()]
            user_data = np.concatenate((user_data, idxs[begin : begin+k]),axis=0)
        dict_users[i] = user_data
    return dict_users

def mnist_noniid_unequal(dataset, num_users):
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    min_shard = 1
    max_shard = 30

    random_shard_size = np.random.randint(min_shard, max_shard+1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    if sum(random_shard_size) > num_shards:
        for i in range(num_users):
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size-1

        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            shard_size = len(idx_shard)
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

    return dict_users

def femnist_iid(dataset, num_users):
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def femnist_noniid(args, num_users, n_list, k_list):
    dict_users = {}
    classes_list = []
    classes_list_gt = []

    for i in range(num_users):
        n = n_list[i]
        k = k_list[i]
        k_len = args.train_shots_max
        classes = random.sample(range(0, args.num_classes), n)
        classes = np.sort(classes)
        print("user {:d}: {:d}-way {:d}-shot".format(i + 1, n, k))
        print("classes:", classes)
        print("classes_gt:", classes)
        user_data = np.array([])
        for class_idx in classes:
            begin = class_idx * k_len * num_users + i * k_len
            user_data = np.concatenate((user_data, np.arange(begin, begin + k)),axis=0)
        dict_users[i] = user_data
        classes_list.append(classes)
        classes_list_gt.append(classes)

    return dict_users, classes_list, classes_list_gt

def femnist_noniid_lt(args, num_users, classes_list):
    dict_users = {}

    for i in range(num_users):
        k = args.test_shots
        classes = classes_list[i]
        user_data = np.array([])
        for class_idx in classes:
            begin = class_idx * k * num_users + i * k
            user_data = np.concatenate((user_data, np.arange(begin, begin + k)), axis=0)
        dict_users[i] = user_data

    return dict_users

def femnist_noniid_unequal(dataset, num_users):
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    min_shard = 1
    max_shard = 30

    random_shard_size = np.random.randint(min_shard, max_shard+1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size-1

        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            shard_size = len(idx_shard)
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

    return dict_users

def cifar10_noniid(args, dataset, num_users, n_list, k_list):
    num_shards, num_imgs = 10, 5000
    dict_users = {}
    idxs = np.arange(num_shards * num_imgs)
    labels = np.array(dataset.targets)
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    label_begin = {}
    cnt = 0
    for i in idxs_labels[1, :]:
        if i not in label_begin:
            label_begin[i] = cnt
        cnt += 1

    classes_list = []
    classes_list_gt = []
    k_len = args.train_shots_max
    for i in range(num_users):
        n = n_list[i]
        k = k_list[i]
        classes = random.sample(range(0, args.num_classes), n)
        classes = np.sort(classes)
        print("user {:d}: {:d}-way {:d}-shot".format(i + 1, n, k))
        print("classes:", classes)
        user_data = np.array([])
        for each_class in classes:
            begin = i * k_len + label_begin[each_class.item()]
            user_data = np.concatenate((user_data, idxs[begin: begin + k]), axis=0)
        dict_users[i] = user_data
        classes_list.append(classes)
        classes_list_gt.append(classes)

    return dict_users, classes_list, classes_list_gt

def cifar10_noniid_lt(args, test_dataset, num_users, n_list, k_list, classes_list):
    num_shards, num_imgs = 10, 1000
    dict_users = {}
    idxs = np.arange(num_shards*num_imgs)
    labels = np.array(test_dataset.targets)

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    label_begin = {}
    cnt=0
    for i in idxs_labels[1,:]:
        if i not in label_begin:
                label_begin[i] = cnt
        cnt+=1

    for i in range(num_users):
        k = args.test_shots
        classes = classes_list[i]
        print("local test classes:", classes)
        user_data = np.array([])
        for each_class in classes:
            begin = i * k + label_begin[each_class.item()]
            user_data = np.concatenate((user_data, idxs[begin : begin+k]),axis=0)
        dict_users[i] = user_data


    return dict_users


def cifar_iid(dataset, num_users):
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users