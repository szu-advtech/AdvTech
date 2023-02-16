import random
import numpy as np


def schedule(all_client, Q_client, rounds):
    scheduler = []
    for i in range(rounds):
        scheduler.append(random.sample(range(all_client), 2 * Q_client))
    scheduler = np.asarray(scheduler)
    scheduler = scheduler.reshape(2 * rounds, Q_client)
    return scheduler


def iid_divide(dataset_length, users_number, items_number):
    users_dict, all_index = {}, [i for i in range(dataset_length)]
    for i in range(users_number):
        users_dict[i] = set(np.random.choice(all_index, items_number, replace=False))
        all_index = list(set(all_index) - users_dict[i])
    return users_dict


def iid_divide_stable(dataset_length, items_number):
    all_index = [i for i in range(dataset_length)]
    return np.random.choice(all_index, items_number, replace=False)


def non_iid_divide(dataset, users_number, heterogeneity, data_amount):
    dataset_length = len(dataset)
    classes = 10

    image_number = 100
    shards_number = int(data_amount / image_number)
    num_shards = int(dataset_length / image_number)
    class_shards = int(num_shards / classes)

    dict_users = {i: np.array([]) for i in range(users_number)}
    indexes = np.arange(num_shards * image_number).astype(int)
    labels = dataset.train_labels.numpy()

    # sort labels
    indexes_labels = np.vstack((indexes, labels))
    indexes_labels = indexes_labels[:, indexes_labels[1, :].argsort()]
    indexes = indexes_labels[0, :]

    # divide and assign heterogeneity shards/client
    for i in range(users_number):
        # get rand_set
        class_select = random.sample(range(1, classes), heterogeneity)
        choose_shard = []
        for j in class_select:
            choose_shard = np.append(choose_shard, [k for k in range((j - 1) * class_shards, j * class_shards)])
        rand_set = set(np.random.choice(choose_shard, shards_number, replace=False))
        for rand in rand_set:
            rand = int(rand)
            dict_users[i] = np.concatenate(
                (dict_users[i], indexes[rand * image_number:(rand + 1) * image_number]), axis=0)
    return dict_users
