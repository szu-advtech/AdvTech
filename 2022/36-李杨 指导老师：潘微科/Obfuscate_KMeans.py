import os
import copy
import pickle
import argparse
import numpy as np
import cvxpy as cp
from local_path import data_dir
from sklearn.cluster import KMeans


def random_increase_obfuscate(visiting_matrix, p):
    obfuscated_visiting_matrix = copy.deepcopy(visiting_matrix)
    num_user, num_item = visiting_matrix.shape
    for i in range(num_user):
        # negative_items_site
        negative_items = np.where(visiting_matrix[i] == 0)[0]
        obfuscate_lens = min(len(negative_items), int(num_item * p))
        obfuscated_visiting_matrix[i][np.random.choice(negative_items, obfuscate_lens, replace=False)] = 1
    return obfuscated_visiting_matrix


def random_flip_obfuscate(visiting_matrix, p):
    obfuscated_visiting_matrix = copy.deepcopy(visiting_matrix)
    num_user, num_item = visiting_matrix.shape
    for i in range(num_user):
        for j in range(num_item):
            if np.random.random() < p:
                obfuscated_visiting_matrix[i][j] = (obfuscated_visiting_matrix[i][j] + 1) % 2
    return obfuscated_visiting_matrix



def cvx_obfuscate(visiting_matrix, label_dict, num_cluster=10, epsilon=29):

    # (1) Clustering using KMeans
    clustering_model = KMeans(n_clusters=num_cluster, random_state=2, tol=1e-10).fit(visiting_matrix)
    cluster_labels = clustering_model.labels_
    cluster_centers = clustering_model.cluster_centers_
    cluster_centers[np.where(cluster_centers < 0)] = 0
    print(np.eye(num_cluster)[cluster_labels].sum(0))

    # Accuracy and efficiency losses
    def safe_div(v1, v2):
        result = []
        for i in range(len(v1)):
            if v1[i] == 0 or v2[i] == 0:
                result.append(0)
            else:
                result.append(v1[i] / v2[i])
        return np.array(result)

    accuracy_loss = []
    for i in range(len(cluster_centers)):
        tmp_dist = []
        for j in range(len(cluster_centers)):
            middle = (cluster_centers[i] - cluster_centers[j]) * (cluster_centers[i] > cluster_centers[j])
            tmp_dist.append(np.mean(safe_div(middle, cluster_centers[i])))
        accuracy_loss.append(tmp_dist)
    accuracy_loss = np.array(accuracy_loss)

    efficiency_loss = []
    for i in range(len(cluster_centers)):
        tmp_dist = []
        for j in range(len(cluster_centers)):
            tmp_dist.append((cluster_centers[j].sum() - cluster_centers[i].sum()) / cluster_centers[i].sum())
        efficiency_loss.append(tmp_dist)
    efficiency_loss = np.array(efficiency_loss)

    p_x = np.eye(num_cluster)[cluster_labels].sum(0) / np.eye(num_cluster)[cluster_labels].sum()
    p_x_hat_given_x = cp.Variable((num_cluster, num_cluster))

    # Collect the information of different attributes
    attr_records = []
    for attr in label_dict:
        # Get p_y and p_x_y
        p_y = label_dict[attr].sum(0) / label_dict[attr].sum()
        p_x_y = [label_dict[attr][np.where(cluster_labels == e)] for e in range(num_cluster)]
        p_x_y = np.array([e.sum(0) / label_dict[attr].sum() for e in p_x_y])
        # Record the problem
        p_x_hat_y = p_x_hat_given_x.T @ p_x_y
        p_x_hat = p_x_hat_given_x.T @ p_x
        attr_records.append([p_y, p_x_hat, p_x_hat_y])

    # Solve the convex optimization
    tmp_p_x_hat = p_x
    while True:
        mutual_information = []
        for i in range(len(attr_records)):
            p_y, p_x_hat, p_x_hat_y = attr_records[i]
            mutual_information.append(
                cp.sum(cp.entr(p_y)) + cp.sum(-cp.multiply(tmp_p_x_hat, cp.log(p_x_hat)))
                - cp.sum(cp.entr(p_x_hat_y))
            )
        constraints = [
            0 <= p_x_hat_given_x, p_x_hat_given_x <= 1,
            cp.sum(p_x_hat_given_x, axis=1) == 1,
            cp.sum(cp.multiply(p_x_hat_given_x, accuracy_loss), axis=0) <= 0.2,
            cp.sum(cp.multiply(p_x_hat_given_x, efficiency_loss), axis=0) <= 1.0,
            ]

        problem = cp.Problem(cp.Minimize(cp.sum(mutual_information)), constraints)
        problem.solve()

        diff = np.mean(np.abs(tmp_p_x_hat - p_x_hat.value))
        print('Diff', diff)
        if diff < 10e-10:
            break
        tmp_p_x_hat = p_x_hat_given_x.T.value @ p_x

    obfuscation_matrix = p_x_hat_given_x.value

    # Add DP Noise to each cluster instance
    if epsilon != 'INF':
        delta_f = []
        for i in range(num_cluster):
            items = visiting_matrix[np.where(cluster_labels == i)]
            delta_f_tmp = []
            for j in range(len(items)):
                for num_cluster in range(j, len(items)):
                    delta_f_tmp.append(np.sum(np.abs(items[j] - items[num_cluster])))
            delta_f.append(np.max(delta_f_tmp))
        cluster_b = [e / epsilon for e in delta_f]
        noise_prob = [np.exp(-1/b) / (1 + np.exp(-1/b)) for b in cluster_b]
        # Add noise
        r = np.random.random(visiting_matrix.shape)
        for i in range(len(visiting_matrix)):
            r[i] = (r[i] <
                    np.array([1-noise_prob[cluster_labels[i]],
                             noise_prob[cluster_labels[i]]])[visiting_matrix[i].astype(np.int32)]).astype(np.int16)
        noise_visiting_matrix = (visiting_matrix + r) % 2
    else:
        noise_visiting_matrix = visiting_matrix

    # Generate the obfuscate results
    obfuscate_result = []
    for i in range(len(visiting_matrix)):
        current_cluster = cluster_labels[i]
        # Ob to another cluster
        obfuscate_cluster = np.random.choice(list(range(obfuscation_matrix.shape[0])),
                                             size=1, p=obfuscation_matrix[current_cluster])
        # Random choose one user in this cluster
        obfuscate_result.append(
            noise_visiting_matrix[np.random.choice(np.where(cluster_labels == obfuscate_cluster)[0], size=1)[0]]
        )
    obfuscate_result = np.array(obfuscate_result)
    return obfuscate_result


if __name__ == '__main__':

    params_parser = argparse.ArgumentParser()
    params_parser.add_argument('--dataset', type=str, default='movie_lens')
    args = params_parser.parse_args()

    # (1) Load the data
    with open(os.path.join(data_dir, args.dataset + '.pkl'), 'rb') as f:
        data = pickle.load(f)

    modify = False

    # (2) Random Obfuscate
    random_obfuscate_p = [e * 0.1 for e in range(1, 10, 2)]
    for p in random_obfuscate_p:
        key = 'random_increase_obfuscate_%.1f' % p
        if key not in data:
            print(key)
            data[key] = random_increase_obfuscate(data['user_visit'], p)
            modify = True
    for p in random_obfuscate_p:
        key = 'random_flip_obfuscate_%.1f' % p
        if key not in data:
            print(key)
            data[key] = random_flip_obfuscate(data['user_visit'], p)
            modify = True

    # (3) CVX obfuscate
    old_cvx = [e for e in data if 'cvx' in e]
    for key in old_cvx:
        data.pop(key)
    for epsilon in ['INF', 10, 1, 0.1, 0.01]:
        key = 'kmeans_cvx_obfuscate_all_e%s' % epsilon
        if key not in data:
            print(key)
            data[key] = cvx_obfuscate(data['user_visit'], data['attributes'], epsilon=epsilon)
            modify = True
        if len(data['attributes']) > 1:
            for attr in data['attributes']:
                key = 'kmeans_cvx_obfuscate_%s_e%s' % (attr, epsilon)
                if key not in data:
                    print(key)
                    data[key] = cvx_obfuscate(data['user_visit'], {attr: data['attributes'][attr]}, epsilon=epsilon)
                    modify = True

    if modify:
        print('saving new file')
        with open(os.path.join(data_dir, args.dataset + '.pkl'), 'wb') as f:
            pickle.dump(data, f)
