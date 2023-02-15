import os
import time
import copy
import pickle
import argparse
import numpy as np
import datetime
from functools import reduce
from local_path import data_dir
from sklearn.model_selection import KFold


def iterate(user_vector, item_vector, rate_date, lr, reg_u, reg_v):
    # User updates
    for record in rate_date:
        u_id, v_id, rate = record
        error = rate - np.dot(user_vector[u_id], np.transpose(item_vector[v_id]))
        user_vector[u_id] -= lr * (-2 * error * item_vector[v_id] + 2 * reg_u * user_vector[u_id])
        item_vector[v_id] -= lr * (-2 * error * user_vector[u_id] + 2 * reg_v * item_vector[v_id])
    return user_vector, item_vector


def loss(user_vector, item_vector, rate_data):
    error = []
    for record in rate_data:
        u_id, v_id, rate = record
        error.append((rate - np.dot(user_vector[u_id], item_vector[v_id])) ** 2)
    return np.sqrt(np.mean(error))


if __name__ == '__main__':

    params_parser = argparse.ArgumentParser()
    params_parser.add_argument('--dataset', type=str, default='movie_lens')
    args = params_parser.parse_args()

    dataset = args.dataset

    hidden_dim = 100
    max_iteration = 1000
    repeat = 1

    with open(os.path.join(data_dir, dataset + '.pkl'), 'rb') as f:
        data = pickle.load(f)

    rating_date = [[[e, k[0], k[1]] for k in data['user_rate'][e]] for e in range(len(data['user_rate']))]
    rating_date = reduce(lambda x, y: x+y, rating_date)
    np.random.shuffle(rating_date)

    obfuscate_keys = ['user_visit'] + [e for e in list(data.keys()) if 'obfuscate' in e]
    print('Running evaluation in recommendation error, using', obfuscate_keys)
    recommend_acc = {key: [] for key in obfuscate_keys}

    for _ in range(repeat):

        kf = KFold(n_splits=10, shuffle=True)

        for train, test in kf.split(rating_date):

            for obfuscate_key in obfuscate_keys:

                cvm = data[obfuscate_key]
                train_rating = [rating_date[e] for e in train if cvm[rating_date[e][0]][rating_date[e][1]] == 1]
                test_rating = [rating_date[e] for e in test]

                user_vector = np.zeros([data['num_user'], hidden_dim])
                item_vector = np.zeros([data['num_item'], hidden_dim]) + 0.01

                for iteration in range(max_iteration):

                    print('#################################')
                    print('Repeat', _, obfuscate_key, 'Iteration', iteration)

                    tmp_user_vector = copy.deepcopy(user_vector)
                    tmp_item_vector = copy.deepcopy(item_vector)

                    t = time.time()
                    user_vector, item_vector = iterate(
                        user_vector=user_vector, item_vector=item_vector,
                        rate_date=train_rating, lr=1e-3, reg_u=1e-4, reg_v=1e-4
                    )
                    print('Time', time.time() - t, 's')

                    print('Train RMSE', loss(user_vector=user_vector, item_vector=item_vector, rate_data=train_rating))
                    print('Test RMSE', loss(user_vector=user_vector, item_vector=item_vector, rate_data=test_rating))

                    user_diff = np.mean(np.abs(user_vector - tmp_user_vector))
                    item_diff = np.mean(np.abs(item_vector - tmp_item_vector))

                    print('Difference User', user_diff, 'Item', item_diff)

                    if (user_diff < 1e-4) and (item_diff < 1e-4):
                        print('Converged')
                        train_rmse = loss(user_vector=user_vector, item_vector=item_vector, rate_data=train_rating)
                        test_rmse = loss(user_vector=user_vector, item_vector=item_vector, rate_data=test_rating)
                        print('Converged Train RMSE', train_rmse)
                        print('Converged Test RMSE', test_rmse)
                        recommend_acc[obfuscate_key].append([train_rmse, test_rmse])
                        break

    file_name = 'recommend_acc.csv')
    file_name = os.path.join('results', file_name)
    recommend_acc = [[key] + np.mean(recommend_acc[key], axis=0).tolist() for key in recommend_acc]
    with open(file_name, 'w') as f:
        f.writelines([', '.join([str(e1) for e1 in e]) + '\n' for e in recommend_acc])
