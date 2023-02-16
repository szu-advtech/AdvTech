import os
import gc
import time
import pickle
import argparse
import numpy as np
from pympler import asizeof
from functools import reduce
from local_path import data_dir
from HEWrapper.seal_ckks import generate_pk_and_sk


# def loss(user_vector, item_vector, rate_data):
    # error = []
    # for record in rate_data:
        # u_id, v_id, rate = record
        # error.append((rate - np.dot(user_vector[u_id], item_vector[v_id])) ** 2)
    # return np.sqrt(np.mean(error))

def user_update(current_user_vector, visit_vector, rating_list, encrypted_item_vector, sk, pk, lr, reg_u, reg_v):
    rated_items = [e[1] for e in rating_list]
    for i in range(len(visit_vector)):
        if visit_vector[i] == 1 and i not in rated_items:
            rating_list.append([None, i, None])

    item_gradient = {}
    for u_id, v_id, rate in rating_list:
        if rate is not None:
            item_i_profile = sk.decrypt(encrypted_item_vector[v_id])[:hidden_dim]
            error = rate - np.dot(current_user_vector, item_i_profile)
            current_user_vector = current_user_vector - lr * (
                    -2 * error * item_i_profile + 2 * reg_u * current_user_vector)
            item_gradient[v_id] = (-1) * lr * (-2 * error * current_user_vector + 2 * reg_v * item_i_profile)
        else:
            item_gradient[v_id] = (-1) * np.zeros(len(current_user_vector))
    encrypted_gradient = {v_id: pk.encrypt(item_gradient[v_id]) for v_id in item_gradient}

    return current_user_vector, encrypted_gradient


if __name__ == '__main__':

    params_parser = argparse.ArgumentParser()
    params_parser.add_argument('--dataset', type=str, default='movie_lens')
    params_parser.add_argument('--visiting', type=str, default='user_visit')
    params_parser.add_argument('--iteration', type=int, default=1)
    args = params_parser.parse_args()

    print(args)

    band_width = 1
    repeat = 10
    hidden_dim = 100
    max_iteration = args.iteration
    poly_modulus_degree = 8192

    with open(os.path.join(data_dir, args.dataset + '.pkl'), 'rb') as f:
        data = pickle.load(f)
        user_visiting = data[args.visiting]

    public_key, private_key, evaluator = generate_pk_and_sk(poly_modulus_degree=poly_modulus_degree, lazy_rescale=False)

    # Init process
    user_vector = np.zeros([data['num_user'], hidden_dim])
    item_vector = np.zeros([data['num_item'], hidden_dim]) + 0.01

    # Cut the train and test data
    rating_date = [[[e, k[0], k[1]] for k in data['user_rate'][e]] for e in range(len(data['user_rate']))]
    rating_date = reduce(lambda x, y: x + y, rating_date)
    np.random.shuffle(rating_date)
    train_rating = rating_date[:(int(0.9*len(rating_date)))]
    test_rating = rating_date[(int(0.9*len(rating_date))):]

    # Step 1 Server encrypt item-vector
    t = time.time()
    encrypted_item_vector = [public_key.encrypt(vector) for vector in item_vector]
    print('Item profile encrypt using', time.time() - t, 'seconds')

    time_records = []

    for iteration in range(max_iteration):

        print('###################')
        print('Iteration', iteration)

        # Step 2 User updates
        cache_size = asizeof.asizeof(encrypted_item_vector[0]) * len(encrypted_item_vector)
        print('Size of Encrypted-item-vector', cache_size / (2 ** 20), 'MB')
        communication_time = cache_size * 8 / (band_width * 2 ** 30)
        print('Using a %s Gb/s' % band_width, 'bandwidth, communication will use %s second' % communication_time)

        encrypted_gradient_from_user = []
        user_time_list = []
        for i in range(data['num_user']):
            t = time.time()
            # current_user_vector, visit_vector, rating_list, encrypted_item_vector, sk, lr, reg_u, reg_v
            user_vector[i], gradient = user_update(
                user_vector[i], user_visiting[i], [e for e in train_rating if e[0] == i],
                encrypted_item_vector, private_key, public_key, lr=1e-3, reg_v=1e-4, reg_u=1e-4
            )
            user_time_list.append(time.time() - t)
            print('User-%s update using' % i, user_time_list[-1], 'seconds')
            encrypted_gradient_from_user.append(gradient)
        print('User Average time', np.mean(user_time_list))

        # Step 3 Server update
        cache_size = np.mean([np.sum([[asizeof.asizeof(value)] for key, value in g.items()])
                              for g in encrypted_gradient_from_user])
        print('Size of Encrypted-gradient', cache_size / (2 ** 20), 'MB')
        communication_time = communication_time + cache_size * 8 / (band_width * 2 ** 30)
        print('Using a %s Gb/s' % band_width, 'bandwidth, communication will use %s second' % communication_time)
        t = time.time()
        for i in range(len(encrypted_gradient_from_user)-1, -1, -1):
            g = encrypted_gradient_from_user[i]
            for item_id in g:
                # encrypted_item_vector[item_id] = encrypted_item_vector[item_id] + g[item_id]
                encrypted_item_vector[item_id] =evaluator.add(encrypted_item_vector[item_id], g[item_id])
            del encrypted_gradient_from_user[i]
            gc.collect()
        server_update_time = time.time() - t
        print('Server update using', server_update_time, 'seconds')

        # for computing loss
        item_vector = np.array([private_key.decrypt(vector)[:hidden_dim] for vector in encrypted_item_vector])
        # train_rmse = loss(user_vector=user_vector, item_vector=item_vector, rate_data=train_rating)
        # test_rmse = loss(user_vector=user_vector, item_vector=item_vector, rate_data=test_rating)
        # print('Train RMSE', train_rmse)
        # print('Test RMSE', test_rmse)
        print('Costing', max(user_time_list) + server_update_time + communication_time, 'seconds')
        time_records.append(max(user_time_list) + server_update_time + communication_time)

    result_file = 'efficiency_evaluation.csv'
    result_file = os.path.join('results', result_file)
    if not os.path.isfile(result_file):
        with open(result_file, 'w') as f:
            f.write('Dataset, X, Time\n')

    with open(result_file, 'a+') as f:
        f.write(', '.join([str(e) for e in [args.dataset, args.visiting, np.mean(time_records)]
                           + time_records]) + '\n')
