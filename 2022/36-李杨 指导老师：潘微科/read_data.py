import os
import pickle
import numpy as np

from scipy.sparse import lil_matrix
from local_path import ml_dir, data_dir

# Load data from file
with open(os.path.join(ml_dir, 'u.data')) as f:
    u_data = [e.strip('\n').split('\t') for e in f.readlines()]

with open(os.path.join(ml_dir, 'u.item'), encoding='utf-8') as f:
    u_item = [e.strip('\n').split('|') for e in f.readlines()]
    u_item_cate_dict = {e[0]: np.array([int(c) for c in e[5:]]) for e in u_item}

with open(os.path.join(ml_dir, 'u.user')) as f:
    u_user = [e.strip('\n').split('|') for e in f.readlines()]

# Statistical frequency over items
item_visiting_freq = {}
for record in u_data:
    item_visiting_freq[record[1]] = item_visiting_freq.get(record[1], 0) + 1

# Filter out the rating data according to the item visiting frequency
item_set = list(sorted(set([e[1] for e in u_data])))
user_set = list(sorted(set([e[0] for e in u_data])))
u_attr = {e[0]: e[1:] for e in u_user if e[0] in user_set}
print('Number of users', len(user_set))
print('Number of items', len(item_set))
print('Number of ratings', len(u_data))
occupation_set = list(sorted(set([e[3] for e in u_user if e[0] in user_set])))


def gender_encoder(value):
    return np.array([1, 0]) if value.lower().startswith('m') else np.array([0, 1])


def age_encoder(value):
    value = int(value)
    if value <= 18:
        return np.array([1, 0, 0, 0, 0])
    elif 18 < value <= 30:
        return np.array([0, 1, 0, 0, 0])
    elif 30 < value <= 40:
        return np.array([0, 0, 1, 0, 0])
    elif 40 < value <= 50:
        return np.array([0, 0, 0, 1, 0])
    elif 50 < value:
        return np.array([0, 0, 0, 0, 1])


def occ_encoder(value):
    result = np.zeros([len(occupation_set)])
    result[occupation_set.index(value)] = 1
    return result


u_rating = {}
for record in u_data:
    u_id, i_id, rating, timestamp = record
    u_rating[u_id] = u_rating.get(u_id, []) + [[i_id, int(rating)]]

u_visiting = {}
for u_id in u_rating:
    rating = np.zeros([len(item_set), ], dtype=np.float32)
    for i in u_rating[u_id]:
        rating[item_set.index(i[0])] = 1
    u_visiting[u_id] = rating

collected_gender = np.zeros([len(item_set), 2])
collected_age = np.zeros([len(item_set), 5])
collected_occ = np.zeros([len(item_set), len(occupation_set)])

gender_prior = []
age_prior = []
occ_prior = []

sparse_user_matrix = lil_matrix((len(user_set), len(item_set)), dtype=np.int8)
user_label = []

for u_id in user_set:
    visiting_vector = u_visiting[u_id]
    gender_code = gender_encoder(u_attr[u_id][1])
    age_code = age_encoder(u_attr[u_id][0])
    occ_code = occ_encoder(u_attr[u_id][2])

    gender_prior.append(gender_code)
    age_prior.append(age_code)
    occ_prior.append(occ_code)

    collected_gender += np.dot(visiting_vector.reshape([-1, 1]), gender_code.reshape([1, -1]))
    collected_age += np.dot(visiting_vector.reshape([-1, 1]), age_code.reshape([1, -1]))
    collected_occ += np.dot(visiting_vector.reshape([-1, 1]), occ_code.reshape([1, -1]))

gender_prior = np.array(gender_prior, dtype=np.int32)
age_prior = np.array(age_prior, dtype=np.int32)
occ_prior = np.array(occ_prior, dtype=np.int32)

result_data = {
    'num_user': len(user_set),
    'num_item': len(item_set),
    'attributes': {
        'gender': gender_prior,
        'age': age_prior,
        'occupation': occ_prior
    },
    'user_rate': [[[item_set.index(e1[0]), e1[1]] for e1 in u_rating[e]] for e in user_set],
    'user_visit': np.array([u_visiting[e] for e in user_set]),
}

with open(os.path.join(data_dir, 'movie_lens.pkl'), 'wb') as f:
    pickle.dump(result_data, f)
