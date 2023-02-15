# Process the CelebA dataset 

import numpy as np
import os
import random

attr2idx = {}
idx2attr = {}
selected_attrs = {'Black_Hair', 'Blond_Hair', 'Eyeglasses', 'Male'}
dataset = []

# borrow code from StarGAN
def preprocess(attr_path):
    """Preprocess the CelebA attribute file."""
    lines = [line.rstrip() for line in open(attr_path, 'r')]
    all_attr_names = lines[1].split()
    for i, attr_name in enumerate(all_attr_names):
        attr2idx[attr_name] = i
        idx2attr[i] = attr_name

    lines = lines[2:]
    random.seed(1234)
    random.shuffle(lines)
    for i, line in enumerate(lines):
        print(line)
        split = line.split()
        filename = split[0]
        values = split[1:]

        label = []
        for attr_name in selected_attrs:
            idx = attr2idx[attr_name]
            label.append(values[idx] == '1')

        dataset.append([[filename], label])
    np.save('dataset.npy', dataset)

    print('Finished preprocessing the CelebA dataset...')

preprocess('list_attr_celeba.txt')