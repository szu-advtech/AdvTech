import pandas as pd
import numpy as np

if __name__ == '__main__':
    imagenet_10 = pd.read_table('data/imagenet_subsets/imagenet_10.txt', sep='\n', header=None)
    imagenet_class = pd.read_table('data/imagenet_subsets/imagenet-classes.txt', sep='], ', header=None)
    imagenet_10 = np.array(imagenet_10)
    imagenet_class = np.array(imagenet_class)
    imagenet_class = imagenet_class.reshape(-1)

    collect = []
    for i in imagenet_10:
        for j in imagenet_class:
            if str(i).strip('[\'').strip('\']') in str(j):
                collect.append(j[8:])
                break

    collect = np.array(collect).reshape(-1,1)