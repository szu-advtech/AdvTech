import platform
import numpy as np
import random
import classifier_config
from utils import global_to_standard

"""
    Loaded train data & train label. 
    Do not modify these fields outside dataset.py
"""
__train_data_list__ = []
__train_label_list__ = []
__test_data_list__ = []
__test_label_list__ = []

__NPY_DATA_DIR__ = "D:\\LearnAbstractionDataset\\"
if platform.system() != "Windows":
    __NPY_DATA_DIR__ = "./dataset/npy/"

categories = {
    'cat': 0,
    'chair': 1,
    'face': 2,
    'firetruck': 3,
    'mosquito': 4,
    'owl': 5,
    'pig': 6,
    'purse': 7,
    'shoe': 8
}

# map integer label back to name. now only used to record abstraction process
labels = dict(zip(categories.values(), categories.keys()))

__MAX_POINTS_NUM__: int = 151


def find_max_points_num() -> int:
    global __MAX_POINTS_NUM__
    if __MAX_POINTS_NUM__ != -1:
        return __MAX_POINTS_NUM__
    max_points_num = 0

    for sketch_data in __train_data_list__:
        points_num = len(sketch_data)
        if max_points_num < points_num:
            max_points_num = points_num
    for sketch_data in __test_data_list__:
        points_num = len(sketch_data)
        if max_points_num < points_num:
            max_points_num = points_num

    __MAX_POINTS_NUM__ = max_points_num
    return max_points_num


def load(num_per_category: int, num_training_per_category: int,
         verbose: bool = False, shuffle_each_category: bool = False,
         test_only: bool = False,
         load_cat_only: bool = False):
    """
    :param shuffle_each_category: perform shuffling on each category, used with positive num_training_per_category
    :param verbose: output debug info
    :param num_per_category: read a certain number of sketch images from each category. if < 0, then read all
    :param num_training_per_category: the number of train_data;
    num_training_per_category <= num_per_category must suffice.

    :return:
    """
    assert num_per_category > 0, AssertionError("num_per_category > 0 must suffice")
    assert num_training_per_category > 0, AssertionError("num_training_per_category > 0 must suffice")
    assert num_per_category >= num_training_per_category, AssertionError("num_training_per_category <= num_per_category 0 must suffice")
    global __train_data_list__, __train_label_list__, __test_data_list__, __test_label_list__
    if load_cat_only:
        global categories
        categories = {
            'cat': 0,
        }
    if len(__train_data_list__) < 1 and len(__test_data_list__) < 1:
        __load__(num_per_category, num_training_per_category, verbose, shuffle_each_category, test_only)

    return __train_data_list__, __train_label_list__, __test_data_list__, __test_label_list__


# read all sketches
def __load__(num_per_category: int, num_training_per_category: int,
             verbose: bool = False, shuffle_each_category: bool = False,
             test_only: bool = False) -> None:
    global __train_data_list__, __train_label_list__, __test_data_list__, __test_label_list__

    for key in categories.keys():
        if verbose:
            print("loading category", key)

        label = categories[key]

        if test_only is False:
            train_iterator = np.load(__NPY_DATA_DIR__ + key + '-train.npy', allow_pickle=True).item()
            train_iterator.raw_data, train_iterator.seq_len = None, None
            train_data_num = min(num_training_per_category, len(train_iterator.data_strokes))
            for i in range(train_data_num):
                sketch_np = global_to_standard(sketchBucket=train_iterator.data_strokes[i],
                                               sketchBucketLen=train_iterator.len_strokes[i])
                __train_data_list__.append(sketch_np)
            __train_label_list__.extend([label for _ in range(train_data_num)])

        test_iterator = np.load(__NPY_DATA_DIR__ + key + '-test.npy', allow_pickle=True).item()
        test_iterator.raw_data, test_iterator.seq_len = None, None
        test_data_num = min(num_per_category - num_training_per_category, len(test_iterator.data_strokes))

        for i in range(test_data_num):
            sketch_np = global_to_standard(sketchBucket=test_iterator.data_strokes[i],
                                           sketchBucketLen=test_iterator.len_strokes[i])
            __test_data_list__.append(sketch_np)
        __test_label_list__.extend([label for _ in range(test_data_num)])
    return


def shuffle_train_data(seed:int = 0) -> None:
    print("shuffling train data with seed", seed)
    random.seed(seed)
    random.shuffle(__train_data_list__)
    random.seed(seed)
    random.shuffle(__train_label_list__)
    return


def shuffle_test_data(seed:int = 0) -> None:
    print("shuffling test data with seed", seed)
    random.seed(seed)
    random.shuffle(__test_data_list__)
    random.seed(seed)
    random.shuffle(__test_label_list__)
    return


def __test__() -> None:
    load(shuffle_each_category=False, verbose=True,
         num_per_category=classifier_config.num_per_category(),
         num_training_per_category=classifier_config.num_training_per_category(),
         test_only=True)
    # max_points_num = find_max_points_num()
    # print(max_points_num)
    # points_num_distribution()
    dseg_num = 0
    for test_data_np in __test_data_list__:
        dseg_num += len(test_data_np)

    dseg_num /= len(__test_data_list__)
    print(dseg_num)
    return


if __name__ == '__main__':
    __test__()
