import os
import numpy as np
from VM.lvis import Lvis
from VM.matcher import VisualMatcher


if __name__ == "__main__":

    dataset = Lvis()
    matcher = VisualMatcher()

    np.random.seed(6)

    task_num = 1
    num_per_task = 8

    all_rate = []
    
    test_mode = 'CLIP_V'
    test_mode = 'CLIP_K'
    # test_mode = 'CLIP_N'

    all_test = ['CLIP_N', 'CLIP_K', 'CLIP_V']
    all_test = ['CLIP_N']

    for test_mode in all_test:
        total_rate = []
        for data in dataset.random_test(task_num, num_per_task):
            source_list, target_list, label_list = data
            use_text = True

            if test_mode == 'CLIP_V':
                use_text = False
            elif test_mode == 'CLIP_K':
                label_list = dataset.cat_names

            source_ids, target_ids = matcher.match_images( source_list, target_list, label_list, use_text )
            match_rate = sum(source_ids == target_ids) / num_per_task
            total_rate.append(match_rate)

        rate = np.mean(total_rate)
        all_rate.append(rate)

    print( f"Total categories: {dataset.cat_num}" )
    print( f"    Task num    : {task_num}" )
    print( f"    Num pre task: {num_per_task}" )
    print( "-"*20 )
    for i in range(len(all_rate)):
        print("%7s: %.3f" % (all_test[i], all_rate[i]))

    A = 1
    

# Total categories: 414
#     Task num    : 200
#     Num pre task: 20
# --------------------
#  CLIP_N: 0.577
#  CLIP_K: 0.541
#  CLIP_V: 0.403


# Total categories: 414
#     Task num    : 500
#     Num pre task: 8
# --------------------
#  CLIP_N: 0.771
#  CLIP_K: 0.724
#  CLIP_V: 0.542

