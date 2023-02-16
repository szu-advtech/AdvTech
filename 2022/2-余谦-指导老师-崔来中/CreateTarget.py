import numpy as np
import random


def create_target(patterns):
    target = []
    for arrs in patterns:
        free_pos = []
        free_dimension_num = 0
        Tarrs = arrs.T
        for i in range(32):
            splits = np.bincount(Tarrs[i], minlength=16)
            # print(i, splits, np.argwhere(splits > 0)[0][0])
            if len(splits[splits > 0]) == 1:
                continue
            else:
                free_pos.append(i)
                free_dimension_num += 1

        target_address = []
        while len(target_address) < min(5, 16 ** free_dimension_num-len(arrs)):
            ip = arrs[0].copy()
            for i in range(free_dimension_num):
                nibble = random.randint(0, 15)
                ip[free_pos[i]] = nibble

            flag = True
            for address in target_address:
                if all(address==ip):
                    flag=False
                    break
            for address in arrs:
                if all(address==ip):
                    flag=False
                    break
            if flag:
                target_address.append(ip)

        target.extend(target_address)

    return target