import numpy as np
import queue



def DHC(arrs):
    # q = queue.Queue()
    q = queue.LifoQueue()
    q.put(arrs)
    regions_arrs = []
    while not q.empty():
        arrs = q.get()
        if len(arrs) < 16 and free_dimension(arrs) <= 10:
            regions_arrs.append(arrs)
            continue
        splits = maxcovering(arrs)
        for s in splits:
            q.put(arrs[s])

    return regions_arrs


def maxcovering(arrs):
    Tarrs = arrs.T
    Covering = []
    leftmost_index = -1
    leftmost_Covering = -1
    for i in range(32):
        splits = np.bincount(Tarrs[i], minlength=16)
        if np.count_nonzero(splits) == 1:
            # fixed dimension
            Covering.append(-1)
        else:
            if leftmost_index == -1:
                leftmost_index = i
                leftmost_Covering = np.sum(splits*(splits != 1))
            Covering.append(np.sum(splits*(splits != 1)))

    index = np.argmax(Covering)
    if np.max(Covering) - leftmost_Covering <= index - leftmost_index:

        index = leftmost_index
    splits = np.bincount(Tarrs[index], minlength=16)

    split_nibbles = np.argwhere(splits).reshape(-1)

    return [
        np.argwhere(Tarrs[index] == nibble).reshape(-1) for nibble in split_nibbles
    ]



def free_dimension(arrs):
    Tarrs=arrs.T
    free_dimension_num=0
    for i in range(32):
        splits = np.bincount(Tarrs[i], minlength=16)
        if np.count_nonzero(splits) == 1:
            # fixed dimension
            continue
        free_dimension_num += 1

    return free_dimension_num


# show the regions for test

def show_regions(arrs):
    address_space = []
    Tarrs = arrs.T
    for i in range(32):
        splits = np.bincount(Tarrs[i], minlength=16)
        # print(i, splits, np.argwhere(splits > 0)[0][0])
        if len(splits[splits > 0]) == 1:

            address_space.append(format(np.argwhere(splits > 0)[0][0], "x"))
        else:
            address_space.append("*")

    print("********Region**********")
    print("".join(address_space))
    for i in range(len(arrs)):
        print("".join([format(x, "x") for x in arrs[i]]), " ", i)
    print()


if __name__ == "__main__":

    data = np.load("./seeds.npy")

    results = DHC(data)
    for r in results:
        show_regions(r)
    space = 0
    for r in results:
        i = 0
        address_space = []
        Tarrs = r.T
        for i in range(32):
            splits = np.bincount(Tarrs[i], minlength=16)
            if len(splits[splits > 0]) > 1:
                i = i+1
        space += 16**i
    print(space)
