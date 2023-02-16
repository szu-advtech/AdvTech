
import numpy as np
import queue

from SpacePartition import maxcovering, show_regions


def seed_distance(a, b):
    return len(np.argwhere(a != b))


def IoslatedForest(Weights, splits, Tarr_i):
    OutlierNum = np.sum(splits == 1)

    # unique nibbles
    for j in np.argwhere(splits == 1).reshape(-1):
        outlier_index = np.argwhere(Tarr_i == j).reshape(-1)[0]
        Weights[outlier_index] += 1/OutlierNum


def Four_D(Weights):
    if len(Weights) <= 2:
        return []
    OutLierIndex = np.argmax(Weights)

    OutRemovedWeights = list(Weights)
    OutRemovedWeights.remove(Weights[OutLierIndex])
    OutRemovedD = np.sqrt(np.var(OutRemovedWeights))
    OutRemovedAvg = np.average(OutRemovedWeights)

    if Weights[OutLierIndex] - OutRemovedAvg > 3*OutRemovedD:
        return [Weights[OutLierIndex]] + Four_D(OutRemovedWeights)
    else:
        return []


def iter_devide(arrs):
    q = queue.LifoQueue()
    q.put(arrs)
    regions_arrs = []
    while not q.empty():
        arrs = q.get()
        splits = maxcovering(arrs)
        if 1 in [len(s) for s in splits]:
            regions_arrs.append(arrs)
        else:
            for s in splits:
                q.put(arrs[s])

    return regions_arrs


def OutlierDetect(arrs):
    if len(arrs) == 1:
        return [], [arrs]

    if len(arrs) == 2:
        if seed_distance(arrs[0], arrs[1]) > 12:
            return [], [arrs]
        else:
            return [arrs], []

    Tarrs = arrs.T
    free_dimension_num = 0
    Weights = [0]*len(arrs)
    # Forest

    for i in range(32):
        splits = np.bincount(Tarrs[i], minlength=16)
        if np.count_nonzero(splits) == 1:
            # fixed dimension
            continue
        free_dimension_num += 1
        IoslatedForest(Weights, splits, Tarrs[i])

    #show_regions(arrs)
    OutlierIndices = []
    for oW in Four_D(Weights):
        OutlierIndices.append(np.where(Weights == oW)[0][0])
    region = arrs[list(set(list(range(len(arrs))))-set(OutlierIndices))]
    outliers = arrs[OutlierIndices]

    patterns = iter_devide(region)
    # for p in patterns:
    #     show_regions(p)
    #
    # print("-"*90)
    return patterns, [outliers]


# for test
def showRegionAndOutliers(region, outliers):
    print("********RegionAndOutliers**********")
    print("-------------Region----------------")
    address_space = []
    Tarrs = region.T
    for i in range(32):
        splits = np.bincount(Tarrs[i], minlength=16)
        if len(splits[splits > 0]) == 1:

            address_space.append(format(
                np.argwhere(splits > 0)[0][0], "x"))
        else:
            address_space.append("*")
    print("".join(address_space))
    for i in range(len(region)):
        print("".join([format(x, "x") for x in region[i]]))

    print()

    print("-------------Outliers--------------")
    for o in outliers:
        print("".join([format(x, "x") for x in o]))
    print()


if __name__ == "__main__":

    ss = ["2a0e1bc0009700000000000000000001",
          "2a0e2400053f00000000000000000001",
          "2a0e04090c820000021132fffee5b604",
          "2a0e04090c8200000000000000000001",
          "2a0e8f02212f00000000000000000001"]
    arrs = np.array([[int(i, 16)for i in s] for s in ss]).astype("int")



    OutlierDetect(arrs)