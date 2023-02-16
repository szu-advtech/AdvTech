from SpacePartition import *
from OutlierDetection import *
from CreateTarget import create_target
from Tostr import tostr


if __name__ == "__main__":

    data = np.load("./seeds.npy")

    outliers = []
    patterns = []
    results = DHC(data)

    for r in results:

        p, o = OutlierDetect(r)
        patterns += p
        outliers += o
        # show_regions(p)
    tr=create_target(patterns)
    print('****************')
    tostr(tr)

