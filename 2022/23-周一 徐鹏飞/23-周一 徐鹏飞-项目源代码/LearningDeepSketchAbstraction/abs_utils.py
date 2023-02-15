import random
import sys
import time
from matplotlib import pyplot as plt


class Timer(object):
    def __init__(self):
        self.started: bool = False
        self.T1 = -1
        self.T2 = -1
        return

    def start(self) -> None:
        assert self.started is False
        self.started = True
        self.T1 = time.perf_counter()
        return

    def end(self) -> float:
        assert self.started is True
        self.T2 = time.perf_counter()
        self.started = False
        return self.T2 - self.T1


def epoch_shuffle_seed(seed: int, epoch: int):
    result = []
    random.seed(seed)
    for _ in range(epoch):
        result.append(random.randint(-10000, 10000))

    return result


def debug_pause() -> None:
    print("=" * 50)
    print("pausing")
    print("=" * 50)
    sys.stdin.read(1)
    return


def plot_sketch(sketch_data, end_on_removed: bool = True) -> None:
    """sketch data is preprocessed, but probably not 0-padded"""
    x_list, y_list = [], []
    x, y = 0.0, 0.0
    for point in sketch_data:
        if point[0] == 0. and point[1] == 0. and point[2] == 0.:
            if len(x_list) != 0:
                raise AssertionError("A stroke is 'half-removed'???")
            if end_on_removed:
                break
            else:
                continue
        x += point[0]
        y += point[1]
        x_list.append(x)
        y_list.append(y)
        if point[2] == 1.0:
            plt.plot(x_list, y_list)
            x_list.clear()
            y_list.clear()

    plt.xlim([-1.0, 1.0])
    plt.ylim([-1.0, 1.0])

    plt.show()
    return

