import hashlib
import math
import numpy
import numpy.random
import scipy.integrate as integrate
import scipy.special
import pandas as pd


# The hash functions used for sampling
# Some (but not all) of the randomness used for sampling comes from the output
# of these hash functions, so in order to get different samples across runs,
# a different seed should be used each time (in the function estimate below).

def hashed_float(s):
    """
    Returns a float in the range uniformly in (0, 1] based on the SHA-256 hash
    of the string.
    """
    number = int(hashlib.sha256(s).hexdigest(), 16)
    return (number + 1) / float(2 ** 256)


def exp_hashed(s):
    """
    A hash function mapping strings into a float drawn from Exp(1).
    """
    return -1.0 * math.log(hashed_float(s))


# Bottom-k Sketch Structure (Algorithm 1)
class BottomK:
    # Initialize structure
    def __init__(self, k):
        # the structure size k
        self.k = k
        # Set of <= k key-value pairs
        self.samples = {}

    # Processes element(e.key, e.val)
    def process(self, key, val):
        if key in self.samples.keys():
            self.samples[key] = min(self.samples[key], val)
        else:
            self.samples[key] = val

        # If the sketch size is greater than k, remove the largest elements
        if len(self.samples.keys()) > self.k:
            sorted_items = sorted(self.samples.items(), key=lambda x: x[1])
            for i in range(self.k, len(sorted_items)):
                del self.samples[sorted_items[i][0]]
            return sorted_items[self.k]
        return None

    # Merges two bottom-k structures (of the same size)
    def merge(self, another):
        if another.k != self.k:
            raise Exception("merging bottom-k of different size")
        for key, value in another.samples.items():
            BottomK.process(self, key, value)


# PPSWOR Sampling Sketch (Algorithm 2)
class Ppswor(BottomK):
    # Processes element
    def process(self, key, value):
        v = numpy.random.exponential(1.0 / value)
        #  Process the element(e.key, v) into the bottom-k structure
        BottomK.process(self, key, v)


# SumMax sampling sketch (Algorithm 3).
class SumMax(BottomK):
    # The string used to separate the primary and secondary key
    KEY_SEP = "!@#$"

    # Initializes an empty sketch of a given size.
    def __init__(self, k, hash_seed):
        # The sketch size
        self.k = k
        # The hash seed
        self.hash_seed = str(hash_seed)
        # Set of <= k key-value pairs
        self.samples = {}

    # Processes element (key, value).
    def process(self, key, value):
        if str(key[0]) == "" or SumMax.KEY_SEP in str(key[0]):
            raise Exception("Primary key is empty or contains forbidden string")
        val = exp_hashed((self.hash_seed + str(key[0]) + SumMax.KEY_SEP + str(key[1])).encode("utf-8")) / value
        BottomK.process(self, key[0], val)

    #  Merges another SumMax sketch (of the same size) into this sketch.
    def merge(self, another):
        if another.k != self.k:
            raise Exception("merging SumMax of different size")
        if another.hash_seed != self.hash_seed:
            raise Exception("merging SumMax with different hash functions")
        for key, val in another.samples.items():
            BottomK.process(self, key, val)


# The main sampling sketch (Algorithms 4 and 5)
class MainSketch:
    def __init__(self, k, eps, funcA, funcB, hash_seed):
        self.k = k
        self.eps = float(eps)
        self.funcA = funcA
        self.funcB = funcB
        # SumMax sketch of size k
        self.summax = SumMax(k, hash_seed)
        # PPSWOR sketch of size k
        self.ppswor = Ppswor(k)
        # A sum of all the elements seen so far
        self.sum = 0
        # Threshold (between the PPSWOR and SumMax samples)
        self.gamma = float("inf")
        # The Sideline structure (elements not processed yet by the SumMax)
        # A composable max-heap/priority queue
        self.sideline = {}
        # The number of output elements per input element
        self.r = int(math.ceil(self.k / self.eps))

        # Statistics on the size of the sketch
        # The maximum number of elements ever stored in Sideline
        self.max_sideline_size = 0
        # The maximum number of distinct input keys ever stored in Sideline
        self.max_distinct_sideline_size = 0
        # The maximum number of elements ever stored anywhere in the sketch
        # (in either the SumMax, PPSWOR, or Sideline structures)
        self.max_total_size = 0
        # The maximum number of distinct input keys ever stored in sketch
        self.max_distinct_total_size = 0

    # Processes element (key, val) into the sketch.
    def process(self, key, val):
        self.ppswor.process(key, val)
        self.sum += val
        self.gamma = (2.0 * self.eps) / self.sum
        for i in range(self.r):
            y = numpy.random.exponential(1.0 / val)
            if (key, i) in self.sideline.keys():
                self.sideline[(key, i)] = min(y, self.sideline[(key, i)])
            else:
                self.sideline[(key, i)] = y
        for sideline_key, sideline_val in list(self.sideline.items()):
            if sideline_val >= self.gamma:
                del self.sideline[sideline_key]
                integral_res = self.funcA(sideline_val)
                if integral_res > 0:
                    self.summax.process(sideline_key, integral_res)

        # Save Sideline size statistics
        self.max_sideline_size = max(self.max_sideline_size, len(self.sideline))
        current_sideline_distinct = len(set([x[0] for x in self.sideline.keys()]))
        self.max_distinct_sideline_size = max(self.max_distinct_sideline_size, current_sideline_distinct)

        # Update total size statistics
        current_max_total_size = (len(self.ppswor.samples) + len(self.summax.samples) + len(self.sideline))
        self.max_total_size = max(self.max_total_size, current_max_total_size)
        current_total_distinct = len(set(list(self.ppswor.samples.keys())
                                         + list(self.summax.samples.keys())
                                         + [x[0] for x in self.sideline.keys()]))
        self.max_distinct_total_size = max(self.max_distinct_total_size, current_total_distinct)

    def output_sample(self):
        new_summax = SumMax(self.k, self.summax.hash_seed)
        new_summax.merge(self.summax)
        integral_res = self.funcA(self.gamma)
        if integral_res > 0:
            for key, value in self.sideline.items():
                new_summax.process(key, integral_res)
        # scale sample by gamma
        for x in new_summax.samples.keys():
            new_summax.samples[x] *= self.r
        new_ppswor = BottomK(self.k)
        integral_res = self.funcB(self.gamma)
        if integral_res > 0:
            for key, val in self.ppswor.samples.items():
                new_ppswor.process(key, val / integral_res)
        new_ppswor.merge(new_summax)
        return new_ppswor.samples, self.gamma


def seedCDF(w, t, gamma, r, funcA, funcB):
    """
    Computes the conditional inclusion probability for a key, as in Section 5.4 in the full version.
    In particular, we compute the probability that the seed of a key with frequency w is below a threshold t.
    Parameters:
    w: The frequency of the key.
    t: The threshold (we return the probability of the seed being below t).
    gamma: The parameter gamma from the sampling sketch.
    r: The number of output elements per input element (should be ceil(k/eps)).
    funcA: The function A for the soft concave sublinear function.
            Depends on the inverse complement Laplace transform of the function.
    funcB: The function B for the soft concave sublinear function.
            Depends on the inverse complement Laplace transform of the function.
    """
    p1 = numpy.exp(-1.0 * w * t * funcB(gamma))
    p2_low = (numpy.exp(-1.0 * funcA(gamma) * t / r) * (1.0 - numpy.exp(-1.0 * w * gamma)))
    func_to_integr = lambda x: w * numpy.exp(-1.0 * w * x - (t * funcA(x) / r))
    p2_high = integrate.quad(func_to_integr, gamma, numpy.inf)
    p2 = p2_low + p2_high[0]
    return 1.0 - p1 * (p2 ** r)


# The function A for the soft concave sublinear function sqrt(x)
def sqrt_funcA(tau):
    return (tau * math.pi) ** (-0.5)


# The function B for the soft concave sublinear function sqrt(x)
def sqrt_funcB(tau):
    return (tau / math.pi) ** 0.5


# The function sqrt(x)
def sqrt_func(x):
    return math.sqrt(x)


# The function A for the soft concave sublinear function ln(1 + x)
def ln_funcA(tau):
    return scipy.special.exp1(tau)


# The function A for the soft concave sublinear function ln(1 + x)
def ln_funcB(tau):
    return 1.0 - (math.e ** (-1.0 * tau))


# The function ln(1 + x)
def ln_func(x):
    return math.log(1.0 + x, math.e)


def estimate(elements, k, func, funcA, funcB, hash_seed="", eps=0.5):
    """
    Produces a sample from a list of data elements and uses it to estimate
    the f-statistics (see Section 2 of the full version) of the dataset.
    These estimates were computed in the experiments described in Section 6 (of the full version).
    """
    # First pass over the data (produces the sample).
    sk = MainSketch(k, eps, funcA, funcB, hash_seed)
    for key, value in elements:
        sk.process(key, value)
    output_sample, gamma = sk.output_sample()

    # Determines the inclusion threshold (the k-th lowest seed).
    t = max(output_sample.values())
    # A list of the sampled keys (at most k - 1 keys with lowest seed).
    k_minus_one = [x for x in output_sample.keys() if output_sample[x] < t]
    # Sanity check: there are k - 1 elements with seed below the inclusion threshold
    if len(k_minus_one) != len(output_sample) - 1:
        raise Exception("WARNING: k-1 size is less than k-1")

    # Second pass over the data (gets the frequencies of the sampled keys).
    counts = {}
    for key, value in elements:
        if key in k_minus_one:
            if key not in counts:
                counts[key] = 0
            counts[key] += value

    # Computes the inverse probability estimator.
    s = sum([func(counts[key]) / seedCDF(counts[key], t, gamma, sk.r, funcA, funcB) for key in k_minus_one])
    return s, output_sample


if __name__ == '__main__':
    data = pd.read_csv('abcnews-date-text.csv', usecols=['publish_date', 'headline_text'], nrows=1000)
    elements = []
    for i in range(len(data)):
        string = data['headline_text'][i]
        array = string.split(" ")
        for j in range(len(array)):
            t = (array[j], 1)
            elements.append(t)
    s, output_sample = estimate(elements, 25, ln_func, ln_funcA, ln_funcB, hash_seed="", eps=0.5)
    print('the output_sample is:')
    for key, value in output_sample.items():
        print(key, value)
    print()
    print('the f-statistics of the dataset is:')
    print(s)
