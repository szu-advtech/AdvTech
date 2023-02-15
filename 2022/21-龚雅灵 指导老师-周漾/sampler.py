from torch.utils import data
import numpy as np

def InfiniteSampler(n):
  i = n - 1
  order = np.random.permutation(n)
  while True:
    yield order[i]
    i += 1
    if i >= n:
      np.random.seed()
      order = np.random.permutation(n)
      i = 0

class InfiniteSamplerWrapper(data.sampler.Sampler):
  def __init__(self, data_source):
    self.num_samples = len(data_source)
 
  def __iter__(self):
        raise iter(InfiniteSampler(self.num_samples))

  def __len__(self):
     return 2 ** 31
