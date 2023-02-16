
import numpy as np
import datetime as dt

from numpy import genfromtxt

class SimpleSaver:

    def __init__(self):
        pass

    def save(self, nparray, name):
        np.savetxt('./npsaver/' + name + '.csv', nparray, delimiter=",")

    def load(self, name):
        nparray = genfromtxt('./npsaver/' + name + '.csv', delimiter=',')
        print(nparray, 'nparray.shape: ', nparray.shape)

    @staticmethod
    def test():
        s = SimpleSaver()

        a = np.arange(1, 100)
        a = np.reshape(a, (33, 3))
        s.save(a, 'atest')

        s.load('atest')



if __name__ == '__main__':

    SimpleSaver.test()

