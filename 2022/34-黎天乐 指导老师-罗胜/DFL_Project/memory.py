import numpy as np


class memory:
    matrix = None
    add_time = None

    def __init__(self, batch_number, batch_size):
        self.matrix = np.zeros([batch_number, batch_size])
        self.add_time = int(0)

    def add_number(self, batch_no, true_array):
        self.matrix[batch_no] = self.matrix[batch_no]+true_array

    def add(self):
        self.add_time = self.add_time + 1

    def get_ambiguous(self, number):
        temp_matrix = self.matrix
        temp_matrix = np.abs(temp_matrix - np.ones(self.matrix.shape) * self.add_time * 0.5)
        temp_array = np.reshape(temp_matrix, self.matrix.shape[0] * self.matrix.shape[1])
        temp = list(map(list, zip(range(len(temp_array)), temp_array)))
        small = sorted(temp, key=lambda x: x[1], reverse=False)
        small_array = []
        for i in range(number):
            small_array.append(small[i][0])
        return small_array
