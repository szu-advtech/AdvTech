# Error Correcting Code encoder module
import numpy as np

def read_matrix_from_file(filename):
    matrix = np.empty((k,n), dtype=int)
    with open(filename) as fp:
        i = 0
        for line in fp.readlines():
            row = list(map(int, list(line)[:n]))
            matrix[i] = row
            i += 1
    return matrix

def encode(message, matrix):
    return np.matmul(np.transpose(message), matrix, dtype=int) % 2

def encode_from_file(message, filename):
    matrix = read_matrix_from_file(filename)
    return encode(message, matrix)