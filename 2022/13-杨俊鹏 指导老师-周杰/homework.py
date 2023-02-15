import os
import torch
import time
import numpy as np


class MatrixMultiplication(object):

    # 直接矩阵乘法
    def direct_multiply(self, matrix_a, matrix_b):
        n = len(matrix_a)
        C = [[0 for col in range(n)] for row in range(n)]
        for i in range(0, n):
            for j in range(0, n):
                for k in range(0, n):
                    C[i][j] = C[i][j] + matrix_a[i][k] * matrix_b[k][j]
        return C

    # 普通分治矩阵乘法
    def square_matrix_multiply(self, matrix_a, matrix_b):
        n = len(matrix_a)
        C = [[0 for col in range(n)] for row in range(n)]
        if n == 1:
            C[0][0] = matrix_a[0][0] * matrix_b[0][0]
        else:
            (A11, A12, A21, A22) = self.matrix_divide(matrix_a)
            (B11, B12, B21, B22) = self.matrix_divide(matrix_b)
            (C11, C12, C21, C22) = self.matrix_divide(C)
            C11 = self.matrix_add(self.square_matrix_multiply(A11, B11), self.square_matrix_multiply(A12, B21))
            C12 = self.matrix_add(self.square_matrix_multiply(A11, B12), self.square_matrix_multiply(A12, B22))
            C21 = self.matrix_add(self.square_matrix_multiply(A21, B11), self.square_matrix_multiply(A22, B21))
            C22 = self.matrix_add(self.square_matrix_multiply(A21, B12), self.square_matrix_multiply(A22, B22))
            C = self.matrix_merge(C11, C12, C21, C22)
        return C

    # strassen分治法
    def strassen(self, A, B):
        n = len(A)
        C = [[0 for i in range(n)] for j in range(n)]
        if n == 1:
            C[0][0] = A[0][0] * B[0][0]
            return C
        A11, A12, A21, A22 = self.matrix_divide(A)
        B11, B12, B21, B22 = self.matrix_divide(B)
        M1 = self.strassen(A11, self.matrix_sub(B12, B22))
        M2 = self.strassen(self.matrix_add(A11, A12), B22)
        M3 = self.strassen(self.matrix_add(A21, A22), B11)
        M4 = self.strassen(A22, self.matrix_sub(B21, B11))
        M5 = self.strassen(self.matrix_add(A11, A22), self.matrix_add(B11, B22))
        M6 = self.strassen(self.matrix_sub(A12, A22), self.matrix_add(B21, B22))
        M7 = self.strassen(self.matrix_sub(A11, A21), self.matrix_add(B11, B12))
        C11 = self.matrix_add(self.matrix_sub(self.matrix_add(M5, M4), M2), M6)
        C12 = self.matrix_add(M1, M2)
        C21 = self.matrix_add(M3, M4)
        C22 = self.matrix_sub(self.matrix_sub(self.matrix_add(M5, M1), M3), M7)
        C = self.matrix_merge(C11, C12, C21, C22)
        return C

    def matrix_add(self, matrix_a, matrix_b):
        rows = len(matrix_a)
        columns = len(matrix_a[0])
        matrix_sum = np.zeros((rows, columns), dtype="i").tolist()
        for i in range(rows):
            for j in range(columns):
                matrix_sum[i][j] = matrix_a[i][j] + matrix_b[i][j]
        return matrix_sum

    def matrix_sub(self, matrix_a, matrix_b):
        rows = len(matrix_a)
        columns = len(matrix_a[0])
        matrix_difference = np.zeros((rows, columns), dtype="i").tolist()
        # matrix_difference = [[]*columns]*rows
        for i in range(rows):
            for j in range(columns):
                matrix_difference[i][j] = matrix_a[i][j] - matrix_b[i][j]
        return matrix_difference

    def matrix_divide(self, matrix):
        # 返回四个子矩阵：分别为矩阵的左上，右上，左下，右下
        rows = len(matrix)
        columns = len(matrix[0])
        x_middle = rows // 2
        y_middle = columns // 2
        matrix_11 = [M[:x_middle] for M in matrix[:y_middle]]
        matrix_12 = [M[x_middle:] for M in matrix[:y_middle]]
        matrix_21 = [M[:x_middle] for M in matrix[y_middle:]]
        matrix_22 = [M[x_middle:] for M in matrix[y_middle:]]
        return matrix_11, matrix_12, matrix_21, matrix_22

    def matrix_merge(self, matrix_11, matrix_12, matrix_21, matrix_22):
        matrix_total = []
        rows1 = len(matrix_11)
        rows2 = len(matrix_21)
        for i in range(rows1):
            matrix_total.append(matrix_11[i] + matrix_12[i])
        for j in range(rows2):
            matrix_total.append(matrix_21[j] + matrix_22[j])
        return matrix_total


if __name__ == '__main__':
    matrix_mul = MatrixMultiplication()
    # n = np.power(10, 1)
    n = 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.set_device(3)
    a = torch.randint(1, 3, device="cuda", size=(n,n), dtype=torch.int8)
    b = torch.randint(1, 3, device="cuda", size=(n,n), dtype=torch.int8)

    # a = np.random.randint(1, 3, size=(n, n), dtype=np.int64)
    # b = np.random.randint(1, 3, size=(n, n), dtype=np.int64)
    start = time.time()
    # ans=torch.mm(a,b)
    # 普通矩阵乘法
    # for i in range(100):
    # ans = np.array(matrix_mul.direct_multiply(a, b))
    # 普通分治的矩阵乘法
    # for i in range(100):
    # ans = np.array(matrix_mul.square_matrix_multiply(a, b))
    # strassen分治的矩阵乘法
    # for i in range(100):
    ans = np.array(matrix_mul.strassen(a, b))

    end = (time.time() - start)
    print("a矩阵:\n", a)
    print("b矩阵:\n", b)
    print("结果:\n", ans)
    print("耗时%s秒" % (time.time() - start))

    # for i in range(1, 7):
    #     n = np.power(10, i)
    #     a = np.random.randint(1, 10, size=(n, n), dtype=np.int8)
    #     b = np.random.randint(1, 10, size=(n, n), dtype=np.int8)
    #     print("n=%d时" % n)
    #     print("a矩阵:\n", a)
    #     print("b矩阵:\n", b)