import csv
import sys
import os
import numpy as np

def generate_data(_type, _n):
    if os.path.exists('dataset') == False:
        print('path do not exist, create new directory!')
        os.mkdir('dataset')
    csv_path = 'dataset/' + _type + '.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        lst = []
        if _type == 'linear':
            for i in range(1, _n + 1):
                lst.append([i])
        elif _type == 'seg1':
            k = 1
            b = 2000
            for i in range(1, _n + 1):
                if i % (_n / 100) == 0:
                    k += 1
                    b += 500
                lst.append([k * i + b])
        elif _type == 'seg10':
            k = 1
            b = 2000
            for i in range(1, _n + 1):
                if i % (_n / 10) == 0:
                    k += 1
                    b += 500
                lst.append([k * i + b])
        elif _type == 'normal':
            mu ,sigma = 10, 1
            sampleNo = _n
            np.random.seed(0)
            s = np.random.normal(mu, sigma, sampleNo)
            s = np.sort(s)
            for i in s:
                lst.append([int(i * _n)])
        writer.writerows(lst)

# python3 datagen.py seg1 1000000
if __name__ == '__main__':
    _type = sys.argv[1]
    _n = int(sys.argv[2])
    generate_data(_type, _n)
    print("ok")
