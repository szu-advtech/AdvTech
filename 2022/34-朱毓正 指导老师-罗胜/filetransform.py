# sourcery skip: avoid-builtin-shadow
import os
import numpy as np
import scipy.io as sio

if __name__ == "__main__":
    path = r"E:\BPD-main\data\PAMAP2_Dataset\Processed\subject101.dat"
    
    save_path = "E:/BPD-main/data/PAMAP2_Dataset/Processed/subject11.mat"
    with open(path) as f:
      all=f.read()
      all=all.split('\n')

      datas = []
      labels = []
      i = 0
      for row in all:
        if row != '':
          data = [float(i) for i in row.split(' ')[:-1]]
          label = int(row.split(' ')[-1])
          datas.append(data)
          labels.append(label)
      datas = np.array(datas)
      labels = np.array(labels)
      sio.savemat(save_path,{'data':datas,'label':labels})
