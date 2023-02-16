def creat_dataset(name):
    data = np.array(pd.read_csv(name))
    data_x1 = data[:, 0:8]#输出第0-5列
    data_y1 = data[:, 8]#输出第6列
    data_y2 = data[:, 9]
    xx, yy, YY = [], [], []
    data_x1max = np.amax(data_x1, axis=0)
    data_x1min = np.amin(data_x1, axis=0)
    data_y1max = np.amax(data_y1, axis=0)
    data_y1min = np.amin(data_y1, axis=0)
    data_y2max = np.amax(data_y2, axis=0)
    data_y2min = np.amin(data_y2, axis=0)
    for i in range(data_x1.shape[1]):#shape[0]为矩阵行数，shape[1]为矩阵列数
        for j in range(data_x1.shape[0]):
            data_x1[j,i] = (data_x1[j, i] - data_x1min[i]) / (data_x1max[i] - data_x1min[i])
    for j in range(data_y1.shape[0]):
        data_y1[j] = (data_y1[j] - data_y1min) / (data_y1max - data_y1min)
        data_y2[j] = (data_y2[j] - data_y2min) / (data_y2max - data_y2min)
    for i in range(data_x1.shape[0]):#shape[0]为矩阵行数，shape[1]为矩阵列数
        xx.append(data_x1[i, :])
        yy.append(data_y1[i])
        YY.append(data_y2[i])
    xx = np.array(xx)
    yy = np.array(yy)
    YY = np.array(YY)
    train_x = np.reshape(xx, (xx.shape[0], 1, xx.shape[1])).astype('float32')
    train_x1 = train_x.astype('float32')
    train_x1 = torch.from_numpy(train_x1).to(device)#转换成张量，对数组进行改变时，原数组也会发生变化
    train_y1 = np.reshape(yy, (yy.shape[0], 1, 1)).astype('float32')
    train_y1 = train_y1.astype('float32')
    train_y1 = torch.from_numpy(train_y1).to(device)
    train_y2 = np.reshape(YY, (YY.shape[0], 1, 1)).astype('float32')
    train_y2 = train_y2.astype('float32')
    train_y2 = torch.from_numpy(train_y2).to(device)
    return train_x1, train_y1, train_y2

All_data, B5_targets, B6_targets = creat_dataset('../dataset_all.csv')