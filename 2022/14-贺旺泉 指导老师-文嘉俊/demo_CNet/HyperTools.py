import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio  

def featureNormalize(X,type):
    #type==1 x = (x-mean)/std(x)
    #type==2 x = (x-max(x))/(max(x)-min(x))
    if type==1:
        mu = np.mean(X,0)
        X_norm = X-mu
        sigma = np.std(X_norm,0)
        X_norm = X_norm/sigma
        return X_norm
    elif type==2:
        minX = np.min(X,0)
        maxX = np.max(X,0)
        X_norm = X-minX
        X_norm = X_norm/(maxX-minX)
        return X_norm    
        
def DrawResult(labels,imageID):
    #ID=1:Pavia University
    #ID=2:Salinas   
    num_class = int(labels.max())
    if imageID == 1:
        row = 610
        col = 340
        # palette = np.array([[216,191,216],
        #                     [0,255,0],
        #                     [0,255,255],
        #                     [45,138,86],
        #                     [255,0,255],
        #                     [255,165,0],
        #                     [159,31,239],
        #                     [255,0,0],
        #                     [255,255,0]])
        palette = np.array([[192,192,192],
                            [0,255,0],
                            [0,255,255],
                            [0,128,0],
                            [255,0,255],
                            [165,82,41],
                            [128,0,128],
                            [255,0,0],
                            [255,255,0]])
        palette = palette*1.0/255
    
    elif imageID ==2:
        row = 512
        col = 217
        # palette = np.array([[37, 58, 150],
        #                     [47, 78, 161],
        #                     [56, 87, 166],
        #                     [56, 116, 186],
        #                     [51, 181, 232],
        #                     [112, 204, 216],
        #                     [119, 201, 168],
        #                     [148, 204, 120],
        #                     [188, 215, 78],
        #                     [238, 234, 63],
        #                     [246, 187, 31],
        #                     [244, 127, 33],
        #                     [239, 71, 34],
        #                     [238, 33, 35],
        #                     [180, 31, 35],
        #                     [123, 18, 20]])
        palette = np.array([[0, 0, 255],
                        [255, 100, 0],
                        [0, 255, 135],
                        [150, 70, 150],
                        [100, 150, 255],
                        [60, 90, 115],
                        [255, 255, 125],
                        [255, 0, 255],
                        [100, 0, 255],
                        [0, 170, 255],
                        [0, 255, 3],
                        [175, 175, 80],
                        [100, 190, 55],
                        [140, 70, 50],
                        [115, 255, 170],
                        [255, 255, 0]])
        palette = palette*1.0/255
    elif imageID ==3:
        row = 145
        col = 145
        # palette = np.array([[37, 58, 150],
        #                     [47, 78, 161],
        #                     [56, 87, 166],
        #                     [56, 116, 186],
        #                     [51, 181, 232],
        #                     [112, 204, 216],
        #                     [119, 201, 168],
        #                     [148, 204, 120],
        #                     [188, 215, 78],
        #                     [238, 234, 63],
        #                     [246, 187, 31],
        #                     [244, 127, 33],
        #                     [239, 71, 34],
        #                     [238, 33, 35],
        #                     [180, 31, 35],
        #                     [123, 18, 20]])
        palette = np.array([[140, 67, 46],
                            [0, 0, 255],
                            [255, 100, 0],
                            [0, 255, 123],
                            [164, 75, 155],
                            [101, 174, 255],
                            [118, 254, 172],
                            [60, 91, 112],
                            [255, 255, 0],
                            [255, 255, 125],
                            [255, 0, 255],
                            [100, 0, 255],
                            [0, 172, 254],
                            [0, 255, 0],
                            [171, 175, 80],
                            [101, 193, 60]])
        palette = palette*1.0/255
    elif imageID ==5:
        row = 456
        col = 352

        palette = np.array([[0,172,255],
                            [255,0,255],
                            [0,255,122],
                            [255,255,122],
                            [255,100,0],
                            [140,67,45]])
        palette = palette*1.0/255
    X_result = np.zeros((labels.shape[0],3))
    for i in range(1,num_class+1):
        X_result[np.where(labels==i),0] = palette[i-1,0]
        X_result[np.where(labels==i),1] = palette[i-1,1]
        X_result[np.where(labels==i),2] = palette[i-1,2]
    
    X_result = np.reshape(X_result,(row,col,3))
    plt.axis ( "off" ) 
    plt.imshow(X_result)    
    return X_result
    
def CalAccuracy(predict,label):
    n = label.shape[0]
    OA = np.sum(predict==label)*1.0/n
    correct_sum = np.zeros((max(label)+1))
    reali = np.zeros((max(label)+1))
    predicti = np.zeros((max(label)+1))
    producerA = np.zeros((max(label)+1))

    for i in range(0,max(label)+1):
        correct_sum[i] = np.sum(label[np.where(predict==i)]==i)
        reali[i] = np.sum(label==i)
        predicti[i] = np.sum(predict==i)
        producerA[i] = correct_sum[i] / reali[i]
   
    Kappa = (n*np.sum(correct_sum) - np.sum(reali * predicti)) *1.0/ (n*n - np.sum(reali * predicti))
    AA = np.average(producerA)
    return OA,AA,Kappa,producerA

def LoadHSI(dataID=1,num_label=60):
    #ID=1:Pavia University
    #ID=2:Salinas
    if dataID==1:        
        data = sio.loadmat('../Data/PaviaU.mat')
        X = data['paviaU']    
        data = sio.loadmat('../Data/PaviaU_gt.mat')
        Y = data['paviaU_gt']
            
    elif dataID==2:        
        data = sio.loadmat('../Data/Salinas_corrected.mat')
        X = data['salinas_corrected']    
        data = sio.loadmat('../Data/Salinas_gt.mat')
        Y = data['salinas_gt']

    elif dataID == 3:
        data = sio.loadmat('../Data/Indian_pines_corrected.mat')
        X = data['indian_pines_corrected']
        data = sio.loadmat('../Data/Indian_pines_gt.mat')
        Y = data['indian_pines_gt']
        num_label = [20,80,80,80,80,80,15,80,10,80,80,80,80,80,50,50]
    elif dataID == 4:
        data = sio.loadmat('../Data/Houston.mat')
        X = data['Houston']
        data = sio.loadmat('../Data/Houston_gt.mat')
        Y = data['Houston_gt']
    elif dataID == 5:
        data = sio.loadmat('../Data/xqh.mat')
        X = data['xqh']
        data = sio.loadmat('../Data/xqh_gt.mat')
        Y = data['xqh_gt']

    ## 对HSI进行裁剪
    X_ori = X
    Y_ori = Y
    # get1
    # X = X[290:390, 90:190, :]
    # Y = Y[290:390, 90:190]
    #4
    # X = X[345:395,125:175, :]
    # Y = Y[345:395,125:175]
    #5
    # X = X[65:115,80:130, :]
    # Y = Y[65:115,80:130]
    #6
    # X = X[130:180,100:150, :]
    # Y = Y[130:180,100:150]
    #7
    # X = X[340:400,130:190, :]
    # Y = Y[340:400,130:190]
    # 2500*2500
    # X = X[228:465,18:255, :]
    # Y = Y[228:465,18:255]

    # X = X[250:400,50:200, :]
    # Y = Y[250:400,50:200]

    [row,col,n_feature] = X.shape
    K = row*col
    X = X.reshape(K, n_feature)       
    
    n_class = Y.max()

    X = featureNormalize(X,2)  
    X = np.reshape(X,(row,col,n_feature))
    X = np.moveaxis(X,-1,0)
    Y = Y.reshape(K,).astype('int')

    for i in range(1,n_class+1):
        
        index = np.where(Y==i)[0]
        n_data = index.shape[0]
        np.random.seed(12345)
        randomArray_label = np.random.permutation(n_data)
        train_num = num_label[i-1]
        # train_num = num_label
        if i==1:
            train_array = index[randomArray_label[0:train_num]]
            test_array = index[randomArray_label[train_num:n_data]]
        else:            
            train_array = np.append(train_array,index[randomArray_label[0:train_num]])
            test_array = np.append(test_array,index[randomArray_label[train_num:n_data]])

    return X,Y,train_array,test_array