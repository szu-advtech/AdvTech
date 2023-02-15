import pandas as pd
import numpy as np
from sklearn import datasets


dir = "E:/study/master1-1/论文里面需要的数据集/"

# 将数据装箱,data是dataframe的list
def bin_to_3(data):
    df = pd.DataFrame(data)
    df.columns = data.columns
    for colName,value in data.iteritems():
        if type(value[0]) != str:
            # 这里不能用value = pd.qcut(value,3,labels=["low", "medium", "high"])

            # data[colName] = pd.qcut(value, 3,duplicates="drop")
            df[colName] = pd.qcut(data[colName].rank(method="first"),3,labels=["low", "medium", "high"],duplicates="drop")
    return df

def bin_to_5(data):
    df = pd.DataFrame(data)
    df.columns = data.columns
    for colName,value in data.iteritems():
        if type(value[0]) != str:
            # data[colName] = pd.qcut(value, 5, duplicates="drop")
            df[colName] = pd.qcut(value.rank(method="first"),5,labels=["lower", "low","medium", "high","higher"],duplicates="drop")
    return df

def bin_to_10(data):
    df = pd.DataFrame(data)
    df.columns = data.columns
    for colName,value in data.iteritems():
        if type(value[0]) != str:
            # data[colName] = pd.qcut(value, 10,duplicates="drop")
            df[colName] = pd.qcut(value.rank(method="first"),10,labels=["1st", "2nd","3rd", "4th","5th","6th","7th", "8th","9th", "10th"],duplicates="drop")
    return df

def ToCSV(data, label, fileName):
    # label = 5
    rowIndex = np.array(list(data.columns))# 特征和标签
    # 将target和feature set分开
    X = data.drop(label, axis=1)
    y = data[label]

    # 将datafrema转换成ndarray
    # XrowIndex = rowIndex.remove(label)# 保存行索引,全是特征，没有标签
    X = X.values # 转换
    y = y.values

    y = y.reshape(-1,1)
    data = np.hstack((X, y))
    rowIndex = rowIndex.reshape(1,-1)
    data = np.insert(data, 0, rowIndex , axis=0)
    data_dir = 'E:/study/master1-1/CSVdataset/'
    np.savetxt(data_dir + fileName + ".csv", data, delimiter = ',',fmt = '%s')

def Load_Abalone():
    col = 9
    data = pd.read_csv('E:/study/master1-1/论文里面需要的数据集/Abalone/abalone.data', sep=',',
                       names=[i for i in range(col)], lineterminator="\n")
    # data.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight',
      #                 'Shell weight', 'Rings']
    data.columns = [i for i in range(col)]
    data = np.array(data)
    data = pd.DataFrame(data)
    sortName = col-1
    data = data.sort_values(by=sortName)
    data_3 = bin_to_3(data)
    ToCSV(data_3, col - 1, "Abalone-3")
    data_5 = bin_to_5(data)
    ToCSV(data_5, col - 1, "Abalone-5")
    data_10 = bin_to_10(data)
    ToCSV(data_10, col - 1, "Abalone-10")

    return data

def Load_Boston():
    # label是'MEDV'
    col = 14
    boston_tmp = datasets.load_boston()
    data = pd.DataFrame(data=boston_tmp.data, columns=boston_tmp.feature_names)
    data["MEDV"] = boston_tmp.target
    data.columns = [i for i in range(col)]
    sortName = col - 1
    data = data.sort_values(by=sortName)
    data_3 = bin_to_3(data)
    ToCSV(data_3, col - 1, "BostonHousing-3")
    data_5 = bin_to_5(data)
    ToCSV(data_5, col - 1, "BostonHousing-5")
    data_10 = bin_to_10(data)
    ToCSV(data_10, col - 1, "BostonHousing-10")


    return data

def Load_Delta_ailerons(): # ok
    col = 6
    data = pd.read_csv(dir + 'delta_ailerons/delta_ailerons.data', sep=' ', names=[i for i in range(col)])
    data = np.array(data)
    data = pd.DataFrame(data)
    data.columns = [i for i in range(col)]
    sortName = col - 1
    data = data.sort_values(by=sortName)
    data_3 = bin_to_3(data)
    ToCSV(data_3, col-1, "DeltaAilerons-3")
    data_5 = bin_to_5(data)
    ToCSV(data_5, col - 1, "DeltaAilerons-5")
    data_10 = bin_to_10(data)
    ToCSV(data_10, col - 1, "DeltaAilerons-10")


def Load_2D_Planes():
    col = 11
    data = pd.read_csv(dir + '2DPlanes/cart_delve.data', sep='  ', names=[i for i in range(col)],lineterminator="\n")
    data = np.array(data)
    data = pd.DataFrame(data)
    data.columns = [i for i in range(col)]
    sortName = col - 1
    data = data.sort_values(by=sortName)
    data_3 = bin_to_3(data)
    ToCSV(data_3, col - 1, "2DPlanes-3")
    data_5 = bin_to_5(data)
    ToCSV(data_5, col - 1, "2DPlanes-5")
    data_10 = bin_to_10(data)
    ToCSV(data_10, col - 1, "2DPlanes-10")


def Load_Elevators():
    col = 19
    data1 = pd.read_csv(dir + 'Elevators/elevators.data', sep=', ',names=[i for i in range(col)], lineterminator="\n")
    data1 = np.array(data1)
    data2 = pd.read_csv(dir + 'Elevators/elevators.test', sep=', ',names=[i for i in range(col)], lineterminator="\n")
    data2 = np.array(data2)
    data = np.insert(data2, 0, data1, axis=0)
    data = pd.DataFrame(data)
    data.columns = [i for i in range(col)]
    sortName = col - 1
    data = data.sort_values(by=sortName)
    data_3 = bin_to_3(data)
    ToCSV(data_3, col - 1, "Elevators-3")
    data_5 = bin_to_5(data)
    ToCSV(data_5, col - 1, "Elevators-5")
    data_10 = bin_to_10(data)
    ToCSV(data_10, col - 1, "Elevators-10")


def Load_California_Housing():
    col = 9
    data = pd.read_csv(dir + 'CaliforniaHousing/cal_housing.data', sep=',',names=[i for i in range(col)])
    data = np.array(data)
    data = pd.DataFrame(data)
    data.columns = [i for i in range(col)]
    sortName = col - 1
    data = data.sort_values(by=sortName)
    data_3 = bin_to_3(data)
    ToCSV(data_3, col - 1, "CaliforniaHousing-3")
    data_5 = bin_to_5(data)
    ToCSV(data_5, col - 1, "CaliforniaHousing-5")
    data_10 = bin_to_10(data)
    ToCSV(data_10, col - 1, "CaliforniaHousing-10")

def Load_delta_elevators():
    col = 7
    data = pd.read_csv('C:/study/master1-1/前沿技术/论文里面需要的数据集/delta_elevators/delta_elevators.data', sep=' ', names=[i for i in range(col)],lineterminator="\n")
    data = np.array(data)
    data = pd.DataFrame(data)
    data.columns = [i for i in range(col)]
    sortName = col - 1
    data = data.sort_values(by=sortName)
    data_3 = bin_to_3(data)
    ToCSV(data_3, col - 1, "deltaElevators-3")
    data_5 = bin_to_5(data)
    ToCSV(data_5, col - 1, "deltaElevators-5")
    data_10 = bin_to_10(data)
    ToCSV(data_10, col - 1, "deltaElevators-10")


def Load_Ailerons():
    col = 41
    data1 = pd.read_csv(dir + 'Ailerons/ailerons.data', sep=', ',names=[i for i in range(col)], lineterminator="\n")
    data1 = np.array(data1)
    data2 = pd.read_csv(dir + 'Ailerons/ailerons.test', sep=', ',names=[i for i in range(col)], lineterminator="\n")
    data2 = np.array(data2)
    data = np.insert(data2, 0, data1, axis=0)
    data = pd.DataFrame(data)
    data.columns = [i for i in range(col)]
    sortName = col - 1
    data = data.sort_values(by=sortName)
    data_3 = bin_to_3(data)
    ToCSV(data_3, col - 1, "Ailerons-3")
    data_5 = bin_to_5(data)
    ToCSV(data_5, col - 1, "Ailerons-5")
    data_10 = bin_to_10(data)
    ToCSV(data_10, col - 1, "Ailerons-10")


def Load_PoleTelecomm():
    col = 49
    data1 = pd.read_csv(dir + 'PoleTelecomm/pol.data', sep=',',names=[i for i in range(col)], lineterminator="\n")
    data1 = np.array(data1)
    data2 = pd.read_csv(dir + 'PoleTelecomm/pol.test', sep=',',names=[i for i in range(col)], lineterminator="\n")
    data2 = np.array(data2)
    data = np.insert(data2, 0, data1, axis=0)
    data = pd.DataFrame(data)
    data.columns = [i for i in range(col)]
    sortName = col - 1
    data = data.sort_values(by=sortName)
    data_3 = bin_to_3(data)
    ToCSV(data_3, col - 1, "PoleTelecomm-3")
    data_5 = bin_to_5(data)
    ToCSV(data_5, col - 1, "PoleTelecomm-5")
    data_10 = bin_to_10(data)
    ToCSV(data_10, col - 1, "PoleTelecomm-10")


def Load_FriedmanArtificial():
    col = 11
    data = pd.read_csv(dir + 'FriedmanArtificial/fried_delve.data', sep=' ',names=[i for i in range(col)])
    data = np.array(data)
    data = pd.DataFrame(data)
    data.columns = [i for i in range(col)]
    sortName = col - 1
    data = data.sort_values(by=sortName)
    data_3 = bin_to_3(data)
    ToCSV(data_3, col - 1, "FriedmanArtificial-3")
    data_5 = bin_to_5(data)
    ToCSV(data_5, col - 1, "FriedmanArtificial-5")
    data_10 = bin_to_10(data)
    ToCSV(data_10, col - 1, "FriedmanArtificial-10")


def Load_Kinematics_of_Robot_Arm():
    col = 9
    data = pd.read_csv(dir + 'Kinematics/kin8nm.data', sep=', ',names=[i for i in range(col)])
    data = np.array(data)
    data = pd.DataFrame(data)
    data.columns = [i for i in range(col)]
    sortName = col - 1
    data = data.sort_values(by=sortName)
    data_3 = bin_to_3(data)
    ToCSV(data_3, col - 1, "Kinematics-3")
    data_5 = bin_to_5(data)
    ToCSV(data_5, col - 1, "Kinematics-5")
    data_10 = bin_to_10(data)
    ToCSV(data_10, col - 1, "Kinematics-10")



def Load_ComputerActivity1():
    col = 13
    data = pd.read_csv(dir + 'ComputerActivity/cpu_small.data', sep=', ',names=[i for i in range(col)])
    data = np.array(data)
    data = pd.DataFrame(data)
    data.columns = [i for i in range(col)]
    sortName = col - 1
    data = data.sort_values(by=sortName)
    data_3 = bin_to_3(data)
    ToCSV(data_3, col - 1, "ComputerActivity1-3")
    data_5 = bin_to_5(data)
    ToCSV(data_5, col - 1, "ComputerActivity1-5")
    data_10 = bin_to_10(data)
    ToCSV(data_10, col - 1, "ComputerActivity1-10")




def Load_ComputerActivity2():
    col = 22
    data = pd.read_csv(dir + 'ComputerActivity/cpu_act.data', sep=', ',names=[i for i in range(col)])
    data = np.array(data)
    data = pd.DataFrame(data)
    data.columns = [i for i in range(col)]
    sortName = col - 1
    data = data.sort_values(by=sortName)
    sortName = col - 1
    data = data.sort_values(by=sortName)
    data_3 = bin_to_3(data)
    ToCSV(data_3, col - 1, "ComputerActivity2-3")
    data_5 = bin_to_5(data)
    ToCSV(data_5, col - 1, "ComputerActivity2-5")
    data_10 = bin_to_10(data)
    ToCSV(data_10, col - 1, "ComputerActivity2-10")


def Load_Auto_MPG():
    col = 8
    data = pd.read_csv(dir + 'Auto-Mpg/auto.data', sep=',',names=[i for i in range(col)])
    data = np.array(data)
    data = pd.DataFrame(data)
    data.columns = [i for i in range(col)]
    sortName = col - 1
    data = data.sort_values(by=sortName)
    data_3 = bin_to_3(data)
    ToCSV(data_3, col - 1, "AutoMpg-3")
    data_5 = bin_to_5(data)
    ToCSV(data_5, col - 1, "AutoMpg-5")
    data_10 = bin_to_10(data)
    ToCSV(data_10, col - 1, "AutoMpg-10")

def Load_Auto_Price():
    col = 16
    data = pd.read_csv(dir + 'Auto-Price/price.data', sep=',',names=[i for i in range(col)])
    data = np.array(data)
    data = pd.DataFrame(data)
    data.columns = [i for i in range(col)]
    sortName = col - 1
    data = data.sort_values(by=sortName)
    data_3 = bin_to_3(data)
    ToCSV(data_3, col - 1, "AutoPrice-3")
    data_5 = bin_to_5(data)
    ToCSV(data_5, col - 1, "AutoPrice-5")
    data_10 = bin_to_10(data)
    ToCSV(data_10, col - 1, "AutoPrice-10")

def Load_Diabetes():
    col = 3
    data = pd.read_csv(dir + 'Diabetes/diabetes.data', sep=',',names=[i for i in range(col)])
    data = np.array(data)
    data = pd.DataFrame(data)
    data.columns = [i for i in range(col)]
    sortName = col - 1
    data = data.sort_values(by=sortName)
    data_3 = bin_to_3(data)
    ToCSV(data_3, col - 1, "Diabetes-3")
    data_5 = bin_to_5(data)
    ToCSV(data_5, col - 1, "Diabetes-5")
    data_10 = bin_to_10(data)
    ToCSV(data_10, col - 1, "Diabetes-10")


def Load_pyrimidines():
    col = 28
    data = pd.read_csv(dir + 'pyrimidines/pyrim.data', sep=', ',names=[i for i in range(col)])
    data = np.array(data)
    data = pd.DataFrame(data)
    data.columns = [i for i in range(col)]
    sortName = col - 1
    data = data.sort_values(by=sortName)
    data_3 = bin_to_3(data)
    ToCSV(data_3, col - 1, "pyrimidines-3")
    data_5 = bin_to_5(data)
    ToCSV(data_5, col - 1, "pyrimidines-5")
    data_10 = bin_to_10(data)
    ToCSV(data_10, col - 1, "pyrimidines-10")

def Load_Triazines():
    col = 61
    data = pd.read_csv(dir + 'Triazines/triazines.data', sep=', ',names=[i for i in range(col)])
    data = np.array(data)
    data = pd.DataFrame(data)
    data.columns = [i for i in range(col)]
    sortName = col - 1
    data = data.sort_values(by=sortName)
    data_3 = bin_to_3(data)
    ToCSV(data_3, col - 1, "Triazines-3")
    data_5 = bin_to_5(data)
    ToCSV(data_5, col - 1, "Triazines-5")
    data_10 = bin_to_10(data)
    ToCSV(data_10, col - 1, "Triazines-10")


def Load_Machine_Cpu():
    col = 7
    data = pd.read_csv(dir + 'Machine-Cpu/machine.data', sep=',',names=[i for i in range(col)])
    data = np.array(data)
    data = pd.DataFrame(data)
    data.columns = [i for i in range(col)]
    sortName = col - 1
    data = data.sort_values(by=sortName)
    data_3 = bin_to_3(data)
    ToCSV(data_3, col - 1, "Machine_Cpu-3")
    data_5 = bin_to_5(data)
    ToCSV(data_5, col - 1, "Machine_Cpu-5")
    data_10 = bin_to_10(data)
    ToCSV(data_10, col - 1, "Machine_Cpu-10")

def Load_Servo():
    col = 5
    data = pd.read_csv(dir + 'Servo/servo.data', sep=',', names=[i for i in range(col)])
    data = np.array(data)
    data = pd.DataFrame(data)
    data.columns = [i for i in range(col)]
    sortName = col - 1
    data = data.sort_values(by=sortName)
    data_3 = bin_to_3(data)
    ToCSV(data_3, col - 1, "Servo-3")
    data_5 = bin_to_5(data)
    ToCSV(data_5, col - 1, "Servo-5")
    data_10 = bin_to_10(data)
    ToCSV(data_10, col - 1, "Servo-10")


def Load_WiscoinBreastCancer():
    col = 33
    data = pd.read_csv(dir + 'WiscoinBreastCancer/r_wpbc.data', sep=',', names=[i for i in range(col)])
    data = np.array(data)
    data = pd.DataFrame(data)
    data.columns = [i for i in range(col)]
    sortName = col - 1
    data = data.sort_values(by=sortName)
    data_3 = bin_to_3(data)
    ToCSV(data_3, col - 1, "WiscoinBreastCancer-3")
    data_5 = bin_to_5(data)
    ToCSV(data_5, col - 1, "WiscoinBreastCancer-5")
    data_10 = bin_to_10(data)
    ToCSV(data_10, col - 1, "WiscoinBreastCancer-10")

def Load_Pumadyn1():
    col = 9
    data1 =[]
    data2 = []
    for line in open(dir + 'Pumadyn1/puma8NH.data', "r"):
        a,b,c,d,e,f,g,h,i, j= line.split(' ')
        [a, b, c, d, e, f, g, h, i] = list(map(float, [a, b, c, d, e, f, g, h, i]))
        data1.append([a,b,c,d,e,f,g,h,i])

    for line in open(dir + 'Pumadyn1/puma8NH.test',"r"):
        a,b,c,d,e,f,g,h,i, j= line.split(' ')
        [a, b, c, d, e, f, g, h, i] = list(map(float, [a, b, c, d, e, f, g, h, i]))
        data2.append([a,b,c,d,e,f,g,h,i])

    data1 = np.array(data1)
    data2 = np.array(data2)
    data = np.insert(data2, 0, data1, axis=0)
    data = pd.DataFrame(data)
    data.columns = [i for i in range(col)]
    sortName = col - 1
    data = data.sort_values(by=sortName)
    data_3 = bin_to_3(data)
    ToCSV(data_3, col - 1, "Pumadyn1-3")
    data_5 = bin_to_5(data)
    ToCSV(data_5, col - 1, "Pumadyn1-5")
    data_10 = bin_to_10(data)
    ToCSV(data_10, col - 1, "Pumadyn1-10")

def Load_Pumadyn2():
    col = 33
    data1 = pd.read_csv(dir + 'pumadyn-32nm/puma32H.data', sep=', ', names=[i for i in range(col)], lineterminator="\n")
    data1 = np.array(data1)
    data2 = pd.read_csv(dir + 'pumadyn-32nm/puma32H.test', sep=', ', names=[i for i in range(col)], lineterminator="\n")
    data2 = np.array(data2)
    data = np.insert(data2, 0, data1, axis=0)
    data = pd.DataFrame(data)
    data.columns = [i for i in range(col)]
    sortName = col - 1
    data = data.sort_values(by=sortName)
    data_3 = bin_to_3(data)
    ToCSV(data_3, col - 1, "Pumadyn2-3")
    data_5 = bin_to_5(data)
    ToCSV(data_5, col - 1, "Pumadyn2-5")
    data_10 = bin_to_10(data)
    ToCSV(data_10, col - 1, "Pumadyn2-10")


def Load_Bank8FM():
    col = 9
    data1 =[]
    data2 = []
    for line in open(dir + 'Bank8FM/bank8FM.data', "r"):
        a,b,c,d,e,f,g,h,i, j= line.split(' ')
        [a, b, c, d, e, f, g, h, i] = list(map(float, [a, b, c, d, e, f, g, h, i]))
        data1.append([a,b,c,d,e,f,g,h,i])

    for line in open(dir + 'Bank8FM/bank8FM.test',"r"):
        a,b,c,d,e,f,g,h,i, j= line.split(' ')
        [a, b, c, d, e, f, g, h, i] = list(map(float, [a, b, c, d, e, f, g, h, i]))
        data2.append([a,b,c,d,e,f,g,h,i])

    data1 = np.array(data1)
    data2 = np.array(data2)
    data = np.insert(data2, 0, data1, axis=0)
    data = pd.DataFrame(data)
    data.columns = [i for i in range(col)]
    sortName = col - 1
    data = data.sort_values(by=sortName)
    data_3 = bin_to_3(data)
    ToCSV(data_3, col - 1, "bank1-3")
    data_5 = bin_to_5(data)
    ToCSV(data_5, col - 1, "bank1-5")
    data_10 = bin_to_10(data)
    ToCSV(data_10, col - 1, "bank1-10")


def Load_Bank32nh():
    col = 33
    data1 =[]
    data2 = []
    for line in open(dir + 'Bank32nh/bank32nh.data', "r"):
        a,b,c,d,e,f,g,h,i, j,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,c1,c2,c3,c4 = line.split(' ')
        [a,b,c,d,e,f,g,h,i, j,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,c1,c2,c3] = list(map(float, [a,b,c,d,e,f,g,h,i, j,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,c1,c2,c3]))
        data1.append([a,b,c,d,e,f,g,h,i, j,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,c1,c2,c3])

    for line in open(dir + 'Bank32nh/bank32nh.test',"r"):
        a, b, c, d, e, f, g, h, i, j, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, c1, c2, c3,c4 = line.split(
            ' ')
        [a, b, c, d, e, f, g, h, i, j, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10,
         c1, c2,c3] = list(map(float,
                            [a, b, c, d, e, f, g, h, i, j, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, b1, b2, b3, b4, b5,
                             b6, b7, b8, b9, b10, c1, c2,c3]))
        data2.append([a,b,c,d,e,f,g,h,i, j,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,c1,c2,c3])

    data1 = np.array(data1)
    data2 = np.array(data2)
    data = np.insert(data2, 0, data1, axis=0)
    data = pd.DataFrame(data)
    data.columns = [i for i in range(col)]
    sortName = col - 1
    data = data.sort_values(by=sortName)
    data_3 = bin_to_3(data)
    ToCSV(data_3, col - 1, "bank2-3")
    data_5 = bin_to_5(data)
    ToCSV(data_5, col - 1, "bank2-5")
    data_10 = bin_to_10(data)
    ToCSV(data_10, col - 1, "bank2-10")


def Load_Stocks():
    col = 10
    data = pd.read_csv(dir + 'stock/stock.data', sep=',', names=[i for i in range(col)])
    data = np.array(data)
    data = pd.DataFrame(data)
    data.columns = [i for i in range(col)]
    sortName = col - 1
    data = data.sort_values(by=sortName)
    data_3 = bin_to_3(data)
    ToCSV(data_3, col - 1, "stock-3")
    data_5 = bin_to_5(data)
    ToCSV(data_5, col - 1, "stock-5")
    data_10 = bin_to_10(data)
    ToCSV(data_10, col - 1, "stock-10")


def Load_delta_elevators():
    col = 7
    data = pd.read_csv(dir + 'delta_elevators/delta_elevators.data', sep=' ', names=[i for i in range(col)])
    data = np.array(data)
    data = pd.DataFrame(data)
    data.columns = [i for i in range(col)]
    sortName = col - 1
    data = data.sort_values(by=sortName)
    data_3 = bin_to_3(data)
    ToCSV(data_3, col - 1, "delta_elevators-3")
    data_5 = bin_to_5(data)
    ToCSV(data_5, col - 1, "delta_elevators-5")
    data_10 = bin_to_10(data)
    ToCSV(data_10, col - 1, "delta_elevators-10")