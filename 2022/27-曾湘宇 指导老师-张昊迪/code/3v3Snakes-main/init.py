from custom_model import ACCNNModel
import pickle

model = ACCNNModel(observation_space = [128,128,3], action_space = 5)
weights = model.get_weights()
'''
关机后数据不能保存，将对象存储在文件中，日后可以读取
保存和读取时需要有class的声明，除非是python空间中已经声明的了
‘w’：ֻ打开即默认创建一个新文件，如果文件已存在，则覆盖写(即文件内原始数据会被新写入的数据清空覆盖)。
‘w+’：写读。打开创建新文件并写入数据，如果文件已存在，则覆盖写。
‘wb’：表示以二进制写方式打开，只能写文件， 如果文件不存在，创建该文件；如果文件已存在，则覆盖写。
'''
with open('/root/model/init_model.pt', 'wb') as f:
    pickle.dump(weights, f)
