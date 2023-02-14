import os
import csv
import pandas as pd

file_dir = '/data/zyz/BPD-main/logs/mhealth/None_ECAnet'
cnn_csv_name = '50_64_cnn_0.0001.csv'
convlstm_csv_name = '50_64_convlstmv2_0.0001.csv'
cnn_csv_save_name = 'cnn.csv'
convlstm_csv_save_name = 'convlstmv2.csv'

def take1(ele):
    return ele[0]

if __name__ == '__main__':
    path_lists = os.listdir(file_dir)
    print(path_lists)
    name = ['target','max_F1']
    data = []
    data1 = []
    for path_list in path_lists:
        sub_path_lists = os.listdir(file_dir + '/' + path_list)
        if cnn_csv_name in sub_path_lists:
            if path_list.find('cnn') != -1:
                with open (file_dir + '/' + path_list + '/' + cnn_csv_name,mode='r') as f:
                    target = int(path_list[-3])
                    reader = csv.reader(f)
                    max_f1 = 0
                    for row in reader:
                        if float(row[2]) > max_f1:
                            max_f1 = float(row[2])
                    data1.append([target, max_f1])

        if convlstm_csv_name in sub_path_lists:
            if path_list.find('convlstm') != -1:
                with open (file_dir + '/' + path_list + '/' + convlstm_csv_name,mode='r') as f:
                    target = int(path_list[-3])
                    reader = csv.reader(f)
                    max_f1 = 0
                    for row in reader:
                        if float(row[2]) > max_f1:
                            max_f1 = float(row[2])
                    data.append([target, max_f1])
    if data1:
        data1.sort(key=lambda ele: ele[0], reverse=False)
        result_cnn = pd.DataFrame(columns = name, data= data1)
        result_cnn.to_csv(file_dir + '/' + cnn_csv_save_name, header=None, index=None)
    if data:
        data.sort(key=lambda ele: ele[0], reverse=False)
        result_convlstm = pd.DataFrame(columns = name, data=data)
        result_convlstm.to_csv(file_dir + '/' + convlstm_csv_save_name,header=None, index=None)

