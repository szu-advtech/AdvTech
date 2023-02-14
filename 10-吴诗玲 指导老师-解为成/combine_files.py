import numpy as np
import h5py
import json
import pandas as pd
import os
import scipy.io as scio

def file_name_AVEC2014(feature_dir,label_dir,save_dir,task):
    listdir=os.listdir(feature_dir)
    j=task
    for i in listdir:
        print(i,j)
        df=pd.DataFrame(columns=['feature','label','person'])
        feature_files=os.path.join(feature_dir,i,j)
        files=os.listdir(feature_files)
        index=0
        for file in files:
            label_name=file[:6]+"Depression.csv"
            label_path=os.path.join(label_dir,i,label_name)
            label=pd.read_csv(label_path,header=None)
            feature_path=os.path.join(feature_files,file)
            feature=np.load(feature_path)
            df.loc[index]=[feature,int(label[0]),file[:5]]
            index+=1
        save_path=os.path.join(save_dir,i)
        os.makedirs(save_path, exist_ok=True)
        feature_path=os.path.join(save_path,j+'.npy')
        np.save(feature_path,df)

def file_name_AVEC2019(feature_dir,label_dir,save_dir):
    listdir=os.listdir(feature_dir)
    for i in listdir:
        label_file=os.path.join(label_dir,i+"_split.csv")
        label_content=pd.read_csv(label_file,header=0)
        df=pd.DataFrame(columns=['feature','label','person'])
        feature_files=os.path.join(feature_dir,i)
        files=os.listdir(feature_files)
        index=0
        for file in files:
            label=label_content.loc[label_content['Participant_ID']==int(file[:3]),'PHQ_Score']
            feature_path=os.path.join(feature_files,file)
            feature=np.load(feature_path)
            if np.all(feature==0):
                print(feature_path)
            df.loc[index]=[feature,int(label.values[0]),file[:3]]
            index+=1
        os.makedirs(save_dir, exist_ok=True)
        save_path=os.path.join(save_dir,i+'.npy')
        print(save_path)
        np.save(save_path,df)
if __name__ == "__main__":
    #file_name_AVEC2014(r'AVEC2014/features_delete',r'AVEC2014/AVEC2014_DepressionLabels','AVEC2014/combine/features_delete','Freeform')
    file_name_AVEC2019('AVEC2019/features_delete',"AVEC2019/labels/",'AVEC2019/combine/features_delete')