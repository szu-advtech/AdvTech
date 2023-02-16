import pandas as pd
# df = pd.read_csv('/home/Sathvik/Desktop/EVM/letter-recognition.csv',header = None) # importing test dataset


import os
print(os.getcwd())      #看相对路径

df = pd.read_csv('./aps_failure_training_set.csv') # importing test dataset
df1=df["class"]
df1.replace(['neg','pos'],[0,1] , inplace = True)
df2=df.iloc[:,1:]
# df1=df.iloc[:,1]
#
# df1.replace(['M','B'],[0,1] , inplace = True)
# print(df2)
# print(df1)
# df2=df.iloc[:,2:]
# print(df2)
df=pd.concat([df2,df1],axis=1)

# df['Label'].replace(['malware','goodware'],[0,1] , inplace = True)

# df1 = pd.read_csv('./secom_labels.data',header = None) # importing test dataset1
#
# df1["ct"]=df1
# df1.drop(df.columns[0], axis=1, inplace=True)
# print(df1)
# df1=df1['ct'].str.split(' ').str[0]
# print(df1)

# df1.drop(df.columns[1], axis=1, inplace=True)
# df=pd.concat([df,df1],axis=1)
df.to_csv('./aps_failure_training_set.txt',header = None, index = False)
# print(df)

# """
# ############################################################################################################################################
# 			Train-Test Split
# ############################################################################################################################################
# """
#
# train=df.sample(frac=0.70,random_state=200) #considering 70% of the data as train data and 30% as test
# test=df.drop(train.index)
#
# """
# ############################################################################################################################################
# 			Preparing train Dataset
# ############################################################################################################################################
# """
#
# known_classes = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T'] #comsidering 20 classses as known classes
# target_known_df = train.loc[df.iloc[:,0].isin(known_classes)] #preparing train dataset
# target_known_df = target_known_df.sort_values(0)
# target_known_df = target_known_df.replace(known_classes, list(range(1,1+len(known_classes)))) #encoding labels of the dataset
#
#
# """
# ############################################################################################################################################
# 			Preparing test Dataset
# ############################################################################################################################################
# """
#
#
# test = test.sort_values(0)
# test_classes = test[0].unique().tolist()
# unknown_classes = list(set(test_classes) - set(known_classes))
# test = test.replace(unknown_classes,[99]*len(unknown_classes)) #encoding unknown classes with label '99'
# test = test.replace(known_classes, list(range(1,1+len(known_classes))))
#
#
# """
# ############################################################################################################################################
# 			Saving datasets in csv format
# ############################################################################################################################################
# """
#
# # target_known_df.to_csv('/home/Sathvik/Desktop/EVM/train.csv',header = None, index = False)
# target_known_df.to_csv('./train.csv',header = None, index = False)
# # test.to_csv('/home/Sathvik/Desktop/EVM/test.csv',header = None, index = False)
# test.to_csv('./test.csv',header = None, index = False)