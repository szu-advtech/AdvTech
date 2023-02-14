import combine_files
import extract_feature
import deeplearning
import select_feature


def AVEC2014(task,fre,N,t_length):
    device_num=4
    theshold=2
    openface_dir='AVEC2014/Openface_delete'
    feature_dir='AVEC2014/features_delete'
    label_dir='AVEC2014/AVEC2014_DepressionLabels'
    combine_dir='AVEC2014/combine/features_delete'
    extract_feature.files_deal_AVEC2014(openface_dir,feature_dir,fre,N,t_length,task)
    combine_files.file_name_AVEC2014(feature_dir,label_dir,combine_dir,task)
    train_file,test_file,feature=select_feature.connect_file_AVEC2014(combine_dir,combine_dir,theshold,task)
    deeplearning.training_AVEC2014(train_file,test_file,feature,device_num,100,64)

def AVEC2019(fre,N,t_length):
    device_num=4
    theshold=2
    openface_dir='AVEC2019/Openface_delete'
    feature_dir='AVEC2019/features_delete'
    label_dir='AVEC2019/labels'
    combine_dir='AVEC2019/combine/features_delete'
    extract_feature.files_deal_AVEC2019(openface_dir, feature_dir,fre,N,t_length)
    combine_files.file_name_AVEC2019(feature_dir,label_dir,combine_dir)
    train_file,test_file,dev_file,feature=select_feature.connect_file_AVEC2019(combine_dir,combine_dir,theshold)
    deeplearning.training_AVEC2019(train_file,test_file,dev_file,feature,device_num,100,25)

if __name__ == "__main__":
    AVEC2019(500,200,200*29)
    # AVEC2014('Northwind',90,32,32*29)
    #AVEC2014('Freeform', 100, 40, 40 * 29)
    # AVEC2014('Northwind', 100, 40, 40 * 29)
    #AVEC2014('Northwind', 500, 150, 150 * 29)
