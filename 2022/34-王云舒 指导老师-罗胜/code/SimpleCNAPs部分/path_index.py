USEPreModel=False

#pretrainORmetatrainFLAG="pretrain"    #Warning: pretrain model is changed to pretrained=true
pretrainORmetatrainFLAG="metatrain"  #Warning: kernel_size=7/5

pretrain_kernel_size=5
metatrain_kernel_size=5

pretrain_trainways=32
pretrain_traindataset="pretrain.pkl" #10 means pretrained
pretrain_validdataset="pretrain.pkl"
pretrain_testdataset ="pretrain.pkl"
pretrain_destination ="pretrainmodel.pkl" # 7 x 7


metatrain_trainways=32
metatrain_trainshots=5
metatrain_pretrainmodel=pretrain_destination
#metatrain_pretrainmodel ="pretrainmodel.pkl"


metatrain_traindataset="metatrain.pkl"#
metatrain_validdataset="metatrain.pkl"#
metatrain_testdataset="test.pkl"#
metatrain_destination="metatrainmodel.pth"

metatrain_testways=1#
metatest_testmodel=metatrain_destination
#metatest_testmodel="metatrainmodel.pth"

print("pretrainORmetatrainFLAG:",pretrainORmetatrainFLAG)
if pretrainORmetatrainFLAG == "pretrain":
    
    print("pretrain_destination:",pretrain_destination)
else:
    print("metatrain_pretrainmodel:",metatrain_pretrainmodel)
    print("metatrain_destination:",metatrain_destination)