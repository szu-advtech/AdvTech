import math
import numpy as np
import random
from PIL import Image
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

#计算三点间的夹角
def cal_ang(point_1,point_2,point_3):
    a=math.sqrt((point_2[0]-point_3[0])*(point_2[0]-point_3[0])+(point_2[1]-point_3[1])*(point_2[1]-point_3[1]))
    b=math.sqrt((point_1[0]-point_3[0])*(point_1[0]-point_3[0])+(point_1[1]-point_3[1])*(point_1[1] - point_3[1]))
    c=math.sqrt((point_1[0]-point_2[0])*(point_1[0]-point_2[0])+(point_1[1]-point_2[1])*(point_1[1]-point_2[1]))

    degree=math.degrees(math.acos((a*a-b*b-c*c)/(-2*b*c)))
    if point_3[0]<point_1[0]:
        degree=-degree
    return degree

#计算中心坐标
def getCenter(label,centerLabel):
    if type(label)==type(np.array(1)):
        mask=(label==centerLabel).astype(float)
        x_dim,y_dim=mask.shape
        mask=mask.reshape(-1,x_dim*y_dim)
        index=np.where(mask>0)    
        indexW=index[1]%y_dim
        indexH=index[1]//y_dim
        indexW=indexW.mean()
        indexH=indexH.mean()
        return int(indexW),int(indexH)
    else:
        mask=(label==centerLabel).float()
        x_dim,y_dim=mask.size()
        index=torch.where(mask>0)
        indexH=index[0].float().mean()
        indexW=index[1].float().mean()
        return int(indexW),int(indexH)


#随机缩放
def RandomRescale(img,label,prior,minScale,maxScale):
    img,label=img.unsqueeze(0),label.unsqueeze(0)
    if len(img.size())==4:
        chs,z,h,w=img.size()
        scaleRange=random.uniform(minScale,maxScale)
        targetH,targetW=int(h*scaleRange),int(w*scaleRange)
        #####################################################
        
        batch_n_prior,chs_prior,h_prior,w_prior=prior.size()
        targetH_prior,targetW_prior=int(h_prior*scaleRange),int(w_prior*scaleRange)
        if targetH_prior%2!=0:
            targetH_prior+=1 
            targetW_prior+=1
        #paddingH,paddingW=int((h_prior-targetH_prior)/2),int((w_prior-targetW_prior)/2)
        #paddingHup=int((64+18)*(1-scaleRange))
        paddingHup=int((80+24)*(1-scaleRange))
        paddingHdown=h_prior-targetH_prior-paddingHup
        paddingWleft=int((w_prior-targetW_prior)/2)
        paddingWright=paddingWleft
        
        #scalePriorOther=transforms.Resize((targetH_prior,targetW_prior),interpolation=Image.NEAREST)(prior[:,1:chs_prior,:,:])
        #scalePriorFirst=transforms.Resize((targetH_prior,targetW_prior),interpolation=Image.NEAREST)(prior[:,0:1,:,:])
        scalePriorOther=transforms.Resize((targetH_prior,targetW_prior))(prior[:,1:chs_prior,:,:])
        scalePriorFirst=transforms.Resize((targetH_prior,targetW_prior))(prior[:,0:1,:,:])
        ZeroPad=nn.ZeroPad2d(padding=(paddingWleft,paddingWright,paddingHup,paddingHdown))#左右上下
        OnesPad=nn.ConstantPad2d(padding=(paddingWleft,paddingWright,paddingHup,paddingHdown),value=1.0)

        scalePriorOther=ZeroPad(scalePriorOther)
        scalePriorFirst=OnesPad(scalePriorFirst)
        prior_res=torch.cat((scalePriorFirst,scalePriorOther),dim=1)
        #####################################################





        img_res,label_res=[],[]
        for index in range(z):
            #scaleImg=transforms.Resize((targetH,targetW),interpolation=Image.NEAREST)(img[:,index,:,:])
            scaleImg=transforms.Resize((targetH,targetW))(img[:,index,:,:])#缩放较大时改用二维线性插值
            scaleLabel=transforms.Resize((targetH,targetW),interpolation=Image.NEAREST)(label[:,index,:,:])
            img_res.append(scaleImg)
            label_res.append(scaleLabel)        
        img_res,label_res=torch.cat(img_res,dim=0),torch.cat(label_res,dim=0)
        return img_res,label_res,prior_res
        #return img_res,label_res,prior
    elif len(img.size())==3:
        chs,h,w=img.size()
        scaleRange=random.uniform(minScale,maxScale)
        chs,h,w=img.size()
        targetH,targetW=int(h*scaleRange),int(w*scaleRange)
        scaleImg=transforms.Resize((targetH,targetW),interpolation=Image.NEAREST)(img)
        scaleLabel=transforms.Resize((targetH,targetW),interpolation=Image.NEAREST)(label)

        scaleImg,scaleLabel=scaleImg.squeeze(0),scaleLabel.squeeze(0)
        return scaleImg,scaleLabel
    else:
        print('Abnormal format')
        return None,None

def centerCrop(img,label):
    if len(img.size())==3:
        cropedImg,cropedLabel=[],[]
        for i in range(img.size(0)):
            if (label[i,:,:]==1).sum()>0: 
                indexWL,indexHL=[int(index) for index in getCenter(label[i,:,:],1)]#左心室中心
            elif (label[i,:,:]==2).sum()>0:
                indexWL,indexHL=[int(index) for index in getCenter(label[i,:,:],2)]#左心室中心
            else:
                indexWL,indexHL=[int(index) for index in getCenter(label[i,:,:],3)]#左心室中心

            cropRange=64#裁剪的范围
            HShift=18
            cropedImg.append(img[i,indexHL-cropRange-HShift:indexHL+cropRange-HShift,indexWL-cropRange:indexWL+cropRange].unsqueeze(0))
            cropedLabel.append(label[i,indexHL-cropRange-HShift:indexHL+cropRange-HShift,indexWL-cropRange:indexWL+cropRange].unsqueeze(0))
        cropedImg,cropedLabel=torch.cat(cropedImg,dim=0),torch.cat(cropedLabel,dim=0)
        return cropedImg,cropedLabel    
    elif len(img.size())==2:
        if (label[:,:]==1).sum()>0: 
            indexWL,indexHL=[int(index) for index in getCenter(label[:,:],1)]#左心室中心
        else:
            indexWL,indexHL=[int(index) for index in getCenter(label[:,:],2)]#左心室中心

        cropRange=64#裁剪的范围
        HShift=15
        cropedImg=img[indexHL-cropRange-HShift:indexHL+cropRange-HShift,indexWL-cropRange:indexWL+cropRange]
        cropedLabel=label[indexHL-cropRange-HShift:indexHL+cropRange-HShift,indexWL-cropRange:indexWL+cropRange]
        return cropedImg,cropedLabel 
    else:
        print('Abnormal format')
        return None,None

def centerCrop128Slice(img,label):
    if (label[:,:]==1).sum()>0: 
        indexWL,indexHL=[int(index) for index in getCenter(label[:,:],1)]#左心室中心
    elif (label[:,:]==2).sum()>0:
        indexWL,indexHL=[int(index) for index in getCenter(label[:,:],2)]#心肌层中心
    else:
        indexWL,indexHL=[int(index) for index in getCenter(label[:,:],3)]#右心室中心
        indexHL+=10#右心室中心向下10个像素认为是左心室中心
    #cropRange=64#裁剪的范围
    #HShift=18
    cropRange=80#裁剪的范围
    HShift=24
    H,W=img.size()

    if indexHL-cropRange-HShift>=0 and indexHL+cropRange-HShift<=H and indexWL-cropRange>=0 and indexWL+cropRange<=W:
        cropedImg,cropedLabel=img,label
    else:
        if indexHL-cropRange-HShift>=0 and indexHL+cropRange-HShift<=H:
            cropedImg,cropedLabel=img,label
        if indexHL-cropRange-HShift<0:
            nowLen=-(indexHL-cropRange-HShift)
            #print('H warning!!!',indexHL-cropRange-HShift)

            insertImg=img[0,:].unsqueeze(0).repeat(nowLen,1)
            insertLabel=label[0,:].unsqueeze(0).repeat(nowLen,1)
            cropedImg=torch.cat([insertImg,img],0)
            cropedLabel=torch.cat([insertLabel,label],0)
            indexHL+=nowLen
            
        if indexHL+cropRange-HShift>H:
            nowLen=indexHL+cropRange-HShift-H
            #print('H warning!!!',indexHL+cropRange-HShift)            
            insertImg=img[H-1,:].unsqueeze(0).repeat(nowLen,1)
            insertLabel=label[H-1,:].unsqueeze(0).repeat(nowLen,1)
            cropedImg=torch.cat([img,insertImg],0)
            cropedLabel=torch.cat([label,insertLabel],0)

        if indexWL-cropRange<0:
            nowLen=-(indexWL-cropRange)
            #print('W warning!!!',indexWL-cropRange)

            insertImg=cropedImg[:,0].unsqueeze(1).repeat(1,nowLen)
            insertLabel=cropedLabel[:,0].unsqueeze(1).repeat(1,nowLen)
            cropedImg=torch.cat([insertImg,cropedImg],1)
            cropedLabel=torch.cat([insertLabel,cropedLabel],1)

            indexWL+=nowLen
        
        if indexWL+cropRange>W:
            nowLen=indexWL+cropRange-W
            #print('W warning!!!',indexWL+cropRange)

            insertImg=cropedImg[:,W-1].unsqueeze(1).repeat(1,nowLen)
            insertLabel=cropedLabel[:,W-1].unsqueeze(1).repeat(1,nowLen)
            cropedImg=torch.cat([cropedImg,insertImg],1)
            cropedLabel=torch.cat([cropedLabel,insertLabel],1)
    
    cropedImgRes=cropedImg[indexHL-cropRange-HShift:indexHL+cropRange-HShift,indexWL-cropRange:indexWL+cropRange]
    cropedLabelRes=cropedLabel[indexHL-cropRange-HShift:indexHL+cropRange-HShift,indexWL-cropRange:indexWL+cropRange]

    assert cropedLabelRes.size()==torch.zeros(160,160).size(),'crop shape 190 is error' 
    #assert cropedLabelRes.size()==torch.zeros(128,128).size(),'crop shape 190 is error' 
    return cropedImgRes,cropedLabelRes 

def centerCrop128(img,label):
    cropedImg,cropedLabel=[],[]
    for index in range(img.size(0)):
        cropedImgSlice,cropedLabelSlice=centerCrop128Slice(img[index],label[index])
        cropedImgSlice=(cropedImgSlice-cropedImgSlice.min())/(cropedImgSlice.max()-cropedImgSlice.min())
        cropedImgSlice*=255.0
        cropedImg.append(cropedImgSlice.unsqueeze(0))
        cropedLabel.append(cropedLabelSlice.unsqueeze(0))
    cropedImg=torch.cat(cropedImg,0)
    cropedLabel=torch.cat(cropedLabel,0)
    '''
    print(cropedImg.size(),cropedLabel.size())
    for i in range(cropedImg.size(0)):
        ax=plt.subplot(1,cropedImg.size(0),i+1)
        plt.imshow(cropedLabel[i].cpu().numpy(),cmap='gray')
    plt.show()
    '''
    return cropedImg,cropedLabel
    

#数据扩充且中心对齐
#需改进：因为填充方式选择了最近邻导致标签边界有点毛躁，需要改进
def centerAlignAndDataAug(img,label,prior):
    warpedImg,warpedLabel=torch.from_numpy(img).cuda(),torch.from_numpy(label).cuda()
    #scaledImg,scaledLabel=RandomRescale(warpedImg,warpedLabel,0.85,1.3)
    #scaledImg,scaledLabel=RandomRescale(warpedImg,warpedLabel,0.93,1.07)
    #scaledImg,scaledLabel=RandomRescale(warpedImg,warpedLabel,0.93,1.15)
    #scaledImg,scaledLabel,scaledPrior=RandomRescale(warpedImg,warpedLabel,prior,0.93,1.07)
    #scaledImg,scaledLabel,scaledPrior=RandomRescale(warpedImg,warpedLabel,prior,0.93,1.3)
    scaledImg,scaledLabel,scaledPrior=RandomRescale(warpedImg,warpedLabel,prior,0.85,1.3)
  
    cropedImg,cropedLabel=centerCrop128(scaledImg,scaledLabel)
    return cropedImg,cropedLabel,scaledPrior
def centerAlignAndDataAugVal(img,label):
    # warpedImg,warpedLabel=torch.from_numpy(img).cuda(),torch.from_numpy(label).cuda()
    warpedImg, warpedLabel = torch.from_numpy(img), torch.from_numpy(label)
    cropedImg,cropedLabel=centerCrop128(warpedImg,warpedLabel)
    return cropedImg,cropedLabel

def get_transform(opt,method=Image.BICUBIC,normalize=True,toTensor=True):
    transfrom_list=[]
    if toTensor:
        transfrom_list+=[transforms.Lambda(lambda img:__ToTensor(img))]
    if normalize:
        transfrom_list+=[transforms.Lambda(lambda img:__Normalize(img))]

    # 用Compose把多个步骤整合到一起#
    return transforms.Compose(transfrom_list)
    
    '''
    if toTensor:
        transfrom_list+=[transforms.ToTensor()]
    if normalize:
        transfrom_list+=[transforms.Normalize((0.5,),(0.5,))]
    return transforms.Compose(transfrom_list)
    '''

def __ToTensor(img):
    #img=(img-torch.min(img))/(torch.max(img)-torch.min(img))
    img=img/255.0
    img=img.unsqueeze(0)
    return img
def __Normalize(img):
    #new将数据按均值和标准差归一化
    '''
    chs,x,y=img.size()
    img_mean=img.mean((1,2)).unsqueeze(1).unsqueeze(1).expand(chs,x,y)
    img_var=torch.mean((img-img_mean)**2,(1,2))
    img_var=img_var.unsqueeze(1).unsqueeze(1).expand(chs,x,y)
    img_std=torch.sqrt(img_var)
    img=(img-img_mean)/img_std
    img=(img-img.min())/(img.max()-img.min())
    img=(img-0.5)/0.5
    '''
    #old假设数据均值标准差均为0.5
    img=(img-0.5)/0.5
    return img

