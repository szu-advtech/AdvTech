import os
import time
import argparse

import numpy as np
import torch
from torch.autograd import Variable
from HyperTools import *
from Models import *
import  os
os.environ["CUDA_VISIBLE_DEVICES"]="4"

DataName = {1:'PaviaU',2:'Salinas',3:'IndinaP',4:'HoustonU',5:'xqh'}

def main(args):
    if args.dataID==1:
        num_classes = 9
        num_features = 103
        m = 610
        n = 340
        save_pre_dir = '../Data/PaviaU/'
    elif args.dataID==2:
        num_classes = 16  
        num_features = 204
        m = 512
        n = 217
        save_pre_dir = '../Data/Salinas/'
    elif args.dataID == 3:
        num_classes = 16
        num_features = 200
        m = 145
        n = 145
        save_pre_dir = '../Data/IndianP/'
    elif args.dataID == 4:
        num_classes = 15
        num_features = 144
        m = 349
        n = 1905
        save_pre_dir = '../Data/HoustonU/'
    elif args.dataID == 5:
        num_classes = 6
        num_features = 310
        m = 456
        n = 352
        save_pre_dir = './Data/xqh/'
    ####### load datas#1 #######
    # X = np.load(save_pre_dir+'X.npy')
    # _,h,w = X.shape
    # Y = np.load(save_pre_dir+'Y.npy')
    iter = args.iter
    OA_clean, OA_attack = np.zeros(iter), np.zeros(iter)
    AA_clean, AA_attack = np.zeros(iter), np.zeros(iter)
    Kappa_clean, Kappa_attack = np.zeros(iter), np.zeros(iter)
    CA_clean, CA_attack = np.zeros((num_classes, iter)), np.zeros((num_classes, iter))
    for eep in range(iter):

        ####### load datas#2 #######
        dataID = args.dataID
        X,Y,train_array,test_array = LoadHSI(dataID,args.train_samples)
        Y -= 1
        _, h, w = X.shape

        X_train = np.reshape(X,(1,num_features,h,w))
        ####### load datas#1 #######
        # train_array = np.load(save_pre_dir+'train_array.npy')
        # test_array = np.load(save_pre_dir+'test_array.npy')
        Y_train = np.ones(Y.shape)*255
        Y_train[train_array] = Y[train_array]
        Y_train = np.reshape(Y_train,(1,h,w))

        # define the targeted label in the attack
        Y_tar = np.zeros(Y.shape)
        Y_tar = np.reshape(Y_tar,(1,h,w))

        save_path_prefix = args.save_path_prefix+'Exp_'+DataName[args.dataID]+'/'

        if os.path.exists(save_path_prefix)==False:
            os.makedirs(save_path_prefix)

        num_epochs = 1000
        if args.model=='SACNet':
            Model = SACNet(num_features=num_features,num_classes=num_classes)
        elif args.model=='DilatedFCN':
            Model = DilatedFCN(num_features=num_features,num_classes=num_classes)
        elif args.model=='SpeFCN':
            Model = SpeFCN(num_features=num_features,num_classes=num_classes)
            num_epochs = 3000
        elif args.model=='SpaFCN':
            Model = SpaFCN(num_features=num_features,num_classes=num_classes)
        elif args.model=='SSFCN':
            Model = SSFCN(num_features=num_features,num_classes=num_classes)
        elif args.model=='NDLNet':
            Model = NDLNet(n_bands=num_features,classes=num_classes)
            num_epochs = 500
        elif args.model=='CNet':
            # [296, 161] // [610, 340]   // [150,82] Aggregation kernel = 3 // [145, 78] Aggregation kernel = 5
            # // [141, 73] Aggregation kernel = 7
            # prior_size = [152, 85]

            # prior_size = [int((((m-12)/2)-12)/2), int((((n-12)/2)-12)/2)]
            # m = 150
            # n= 150
            prior_size = [int((((m-12)/2)-12)/2), int((((n-12)/2)-12)/2)]

            num_epochs =500
            Model = CNet(num_features=num_features, prior_size= prior_size, num_classes=num_classes)
        elif args.model == 'CNet_CP':
            prior_size = [int((((m-12)/2)-12)/2), int((((n-12)/2)-12)/2)]
            # prior_size = [int(m/4), int(n/4)]
            num_epochs = 500
            Model = CNet_CP(num_features=num_features, prior_size= prior_size, num_classes=num_classes)
        elif args.model=='CNet_Agg':
            prior_size = [int((((m-12)/2)-12)/2), int((((n-12)/2)-12)/2)]
            num_epochs =500
            Model = CNet_Agg(num_features=num_features, prior_size= prior_size, num_classes=num_classes)
        elif args.model=='CNet_wo_all':
            prior_size = [int((((m-12)/2)-12)/2), int((((n-12)/2)-12)/2)]
            num_epochs =500
            Model = CNet_wo_all(num_features=num_features, prior_size= prior_size, num_classes=num_classes)

        # torch.backends.cudnn.enabled = False

        Model = Model.cuda()
        Model.train()
        ## optimizer 1
        optimizer = torch.optim.Adam(Model.parameters(),lr=args.lr,weight_decay=args.decay)
        ## optimizer 2
        # optimizer = torch.optim.SGD(Model.parameters(), lr= args.lr, momentum=0.9, weight_decay=args.decay, nesterov=True)

        images = torch.from_numpy(X_train).float().cuda()
        label = torch.from_numpy(Y_train).long().cuda()
        # if args.model=='CNet':
        # criterion = myLoss(num_classes=num_classes, down_sample_size=prior_size).cuda()
        # else:
        # criterion = CrossEntropy2d().cuda()
        criterion = myLoss(num_classes=num_classes, down_sample_size=prior_size).cuda()

        #### Train time ####
        tr1_time = time.time()
        # train the classification model
        for epoch in range(num_epochs):
            adjust_learning_rate(optimizer,args.lr,epoch,num_epochs)
            tem_time = time.time()

            optimizer.zero_grad()
            output, context_prior_map = Model(images)
            # output = Model(images)

            seg_loss = criterion(output, context_prior_map, label)
            # seg_loss = criterion(output,label)
            seg_loss.backward()

            optimizer.step()

            batch_time = time.time()-tem_time
            if (epoch+1) % 1 == 0:
                print('epoch %d/%d:  time: %.2f cls_loss = %.3f'%(epoch+1, num_epochs,batch_time,seg_loss.item()))


        tr2_time = time.time()-tr1_time

        Model.eval()
        output, context_prior_map = Model(images)
        # output = Model(images)

        _, predict_labels = torch.max(output, 1)
        predict_labels = np.squeeze(predict_labels.detach().cpu().numpy()).reshape(-1)

        # results on the clean test set
        OA,AA,kappa,ProducerA = CalAccuracy(predict_labels[test_array],Y[test_array])

        OA_clean[eep] = OA
        AA_clean[eep] = AA
        Kappa_clean[eep] = kappa
        CA_clean[0:num_classes, eep] = ProducerA

        # img = DrawResult(np.reshape(predict_labels+1,-1),args.dataID)
        # plt.imsave(save_path_prefix+args.model+'_clean_OA'+repr(int(OA*10000))+'_kappa'+repr(int(kappa*10000))+'.png',img)

        print('OA=%.3f,AA=%.3f,Kappa=%.3f' %(OA*100,AA*100,kappa*100))
        print('producerA:',ProducerA)

############### adversarial attack (C&W) ###############
        processed_image = Variable(images)
        # label = torch.from_numpy(Y_tar).long().cuda()
        label_mask = np.reshape(Y_tar, (1, h, w))
        label_mask = torch.from_numpy(label_mask + 1).long().cuda()
        # label_mask = F.one_hot(label_mask, num_classes=num_classes+1)
        label_mask = F.one_hot(label_mask)
        label_mask = label_mask.permute(0, 3, 1, 2)
        label_mask = label_mask[:, 1:, :, :]  # 去除0项
        # Start iteration
        for i in range(args.num_iter):
            processed_image = processed_image.requires_grad_()

            output, context_prior_map = Model(processed_image)
            # seg_loss = criterion(output,label)
            correct_logit = torch.mean(torch.sum(label_mask * output, dim=1)[0])
            wrong_logit = torch.mean(torch.max((1 - label_mask) * output, dim=1)[0])
            loss = -(correct_logit - wrong_logit + args.C)

            loss.backward()
            adv_noise = args.epsilon * processed_image.grad.data / torch.norm(processed_image.grad.data, float("inf"))
            processed_image.data = processed_image.data - adv_noise

            # X_adv = torch.clamp(processed_image, 0, 1).cpu().data
            X_adv = torch.clamp(processed_image, 0, 1).data
            processed_image = X_adv

############### adversarial attack (IFSGM) ###############
        # processed_image = Variable(images)
        # label = torch.from_numpy(Y_tar).long().cuda()
        #
        # # Start iteration
        # for i in range(args.num_iter):
        #     processed_image = processed_image.requires_grad_()
        #
        #     output, context_prior_map = Model(processed_image)
        #     # output = Model(processed_image)
        #     #### plot context_map
        #     ## plot context_prior_map
        #     # plot_context_prior_map = np.squeeze(context_prior_map.cpu().data.numpy())
        #     # plt.imsave(save_path_prefix+'Affinity2_epoch'+repr(int(epoch))+'.png',plot_context_prior_map)
        #     ## plot context_prior_map_ori
        #     # plot_context_prior_map = np.squeeze(context_prior_map_ori.cpu().data.numpy())
        #     # plt.imsave(save_path_prefix + 'Affinity_ori_epoch' + repr(int(epoch)) + '.png', plot_context_prior_map)
        #
        #     seg_loss = criterion(output, context_prior_map, label)
        #     # seg_loss = criterion(output,label)
        #     seg_loss.backward()
        #     adv_noise = args.epsilon * processed_image.grad.data / torch.norm(processed_image.grad.data,float("inf"))
        #     processed_image.data = processed_image.data - adv_noise
        #
        #     X_adv = torch.clamp(processed_image, 0, 1).data
        #     processed_image = X_adv
##################################################
        processed_image = processed_image.cuda().cpu().numpy()[0]
        X_adv = np.reshape(processed_image,(1,num_features,h,w))
        adv_images = torch.from_numpy(X_adv).float().cuda()

        # output = Model(adv_images)
        output, context_prior_map = Model(adv_images)

        _, predict_labels = torch.max(output, 1)

        predict_labels = np.squeeze(predict_labels.detach().cpu().numpy()).reshape(-1)
        # print('Train_time: %.2f, Test_time: %.2f' % ( tr2_time, te2_time))
        # results on the adversarial test set
        OA,AA,kappa,ProducerA = CalAccuracy(predict_labels[test_array],Y[test_array])

        img = DrawResult(np.reshape(predict_labels+1,-1),args.dataID)
        plt.imsave(save_path_prefix+args.model+'_FGSM_OA'+repr(int(OA*10000))+'_kappa'+repr(int(kappa*10000))+'Epsilon'+str(args.epsilon)+'.png',img)

        print('OA=%.3f,AA=%.3f,Kappa=%.3f' %(OA*100,AA*100,kappa*100))
        print('producerA:',ProducerA)
        print('iter:', eep + 1)

        OA_attack[eep] = OA
        AA_attack[eep] = AA
        Kappa_attack[eep] = kappa
        CA_attack[0:num_classes, eep] = ProducerA

    OA_1 = np.average(OA_clean)
    OA_std1 = np.std(OA_clean)
    AA_1 = np.average(AA_clean)
    AA_std1 = np.std(AA_clean)
    Kappa_1 = np.average(Kappa_clean)
    Kappa_std1 = np.std(Kappa_clean)
    CA_1 = np.average(CA_clean,1)
    CA_std1 = np.std(CA_clean,1)

    OA_2 = np.average(OA_attack)
    OA_std2 = np.std(OA_attack)
    AA_2 = np.average(AA_attack)
    AA_std2 = np.std(AA_attack)
    Kappa_2 = np.average(Kappa_attack)
    Kappa_std2 = np.std(Kappa_attack)
    CA_2 = np.average(CA_attack,1)
    CA_std2 = np.std(CA_attack, 1)
    print('===============Clean===============')
    print('OA=%.3f,AA=%.3f,Kappa=%.3f' % (OA_1 * 100, AA_1 * 100, Kappa_1 * 100))
    print('OA_std=%.3f,AA_std=%.3f,Kappa_std=%.3f' % (OA_std1 * 100, AA_std1 * 100, Kappa_std1 * 100))
    print('producerA:', CA_1)
    print('producerA_std:', CA_std1)
    print('===============Attack===============')
    print('OA=%.3f,AA=%.3f,Kappa=%.3f' % (OA_2 * 100, AA_2 * 100, Kappa_2 * 100))
    print('OA_std=%.3f,AA_std=%.3f,Kappa_std=%.3f' % (OA_std2 * 100, AA_std2 * 100, Kappa_std2 * 100))
    print('producerA:', CA_2)
    print('producerA_std:', CA_std2)

    # import pandas as pd
    # # create and writer datas to excel
    # data_clean = pd.DataFrame(OA_1)
    # data_attack = pd.DataFrame(OA_2)
    # writer = pd.ExcelWriter('OA_'+ "dataID"+ args.dataID + '_60pre_class_0308.xlsx')
    # data_clean.to_excel(writer, 'page1', float_format='%.6f')
    # data_attack.to_excel(writer, 'page2', float_format='%.6f')
    # writer.save()
    #
    # data = pd.DataFrame(OA_1)
    # writer = pd.ExcelWriter('OA_clean'+ "dataID"+ args.dataID + '_60pre_class_0308.xlsx')
    # data.to_excel(writer, 'page1', float_format='%.6f')
    # writer.save()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # DataName = {1:'PaviaU',2:'Salinas',3:'IndinaP',4:'HoustonU',5:'xqh';6:KSC}

    parser.add_argument('--dataID', type=int, default= 3)
    parser.add_argument('--save_path_prefix', type=str, default='./')
    parser.add_argument('--model', type=str, default='CNet')
    
    # trainy
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--decay', type=float, default=5e-5)
    parser.add_argument('--epsilon', type=float, default=0.04)
    parser.add_argument('--train_samples', type=int, default=100)
    parser.add_argument('--iter', type=int, default=10)
    parser.add_argument('--num_iter', type=int, default=5)
    parser.add_argument('--C', type=float, default=10)
    main(parser.parse_args())
