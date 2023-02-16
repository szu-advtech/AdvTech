import os
import time
import argparse

import numpy as np
import torch
from torch.autograd import Variable
from HyperTools import *
from Models import *
from PIL import Image

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
        if args.model=='DilatedFCN':
            Model = DilatedFCN(num_features=num_features,num_classes=num_classes)
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

        # adversarial attack
        # epsilon = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8, 10]
        epsilon = [0.1]
        for i in range(len(epsilon)):
            print('Generate adversarial example with epsilon = %.2f' % (epsilon[i]))
            processed_image = Variable(images)
            processed_image = processed_image.requires_grad_()
            label = torch.from_numpy(Y_tar).long().cuda()

            output, context_prior_map = Model(processed_image)
            seg_loss = criterion(output, context_prior_map, label)
            seg_loss.backward()

            adv_noise = epsilon[i] * processed_image.grad.data / torch.norm(processed_image.grad.data, float("inf"))

            processed_image.data = processed_image.data - adv_noise

            X_adv = torch.clamp(processed_image, 0, 1).cpu().data.numpy()[0]
            noise_image = X_adv - images.cpu().data.numpy()[0]
            noise_image[noise_image > 1] = 1
            noise_image[noise_image < 0] = 0

            if args.dataID == 1:
                # perturbation
                im = Image.fromarray(np.moveaxis((noise_image[[102, 56, 31], :, :] * 25500).astype('uint8'), 0, -1))
                im.save(save_path_prefix + 'perturbation' + str(epsilon[i]) + '.png', 'png')
                # advimage
                im = Image.fromarray(np.moveaxis((X_adv[[102, 56, 31], :, :] * 255).astype('uint8'), 0, -1))
                im.save(save_path_prefix + 'advimage' + str(epsilon[i]) + '.png', 'png')
                # norimage
                if i == 0:
                    im = Image.fromarray(
                        np.moveaxis((images.cpu().data.numpy()[0][[102, 56, 31], :, :] * 255).astype('uint8'), 0, -1))
                    im.save(save_path_prefix + 'norimage' + str(epsilon[i]) + '.png', 'png')
            elif args.dataID == 2:
                im = Image.fromarray(np.moveaxis((noise_image[[102, 56, 31], :, :] * 25500).astype('uint8'), 0, -1))
                im.save(save_path_prefix + 'perturbation' + str(epsilon[i]) + '.png', 'png')
                im = Image.fromarray(np.moveaxis((X_adv[[102, 56, 31], :, :] * 255).astype('uint8'), 0, -1))
                im.save(save_path_prefix + 'advimage' + str(epsilon[i]) + '.png', 'png')
            elif args.dataID == 5:
                # perturbation
                im = Image.fromarray(np.moveaxis((noise_image[[102, 56, 31], :, :] * 25500).astype('uint8'), 0, -1))
                # im = Image.fromarray(np.moveaxis((noise_image[[111, 62, 38], :, :] * 25500).astype('uint8'), 0, -1))
                im.save(save_path_prefix + 'perturbation' + str(epsilon[i]) + '.png', 'png')
                # advimage
                im = Image.fromarray(np.moveaxis((X_adv[[102, 56, 31], :, :] * 255).astype('uint8'), 0, -1))
                # im = Image.fromarray(np.moveaxis((X_adv[[111, 62, 38], :, :] * 255).astype('uint8'), 0, -1))
                im.save(save_path_prefix + 'advimage' + str(epsilon[i]) + '.png', 'png')
                # norimage
                if i == 0:
                    im = Image.fromarray(
                        np.moveaxis((images.cpu().data.numpy()[0][[102, 56, 31], :, :] * 255).astype('uint8'), 0, -1))
                        # np.moveaxis((images.cpu().data.numpy()[0][[111, 62, 38], :, :] * 255).astype('uint8'), 0, -1))
                    im.save(save_path_prefix + 'norimage' + str(epsilon[i]) + '.png', 'png')



        # adversarial attack
        processed_image = Variable(images)
        processed_image = processed_image.requires_grad_()
        label = torch.from_numpy(Y_tar).long().cuda()

        output, context_prior_map = Model(processed_image)
        # output = Model(processed_image)
        #### plot context_map
        ## plot context_prior_map
        # plot_context_prior_map = np.squeeze(context_prior_map.cpu().data.numpy())
        # plt.imsave(save_path_prefix+'Affinity2_epoch'+repr(int(epoch))+'.png',plot_context_prior_map)
        ## plot context_prior_map_ori
        # plot_context_prior_map = np.squeeze(context_prior_map_ori.cpu().data.numpy())
        # plt.imsave(save_path_prefix + 'Affinity_ori_epoch' + repr(int(epoch)) + '.png', plot_context_prior_map)

        seg_loss = criterion(output, context_prior_map, label)
        # seg_loss = criterion(output,label)

        #### Test time ####
        te1_time = time.time()

        seg_loss.backward()
        adv_noise = args.epsilon * processed_image.grad.data / torch.norm(processed_image.grad.data,float("inf"))

        processed_image.data = processed_image.data - adv_noise

        X_adv = torch.clamp(processed_image, 0, 1).cpu().data.numpy()[0]
        X_adv = np.reshape(X_adv,(1,num_features,h,w))

        adv_images = torch.from_numpy(X_adv).float().cuda()
        # output = Model(adv_images)

        output, context_prior_map = Model(adv_images)
        _, predict_labels = torch.max(output, 1)

        te2_time =time.time() - te1_time

        predict_labels = np.squeeze(predict_labels.detach().cpu().numpy()).reshape(-1)
        print('Train_time: %.2f, Test_time: %.2f' % ( tr2_time, te2_time))
        # results on the adversarial test set
        OA,AA,kappa,ProducerA = CalAccuracy(predict_labels[test_array],Y[test_array])

        img = DrawResult(np.reshape(predict_labels+1,-1),args.dataID)
        plt.imsave(save_path_prefix+args.model+'_FGSM_OA'+repr(int(OA*10000))+'_kappa'+repr(int(kappa*10000))+'Epsilon'+str(args.epsilon)+'.png',img)

        print('OA=%.3f,AA=%.3f,Kappa=%.3f' %(OA*100,AA*100,kappa*100))
        print('producerA:',ProducerA)
        print('iter:', eep + 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # DataName = {1:'PaviaU',2:'Salinas',3:'IndinaP',4:'HoustonU',5:'xqh';6:KSC}

    parser.add_argument('--dataID', type=int, default=5)
    parser.add_argument('--save_path_prefix', type=str, default='./')
    parser.add_argument('--model', type=str, default='CNet')
    
    # train
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--decay', type=float, default=5e-5)
    parser.add_argument('--epsilon', type=float, default=0.06)
    parser.add_argument('--train_samples', type=int, default=100)
    parser.add_argument('--iter', type=int, default=1)

    main(parser.parse_args())
