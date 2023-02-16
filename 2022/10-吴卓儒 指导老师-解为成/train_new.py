import argparse
import time
import tqdm
import os
os.environ["KMP_DUPLICATE_LIB_OK"]= "TRUE"
# 加载gc模块
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot
from torchvision import transforms
from data_loader import load_real_data,generate_real_data,resize,generate_fake_data_coarse,generate_fake_data_fine
from model_pytorch import Novel_Residual_Block,Coarse_Generator,Fine_Generator,Discriminator,Fundus2Angio,Fundus2Angio_Gan,freeze,unfreeze

def data_normalize_trans_tensor(input):
    cuda = True if torch.cuda.is_available() else False
    T = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    # normalize from [0,255] to [-1,1]
    input = (input - 127.5) / 127.5
    input_trans = np.transpose(input,(0,3,1,2))
    input_tensor = torch.from_numpy(input_trans).type(T)
    return input_tensor

def d_train_on_batch(model,loss_func,opt,input,label):
    unfreeze(model)
    # model.train() # 训练过程中开启 Dropout
    opt.zero_grad()
    pred_label = model(input)
    loss = loss_func(pred_label,label)
    loss.backward()
    opt.step()
    return loss.item()

def coarse_train_on_batch(model,loss_func,opt,input,label):
    unfreeze(model)
    # model.train() # 训练过程中开启 Dropout
    opt.zero_grad()
    pred, feature_pred = model(input)
    loss = loss_func(pred,label)
    loss.backward()
    opt.step()
    return loss.item()

def fine_train_on_batch(model,loss_func,opt,input,label):
    unfreeze(model)
    # model.train() # 训练过程中开启 Dropout
    opt.zero_grad()
    pred = model(input)
    loss = loss_func(pred,label)
    loss.backward()
    opt.step()
    return loss.item()

def gan_train_on_batch(model,loss_func,opt,input,label):
    unfreeze(model)
    # model.train() # 训练过程中开启 Dropout
    opt.zero_grad()
    coarse_output, fine_output, d1_output, d2_output, d3_output, d4_output = model(input)
    label1,label2,label2_same,label3 = label
    label_real,label_real_half = input[1],input[3]

    coarse_loss = loss_func(coarse_output,label_real_half)
    coarse_loss.backward()
    fine_loss = loss_func(fine_output,label_real)
    fine_loss.backward()

    # Fine D
    d1_loss2 = loss_func(d1_output,label1)
    # d1_loss2.requires_grad_()
    # d1_loss2.backward()

    d2_loss4 = loss_func(d2_output,label2)
    # d2_loss4.requires_grad_()
    # d2_loss4.backward()

    # Coarse D
    d3_loss6 = loss_func(d3_output,label2_same)
    # d3_loss6.requires_grad_()
    # d3_loss6.backward()

    d4_loss8 = loss_func(d4_output,label3)
    # d4_loss8.requires_grad_()
    # d4_loss8.backward()

    opt.step()
    return d1_loss2.item()+d2_loss4.item()+d3_loss6.item()+d4_loss8.item()+10*coarse_loss.item()+10*fine_loss.item()

def plot_history(d1_hist, d2_hist, d3_hist, d4_hist, d5_hist, d6_hist, d7_hist, d8_hist, cg_hist, fg_hist, g_hist, savedir='weights_plots'):
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    pyplot.plot(d1_hist, label='d1_loss_real')
    pyplot.plot(d2_hist, label='d1_loss_fake')
    pyplot.plot(d3_hist, label='d2_loss_real')
    pyplot.plot(d4_hist, label='d2_loss_fake')
    pyplot.plot(d5_hist, label='d3_loss_real')
    pyplot.plot(d6_hist, label='d3_loss_fake')
    pyplot.plot(d7_hist, label='d4_loss_real')
    pyplot.plot(d8_hist, label='d4_loss_fake')
    pyplot.plot(cg_hist, label='cg_loss')
    pyplot.plot(fg_hist, label='fg_loss')
    pyplot.plot(g_hist, label='g_loss')
    pyplot.legend()
    filename1 = savedir + '/plot_line_plot_loss.png'
    pyplot.savefig(filename1)
    pyplot.close()
    print('Saved %s' % (filename1))

    df = []
    df.append(d1_hist)
    df.append(d2_hist)
    df.append(d3_hist)
    df.append(d4_hist)
    df.append(d5_hist)
    df.append(d6_hist)
    df.append(d7_hist)
    df.append(d8_hist)
    df.append(cg_hist)
    df.append(fg_hist)
    df.append(g_hist)
    df = pd.DataFrame(df)
    df.index = ['d1_loss_real,', 'd1_loss_fake,', 'd2_loss_real,', 'd2_loss_fake,', 'd3_loss_real,', 'd3_loss_fake,',
                'd4_loss_real,', 'd4_loss_fake,', 'cg_loss,', 'fg_loss,', 'g_loss']
    filename2 = savedir + '/plot_line_plot_loss.csv'
    df.to_csv(filename2)
    print('Saved %s' % (filename2))

def visualize_save_weight_coarse(step, coarse_generator, dataset, n_samples=3, savedir='weights_plots'):
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # select a sample of input images
    n_patch = [1,1,1]
    [X_realA, X_realB], _ = generate_real_data(dataset, n_samples, n_patch)

    # Resize to half
    out_shape = (int(X_realA.shape[1]/2),int(X_realA.shape[2]/2))
    # print(out_shape) (256,256)
    [X_realA_half,X_realB_half] = resize(X_realA,X_realB,out_shape)

    X_realA = data_normalize_trans_tensor(X_realA)
    X_realB = data_normalize_trans_tensor(X_realB)
    X_realA_half = data_normalize_trans_tensor(X_realA_half)
    X_realB_half = data_normalize_trans_tensor(X_realB_half)
    X_realA = X_realA.to(device)
    X_realB = X_realB.to(device)
    X_realA_half = X_realA_half.to(device)
    X_realB_half = X_realB_half.to(device)

    # generate a batch of fake samples
    [X_fakeB_half, X_feature], _ = generate_fake_data_coarse(coarse_generator, X_realA_half, n_patch)

    # scale all pixels from [-1,1] to [0,1]
    X_realA_half = (X_realA_half + 1) / 2.0
    X_realB_half = (X_realB_half + 1) / 2.0
    X_fakeB_half = (X_fakeB_half + 1) / 2.0
    X_realA_half = torch.transpose(X_realA_half, 1, 3)
    X_realA_half = torch.transpose(X_realA_half, 1, 2).cpu().detach()
    # plot real source images
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(X_realA_half[i,:,:,:])
    # plot generated target image
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + n_samples + i)
        pyplot.axis('off')
        twoD_img = X_fakeB_half[:,0,:,:].cpu().detach()
        pyplot.imshow(twoD_img[i,:,:],cmap="gray")
    # plot real target image
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)
        pyplot.axis('off')
        twoD_img = X_realB_half[:,0,:,:].cpu().detach()
        pyplot.imshow(twoD_img[i,:,:],cmap="gray")

    # save plot to file
    filename1 = savedir+'/coarse_generator_plot_%06d.png' % (step+1)
    pyplot.savefig(filename1)
    pyplot.close()
    # save the coarse_generator model
    filename2 = savedir+'/coarse_generator_model_%06d.pth' % (step+1)
    torch.save(coarse_generator.state_dict(), filename2)
    print('>Saved: %s and %s' % (filename1, filename2))
    #return X_feature

def visualize_save_weight_fine(step, coarse_generator, fine_generator, dataset, n_samples=3, savedir='weights_plots'):
    if not os.path.exists(savedir):
            os.makedirs(savedir)

    # select a sample of input images
    n_patch = [1, 1, 1]
    [X_realA, X_realB], _ = generate_real_data(dataset, n_samples, n_patch)

    # Resize to half
    out_shape = (int(X_realA.shape[1] / 2), int(X_realA.shape[2] / 2))
    # print(out_shape) (256,256)
    [X_realA_half, X_realB_half] = resize(X_realA, X_realB, out_shape)

    X_realA = data_normalize_trans_tensor(X_realA)
    X_realB = data_normalize_trans_tensor(X_realB)
    X_realA_half = data_normalize_trans_tensor(X_realA_half)
    X_realB_half = data_normalize_trans_tensor(X_realB_half)
    X_realA = X_realA.to(device)
    X_realB = X_realB.to(device)
    X_realA_half = X_realA_half.to(device)
    X_realB_half = X_realB_half.to(device)

    # generate a batch of fake samples
    [X_fakeB_half, X_feature], _ = generate_fake_data_coarse(coarse_generator, X_realA_half, n_patch)

    ##########################################
    # generate a batch of fake samples
    X_fakeB, _ = generate_fake_data_fine(fine_generator, X_realA, X_feature, n_patch)

    # scale all pixels from [-1,1] to [0,1]
    X_realA = (X_realA + 1) / 2.0
    X_realB = (X_realB + 1) / 2.0
    X_fakeB = (X_fakeB + 1) / 2.0
    X_realA = torch.transpose(X_realA, 1, 3)
    X_realA = torch.transpose(X_realA, 1, 2).cpu().detach()
    # plot real source images
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(X_realA[i,:,:,:])

    # plot generated target image
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + n_samples + i)
        pyplot.axis('off')
        twoD_img = X_fakeB[:,0,:,:].cpu().detach()
        pyplot.imshow(twoD_img[i,:,:],cmap="gray")

    # plot real target image
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)
        pyplot.axis('off')
        twoD_img = X_realB[:,0,:,:].cpu().detach()
        pyplot.imshow(twoD_img[i,:,:],cmap="gray")

    # save plot to file
    filename1 = savedir+'/fine_generator_plot_%06d.png' % (step+1)
    pyplot.savefig(filename1)
    pyplot.close()
    # save the fine_generator model
    filename2 = savedir+'/fine_generator_model_%06d.pth' % (step+1)
    torch.save(fine_generator.state_dict(), filename2)
    print('>Saved: %s and %s' % (filename1, filename2))

if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--epochs', type=int, default=100)
    # parser.add_argument('--batch_size', type=int, default=4)
    # parser.add_argument('--npz_file', type=str, default='fun2angio', help='path/to/npz/file')
    # parser.add_argument('--input_dim', type=int, default=512)
    # #parser.add_argument('--datadir', type=str, required=True, help='path/to/data_directory',default='fundus2angio')
    # args = parser.parse_args()

    print(torch.cuda.is_available())
    print(torch.version.cuda)
    print(os.getcwd())

    #垃圾回收 gc.collect() 返回处理这些循环引用一共释放掉的对象个数
    gc.collect()
    start_time = time.time()
    dataset = load_real_data('fun2angio.npz') #args.npz_file+'.npz'
    print('Loaded', dataset[0].shape, dataset[1].shape)
    #print(dataset[0][790][200][200][2])
    #print(dataset[1][790][200][200][0])

    # define input shape based on the loaded dataset
    # in_size = 512
    # image_shape_coarse = (in_size // 2, in_size // 2, 3) #256 256 3
    # label_shape_coarse = (in_size // 2, in_size // 2, 1) #256 256 1
    #
    # image_shape_fine = (in_size, in_size, 3) #512 512 3
    # label_shape_fine = (in_size, in_size, 1) #512 512 1
    #
    # image_shape_xglobal = (in_size // 2, in_size // 2, 64) #256 256 64
    # ndf = 32
    # ncf = 64
    # nff = 32
    # print(image_shape_coarse,label_shape_coarse,image_shape_fine,label_shape_fine,image_shape_xglobal)

    # define discriminator models
    d1 = Discriminator(input_cf_nc=3,input_ffa_nc=1,n_downsampling=0,hw=64)  # D1 Fine
    d2 = Discriminator(input_cf_nc=3,input_ffa_nc=1,n_downsampling=1,hw=32) # D2 Fine
    d3 = Discriminator(input_cf_nc=3,input_ffa_nc=1,n_downsampling=0,hw=32) # D1 Coarse
    d4 = Discriminator(input_cf_nc=3,input_ffa_nc=1,n_downsampling=1,hw=16) # D2 Coarse

    # define generator models
    coarse_generator = Coarse_Generator(input_nc=3)

    fine_generator = Fine_Generator(input_nc=3)

    # define fundus2angio
    gan_model = Fundus2Angio_Gan(coarse_generator,fine_generator, d1, d2, d3, d4)

    # ####################################test-gan-model####################################
    # n_patch = [64, 32, 16]
    # [X_realA, X_realB], [y1, y2, y3] = generate_real_data(dataset, 4, n_patch)
    # out_shape = (int(X_realA.shape[1] / 2), int(X_realA.shape[2] / 2))
    # print(out_shape)
    # [X_realA_half, X_realB_half] = resize(X_realA, X_realB, out_shape)
    # input_cf = np.transpose(X_realA,(0,3,1,2))
    # input_cf = torch.from_numpy(input_cf).type(torch.FloatTensor) #transforms.ToTensor()
    # print(input_cf.shape)
    # input_ffa = np.transpose(X_realB,(0,3,1,2))
    # input_ffa = torch.from_numpy(input_ffa).type(torch.FloatTensor)
    # print(input_ffa.shape)
    # input_cf_half = np.transpose(X_realA_half,(0,3,1,2))
    # input_cf_half = torch.from_numpy(input_cf_half).type(torch.FloatTensor)
    # print(input_cf_half.shape)
    # input_ffa_half = np.transpose(X_realB_half,(0,3,1,2))
    # input_ffa_half = torch.from_numpy(input_ffa_half).type(torch.FloatTensor)
    # print(input_ffa_half.shape)
    # coarse_feature_out = torch.arange(256*256*64*4, dtype=torch.float).reshape(4, 64, 256, 256).type(torch.FloatTensor)
    # print(coarse_feature_out.shape)
    # coarse_output, fine_output, d1_output, d2_output, d3_output, d4_output = gan_model((input_cf, input_ffa,input_cf_half, input_ffa_half, coarse_feature_out))
    # print(coarse_output.shape, fine_output.shape, d1_output.shape, d2_output.shape, d3_output.shape, d4_output.shape)

    #unpack dataset
    n_epochs = 100 #args.epochs
    n_batch = 4 #args.batch_size

    trainA, trainB = dataset

    # calculate the number of batches per training epoch
    bat_per_epo = int(len(trainA) / n_batch)
    print(bat_per_epo)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    print(n_steps)

    # lists for storing loss, for plotting later
    d1_hist, d2_hist, d3_hist, d4_hist, d5_hist, d6_hist, d7_hist, d8_hist = list(), list(), list(), list(), list(), list(), list(), list()
    cg_hist, fg_hist, g_hist = list(), list(), list()

    # 损失函数
    loss_function = nn.MSELoss()
    # 优化器
    optimizer_d1 = torch.optim.Adam(d1.parameters(),lr = 0.0002, betas= (0.5, 0.999))
    optimizer_d2 = torch.optim.Adam(d2.parameters(),lr = 0.0002, betas= (0.5, 0.999))
    optimizer_d3 = torch.optim.Adam(d3.parameters(),lr = 0.0002, betas= (0.5, 0.999))
    optimizer_d4 = torch.optim.Adam(d4.parameters(),lr = 0.0002, betas= (0.5, 0.999))

    optimizer_cg = torch.optim.Adam(coarse_generator.parameters(),lr = 0.0002, betas= (0.5, 0.999))
    optimizer_fg = torch.optim.Adam(fine_generator.parameters(),lr = 0.0002, betas= (0.5, 0.999))
    optimizer_g = torch.optim.Adam(gan_model.parameters(),lr = 0.0002, betas= (0.5, 0.999))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    d1.to(device)  # 分配网络到指定的设备（GPU/CPU）训练
    d2.to(device)  # 分配网络到指定的设备（GPU/CPU）训练
    d3.to(device)  # 分配网络到指定的设备（GPU/CPU）训练
    d4.to(device)  # 分配网络到指定的设备（GPU/CPU）训练

    coarse_generator.to(device)  # 分配网络到指定的设备（GPU/CPU）训练
    fine_generator.to(device)  # 分配网络到指定的设备（GPU/CPU）训练
    gan_model.to(device)  # 分配网络到指定的设备（GPU/CPU）训练

    n_patch = [64, 32, 16]
    # manually enumerate epochs
    for i in tqdm.tqdm(range(n_steps)):
        step_start_time = time.time()

        freeze(coarse_generator)
        freeze(fine_generator)
        freeze(gan_model)
        for j in range(2):
            # select a batch of real samples
            [X_realA, X_realB], [y1, y2, y3] = generate_real_data(dataset, n_batch, n_patch)
            # generate a batch of fake samples for Coarse Generator
            out_shape = (int(X_realA.shape[1] / 2), int(X_realA.shape[2] / 2))
            #print(out_shape) (256,256)
            [X_realA_half, X_realB_half] = resize(X_realA, X_realB, out_shape)
            #print(X_realA)
            #print(X_realA.shape)
            #print(X_realB)
            #print(X_realB.shape)
            #print(X_realA_half)
            #print(X_realA_half.shape)
            #print(X_realB_half)
            #print(X_realB_half.shape)
            X_realA = data_normalize_trans_tensor(X_realA)
            X_realB = data_normalize_trans_tensor(X_realB)
            X_realA_half = data_normalize_trans_tensor(X_realA_half)
            X_realB_half = data_normalize_trans_tensor(X_realB_half)
            # print(X_realA)
            #print(X_realA.shape)
            # print(X_realB)
            #print(X_realB.shape)
            # print(X_realA_half)
            #print(X_realA_half.shape)
            # print(X_realB_half)
            #print(X_realB_half.shape)
            X_realA = X_realA.to(device)
            X_realB = X_realB.to(device)
            X_realA_half = X_realA_half.to(device)
            X_realB_half = X_realB_half.to(device)

            [X_fakeB_half, X_feature], [y1_coarse, y2_coarse] = generate_fake_data_coarse(coarse_generator, X_realA_half, n_patch)
            #print(X_realB_half)
            # print(X_fakeB_half.shape)
            # print(X_feature)
            # print(X_feature.shape)
            # print(y1_coarse)
            # print(y1_coarse.shape)
            # print(y2_coarse)
            # print(y2_coarse.shape)

            #generate a batch of fake samples for Fine Generator
            X_fakeB, [y1_fine, y2_fine] = generate_fake_data_fine(fine_generator, X_realA, X_feature, n_patch)

            ## FINE DISCRIMINATOR
            # update discriminator for real samples
            d_loss1 = d_train_on_batch(d1,loss_function,optimizer_d1,(X_realA.detach(), X_realB.detach()),y1)
            # update discriminator for generated samples
            d_loss2 = d_train_on_batch(d1,loss_function,optimizer_d1,(X_realA.detach(), X_fakeB.detach()),y1_fine)
            # # update discriminator for real samples
            d_loss3 = d_train_on_batch(d2,loss_function,optimizer_d2,(X_realA.detach(), X_realB.detach()),y2)
            # # update discriminator for generated samples
            d_loss4 = d_train_on_batch(d2,loss_function,optimizer_d2,(X_realA.detach(), X_fakeB.detach()),y2_fine)

            #print(d_loss1, d_loss2, d_loss3, d_loss4)

            ## COARSE DISCRIMINATOR
            # update discriminator for real samples
            d_loss5 = d_train_on_batch(d3,loss_function,optimizer_d3,(X_realA_half.detach(), X_realB_half.detach()),y2)
            # update discriminator for generated samples
            d_loss6 = d_train_on_batch(d3,loss_function,optimizer_d3,(X_realA_half.detach(), X_fakeB_half.detach()),y1_coarse)
            # update discriminator for real samples
            d_loss7 = d_train_on_batch(d4,loss_function,optimizer_d4,(X_realA_half.detach(), X_realB_half.detach()),y3)
            # update discriminator for generated samples
            d_loss8 = d_train_on_batch(d4,loss_function,optimizer_d4,(X_realA_half.detach(), X_fakeB_half.detach()),y2_coarse)

            #print(d_loss5,d_loss6,d_loss7,d_loss8)
        d1_loss_real = d_loss1
        d1_loss_fake = d_loss2

        d2_loss_real = d_loss3
        d2_loss_fake = d_loss4

        d3_loss_real = d_loss5
        d3_loss_fake = d_loss6

        d4_loss_real = d_loss7
        d4_loss_fake = d_loss8

        # turn coarse_generator trainable
        freeze(d1)
        freeze(d2)
        freeze(d3)
        freeze(d4)
        freeze(fine_generator)
        freeze(gan_model)

        # select a batch of real samples for fine generator
        [X_realA, X_realB], _ = generate_real_data(dataset, n_batch, n_patch)
        # Coarse Generator image fake and real
        out_shape = (int(X_realA.shape[1] / 2), int(X_realA.shape[2] / 2))
        # print(out_shape) (256,256)
        [X_realA_half, X_realB_half] = resize(X_realA, X_realB, out_shape)
        X_realA = data_normalize_trans_tensor(X_realA)
        X_realB = data_normalize_trans_tensor(X_realB)
        X_realA_half = data_normalize_trans_tensor(X_realA_half)
        X_realB_half = data_normalize_trans_tensor(X_realB_half)
        X_realA = X_realA.to(device)
        X_realB = X_realB.to(device)
        X_realA_half = X_realA_half.to(device)
        X_realB_half = X_realB_half.to(device)

        [X_fakeB_half, X_feature], _ = generate_fake_data_coarse(coarse_generator, X_realA_half, n_patch)

        # update the coarse_generator
        cg_loss = coarse_train_on_batch(coarse_generator, loss_function, optimizer_cg, X_realA_half.detach(), X_realB_half.detach())
        #print(cg_loss)

        # turn fine_generator trainable
        freeze(d1)
        freeze(d2)
        freeze(d3)
        freeze(d4)
        freeze(coarse_generator)
        freeze(gan_model)

        # update the fine_generator
        fg_loss = fine_train_on_batch(fine_generator, loss_function, optimizer_fg, (X_realA.detach(),X_feature.detach()), X_realB.detach())
        #print(fg_loss)

        # turn CG, FG and GAN trainable, not D1,D2 D3 and D4
        freeze(d1)
        freeze(d2)
        freeze(d3)
        freeze(d4)
        unfreeze(fine_generator)
        unfreeze(coarse_generator)

        g_loss = gan_train_on_batch(gan_model, loss_function, optimizer_g, (X_realA.detach(),X_realB.detach(),X_realA_half.detach(),X_realB_half.detach(),X_feature.detach()),(y1,y2,y2,y3))
        #print(g_loss)

        d1_hist.append(d1_loss_real)
        d2_hist.append(d1_loss_fake)
        d3_hist.append(d2_loss_real)
        d4_hist.append(d2_loss_fake)
        d5_hist.append(d3_loss_real)
        d6_hist.append(d3_loss_fake)
        d7_hist.append(d4_loss_real)
        d8_hist.append(d4_loss_fake)
        cg_hist.append(cg_loss)
        fg_hist.append(fg_loss)
        g_hist.append(g_loss)

        if (i+1) % (bat_per_epo * 1) == 0:

            visualize_save_weight_coarse(i, coarse_generator, dataset, n_samples=3)

            visualize_save_weight_fine(i, coarse_generator, fine_generator, dataset, n_samples=3)

        # end of step
        iter_end_time = time.time()
        # summarize performance
        print(
            'Step%d, Time Taken: %d s, d1_loss_real[%.3f] d1_loss_fake[%.3f] d2_loss_real[%.3f] d2_loss_fake[%.3f] d3_loss_real[%.3f] d3_loss_fake[%.3f] d4_loss_real[%.3f] d4_loss_fake[%.3f] cg_loss[%.3f] fg_loss[%.3f] g_loss[%.3f]' %
            (i + 1, iter_end_time-step_start_time,d1_loss_real, d1_loss_fake, d2_loss_real, d2_loss_fake, d3_loss_real, d3_loss_fake, d4_loss_real,
             d4_loss_fake, cg_loss, fg_loss, g_loss))
    plot_history(d1_hist, d2_hist, d3_hist, d4_hist, d5_hist, d6_hist, d7_hist, d8_hist, cg_hist, fg_hist, g_hist)
    end_time = time.time()
    time_taken = (end_time - start_time) / 3600.0
    print("Total Time Taken: ", time_taken)
