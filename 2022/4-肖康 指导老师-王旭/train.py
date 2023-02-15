import numpy as np
import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter
import ipdb
from data_utils import TrainDatasetFromFolder
from loss import DiscriminateLoss, rankLoss, type_strenghloss
from model import Discriminator

def setup_seed(seed):
   torch.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    # Import data from path
    CROP_SIZE=128
    NUM_EPOCHS = 50
    PATCHES=30
    LR=0.0001
    t_patch_path = '../datasets/SIQAD_4Patches/DistortedImages'
    v_patch_path = '../datasets/SIQAD_4Patches/DistortedImages'
    t_data_path = './imagedata/DMOS_SIQAD.mat'
    v_data_path = './imagedata/DMOS_SIQAD.mat'
    t_si, t_sj = 49,7
    v_si, v_sj = 49,7
    train_index = (1,2,3,4,5,6,7,8,9,12,13,14,16,17,19,20)
    val_index = (10,11,15,18)
    test_index = val_index
    train_set = TrainDatasetFromFolder(t_patch_path, crop_size=CROP_SIZE,patches=PATCHES, istrain=True, data_path=t_data_path, si=t_si, sj=t_sj, indexlist=train_index)
    val_set =   TrainDatasetFromFolder(v_patch_path, crop_size=CROP_SIZE,patches=1, istrain=False, data_path= v_data_path, si=v_si, sj=v_sj, indexlist=val_index)
    setup_seed(20)
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)
    # Instantiate a network
    netD = Discriminator()
    # loss function
    generator_criterion = DiscriminateLoss()
    generator_rank = rankLoss()
    criterion_type_strengh = type_strenghloss()
    if torch.cuda.is_available():
        netD.cuda()
        generator_criterion.cuda()
        generator_rank.cuda()
        criterion_type_strengh.cuda()
    optimizerD = optim.Adam(netD.parameters(),lr=LR, weight_decay=(0.00001))
    train_times=1
    val_times=1
    writer = SummaryWriter()
    # start to train
    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)
        running_results = {'type_loss': 0,'type_accurate': 0, 'batch_sizes': 0, 'd_loss': 0, 'mse_loss': 0, 'rank_loss': 0}
        netD.train()
        for data1, data2, data3, data4, data5, data6, data7, data8, label1, type, label2 in train_bar:
            batch_size = data1.size(0)
            running_results['batch_sizes'] += batch_size

            noise_type = Variable(type).long()
            image_labels1 = Variable(label1)
            image_labels1=np.reshape(image_labels1,[batch_size,1])
            image_labels2 = Variable(label2)
            image_labels2=np.reshape(image_labels2,[batch_size,1])

            if torch.cuda.is_available():
                image_labels1 = image_labels1.cuda()
                noise_type = noise_type.cuda()
                image_labels2 = image_labels2.cuda()

            z1 = Variable(data1)
            z2 = Variable(data2)
            z3 = Variable(data3)
            z4 = Variable(data4)
            z5 = Variable(data5)
            z6 = Variable(data6)
            z7 = Variable(data7)
            z8 = Variable(data8)
            if torch.cuda.is_available():
                z1 = z1.cuda()
                z2 = z2.cuda()
                z3 = z3.cuda()
                z4 = z4.cuda()
                z5 = z5.cuda()
                z6 = z6.cuda()
                z7 = z7.cuda()
                z8 = z8.cuda()
            netD.zero_grad()
            predict_type,predict_labels1,predict_labels2 = netD(z1,z2,z3,z4,z5,z6,z7,z8)

            type_loss, type_accurate = criterion_type_strengh(predict_type, noise_type)
            mse_loss=0.1*generator_criterion(predict_labels1,image_labels1)
            rank_loss=0.1*generator_rank(predict_labels1,predict_labels2,image_labels1,image_labels2)
            d_loss = mse_loss+0.1*rank_loss+type_loss
            d_loss.backward()
            optimizerD.step()
            writer.add_scalar('train_loss', d_loss, train_times)
            train_times=train_times+1
            running_results['type_loss'] += type_loss.item() * batch_size
            running_results['type_accurate'] += type_accurate * batch_size
            running_results['mse_loss'] += mse_loss.item()  * batch_size
            running_results['rank_loss'] += rank_loss.item()  * batch_size
            running_results['d_loss'] += d_loss.item()  * batch_size

            train_bar.set_description(desc='[%d/%d] type_accurate: %.4f Loss_type: %.4f Learningrate:%.13f mse_D: %.4f rank_D: %.4f Loss_D: %.4f' % (epoch, NUM_EPOCHS, running_results['type_accurate'] / running_results['batch_sizes'],running_results['type_loss'] / running_results['batch_sizes'], LR,running_results['mse_loss'] / running_results['batch_sizes'],running_results['rank_loss'] / running_results['batch_sizes'],running_results['d_loss'] / running_results['batch_sizes']))
            writer.add_scalar('train_all_loss', running_results['d_loss'] / running_results['batch_sizes'], epoch)


        netD.eval()
        val_bar = tqdm(val_loader)
        valing_results = {'mse': 0,  'batch_sizes': 0,'strengh_accurate': 0}
        val_images = []
        num=1
        for val_data1,val_data2,val_data3,val_data4,val_label,val_type in val_bar:
            batch_size = val_data1.size(0)
            valing_results['batch_sizes'] += batch_size
            val_z1 = Variable(val_data1)
            val_z2 = Variable(val_data2)
            val_z3 = Variable(val_data3)
            val_z4 = Variable(val_data4)

            image_vallabels = Variable(val_label)
            image_vallabels = np.reshape(image_vallabels, [batch_size, 1])
            image_valstrengh = Variable(val_type)
            if torch.cuda.is_available():
                image_vallabels = image_vallabels.cuda()
                image_valstrengh=image_valstrengh.cuda()
                val_z1 = val_z1.cuda()
                val_z2 = val_z2.cuda()
                val_z3 = val_z3.cuda()
                val_z4 = val_z4.cuda()
            torch.no_grad()
            predict_valstrengh,predict_vallabels,predict_vallabels1 = netD(val_z1,val_z2,val_z3,val_z4,val_z1,val_z2,val_z3,val_z4)

            _, pred = predict_valstrengh.max(1)
            num_correct = (pred == image_valstrengh).sum().item()
            val_strengh_accurate = num_correct / batch_size
            valing_results['strengh_accurate'] += val_strengh_accurate * batch_size
            writer.add_scalar('val_strengh_accurate', val_strengh_accurate, val_times)

            batch_mse = (torch.abs(predict_vallabels - image_vallabels) ).data
            valing_results['mse'] += batch_mse * batch_size
            writer.add_scalar('val_loss', batch_mse, val_times)
            val_times=val_times+1
            val_bar.set_description(desc='[validation] strengh: %.4f  MSE: %.4f' % ( valing_results['strengh_accurate']/valing_results['batch_sizes'], valing_results['mse']/valing_results['batch_sizes']))
            writer.add_scalar('val_all_loss', valing_results['mse']/valing_results['batch_sizes'], epoch)

        # save model parameters
        if epoch % 10 == 0 and epoch != 0:
            torch.save(netD.state_dict(), 'epochs/netD_epoch_%d.pth' % (epoch))

        if epoch % 10 == 0 and epoch != 0:
            LR=LR*0.1

    writer.close()