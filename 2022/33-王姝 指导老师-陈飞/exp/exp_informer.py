from base64 import encode
import enum
from pyexpat import model
from tkinter.tix import Tree
from data.data_loader import Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack

from utils.tools import EarlyStopping, adjust_learning_rate,compute_entropy
from utils.metrics import metric

import numpy as np

import torch
import torch.nn as nn
import torchsummary
from torch import optim
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

from plot_history import plot_figure


import os
import time

import warnings
warnings.filterwarnings('ignore')
class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer,self).__init__(args)

    def _build_model(self):
        
        model=Informer(
            self.args.enc_in,
            self.args.dec_in,
            self.args.c_out,
            self.args.seq_len,
            self.args.label_len,
            self.args.pred_len,
            self.args.factor,
            self.args.d_model,
            self.args.n_heads,
            self.args.e_layers,
            self.args.d_layers,
            self.args.d_ff,
            self.args.dropout,
            self.args.attn,
            self.args.embed,
            self.args.freq,
            self.args.activation,
            self.args.output_attention,
            self.args.distil,
            self.args.mix,
            self.device
        ).float()
       # print(model)
        return model
    
    def _get_data(self,flag):
        args=self.args
        Data=Dataset_Custom
        timeenc= 0 if args.embed!='timeF' else 1
        
        if flag=='pred':
            shuffle_flag=False; drop_last=False; batch_size=1; freq=args.detail_freq
            Data=Dataset_Pred
        elif flag=='test':
            shuffle_flag=False; drop_last=True; batch_size=args.batch_size; freq=args.freq
        else:
            shuffle_flag=True; drop_last=True; batch_size=args.batch_size; freq=args.freq

        data_set=Data(
            root_path=args.root_path,
            size=[args.seq_len,args.label_len,args.pred_len],
            flag=flag,
            features=args.features,
            data_path=args.data_path,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        print(flag,len(data_set))
        data_loader=DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last
        )

        return data_set,data_loader

    def _select_optimizer(self):
        model_optim=optim.Adam(self.model.parameters(),lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion=nn.MSELoss()
        return criterion

    def vali(self,vali_data,vali_loader,criterion):
        self.model.eval()
        total_loss=[]
        for i,(batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(vali_loader):
            pred,true=self._process_one_batch(
                vali_data,batch_x,batch_y,batch_x_mark,batch_y_mark
            )
            loss=criterion(pred.detach().cpu(),true.detach().cpu())
            total_loss.append(loss)
        total_loss=np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self,setting):
        train_data,train_loader=self._get_data(flag='train')
        vali_data,vali_loader=self._get_data(flag='val')
        test_data,test_loader=self._get_data(flag='test')

        path=os.path.join(self.args.checkpoints,setting)
        path2 = f"results/{setting}"
        if not os.path.exists(path):
            os.makedirs(path)

        time_now=time.time()
        train_steps=len(train_loader)
        early_stopping=EarlyStopping(patience=self.args.patience,verbose=True)

        writer=SummaryWriter('./train_log',flush_secs=30)
        model_optim=self._select_optimizer()
        criterion=self._select_criterion()


        history = {
            'scaled': { 'train': [], 'vali': [], 'test': [] },
            'learning_rate': []
        }



        print(self.args.train_epochs)
        entropys=[0]*self.args.train_epochs
        for epoch in range(self.args.train_epochs):
            iter_count=0
            train_loss=[]
            self.model.train()
            epoch_time=time.time()

            for i,(batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_loader):
                iter_count+=1
                model_optim.zero_grad()
                pred,true=self._process_one_batch(train_data,batch_x,batch_y,batch_x_mark,batch_y_mark)
                loss=criterion(pred,true)
                train_loss.append(loss.item())
                writer.add_scalar('loss',loss,epoch*len(train_loader)+iter_count)

                #compute entropy
                
                first_layer_weight=True
                for name in self.model.state_dict():
                    if first_layer_weight:
                        weight=self.model.state_dict()[name].reshape(-1)
                        first_layer_weight=False 
                    else:
                        weight=torch.cat((weight,self.model.state_dict()[name].reshape(-1)),dim=0)
                
                entropy=compute_entropy(weight,100)
                writer.add_scalar('entropy',entropy,epoch*len(train_loader)+iter_count)
                entropys[epoch]+=entropy

                if (i+1)%100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed=(time.time()-time_now)/iter_count
                    left_time=speed*((self.args.train_epochs-epoch)*train_steps-i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count=0
                    time_now=time.time()
                
                loss.backward()
                model_optim.step()
            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss=np.average(train_loss)
            vali_loss=self.vali(vali_data,vali_loader,criterion)
            test_loss=self.vali(test_data,test_loader,criterion)
            
            history['scaled']['train'].append(train_loss)
            history['scaled']['vali'].append(vali_loss)
            history['scaled']['test'].append(test_loss)


            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path2)
            if early_stopping.early_stop:
                print("Early stopping")
                break
           
            learning_rate = adjust_learning_rate(model_optim, epoch+1, self.args)
            history['learning_rate'].append(learning_rate)
           
            plot_figure(path2, history, self.args)

        entropys=torch.Tensor(entropys)
        entropys=entropys/len(train_loader)

      #  print(entropys)
    

    def test(self,setting):
        test_data,test_loader=self._get_data('test')
        self.model.eval()
        
        results=[]   #记录反归一化后的预测收盘价
        preds=[]
        trues=[]

        for i,(batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(test_loader):
            pred,true=self._process_one_batch(test_data,batch_x,batch_y,batch_x_mark,batch_y_mark)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())
    
        preds=np.array(preds)
        trues=np.array(trues)
        print("test shape",preds.shape,trues.shape)
        preds=preds.reshape(-1,preds.shape[-2],preds.shape[-1])
        trues=trues.reshape(-1,trues.shape[-2],trues.shape[-1])
        print("test shape",preds.shape,trues.shape)

        folder_path='./results/'+setting+'/'
        mae,mse,rmse,mape,mspe=metric(preds,trues)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path+'pred.npy', preds)
        np.save(folder_path+'true.npy', trues)


        return





    def predict(self,setting,load=False):
        pass
        

    def _process_one_batch(self,dataset_object,batch_x,batch_y,batch_x_mark,batch_y_mark):
        batch_x=batch_x.float().to(self.device)
        batch_y=batch_y.float().to(self.device)

        batch_x_mark=batch_x_mark.float().to(self.device)
        batch_y_mark=batch_y_mark.float().to(self.device)

        if self.args.padding==0:
            dec_inp=torch.zeros([batch_y.shape[0],self.args.pred_len,batch_y.shape[-1]]).float().to(self.device)
        else:
            dec_inp=torch.ones([batch_y.shape[0],self.args.pred_len,batch_y.shape[-1]]).float().to(self.device)

        dec_inp=torch.cat([batch_y[:,:self.args.label_len,:],dec_inp],dim=1).float().to(self.device)

        if self.args.output_attention:
            outputs,attn=self.model(batch_x,batch_x_mark,dec_inp,batch_y_mark)
        else:
            outputs=self.model(batch_x,batch_x_mark,dec_inp,batch_y_mark)[0]
        
        if self.args.inverse:
            outputs=dataset_object.inverse_transform(outputs)
       # f_dim=-1 if self.args.features=='MS' else 0
        #batch_y=batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
        batch_y=batch_y[:,-self.args.pred_len:,[0]].to(self.device)
        return outputs,batch_y