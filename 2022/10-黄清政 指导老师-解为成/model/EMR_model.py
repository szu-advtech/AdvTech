import torch
import torch.nn as nn
from torchsummary import summary

class EMR(nn.Module):
    def __init__(self,input_size=1,hidden_size=16):
        super(EMR, self).__init__()
        #sequence batch feature
        self.EMR_RNN = nn.RNN(input_size=input_size,hidden_size=hidden_size,bias=True)
        self.EMR_Linear = nn.Linear(in_features=hidden_size,out_features=input_size,bias=True)
        self.Sigmoid = nn.Sigmoid()

        # 初始化RNN
        # for name,param  in self.EMR_RNN.named_parameters():
        #     if name.startswith("weight"):
        #         nn.init.normal_(param)
        #     else:
        #         nn.init.zeros_(param)
        # #初始化Linear
        # nn.init.uniform_(self.EMR_Linear.weight,a=-0.1,b=0.1)
        # nn.init.constant_(self.EMR_Linear.bias,0.1)

    def forward(self,mask):
        # B, C, W, H  -> B, 1, 7, 7
        # get batch size
        b = mask.shape[0]
        # change shape
        seq_mask = mask.view(mask.shape[0],-1,1)
        first_element = seq_mask[:,:1,:]
        # print(first_element.shape)
        # print(first_element.shape)

        input_seq = torch.cat([seq_mask,first_element],dim=1)

        # print(input_seq.shape)
        # print(input_seq)

        # print(seq_mask.storage().data_ptr())
        # print(first_element.storage().data_ptr())
        # print(input_seq.storage().data_ptr())
        # cat操作后tensor数据地址不一样

        # change dim
        input_seq = input_seq.transpose(0,1)
        # seq_mask = seq_mask.transpose(0,1)

        # input  seq  batch input_size
        # output seq batch hidden_size

        output , _ = self.EMR_RNN(input_seq)

        # 取倒数第二个Hn
        Hn = output[48:49,:,:]
        Hn = Hn.transpose(0,1).view(b,-1)

        # 取整个用于预测的Hn
        output = output[1:,:,:].transpose(0,1)
        # print(output.shape)

        _mask = self.EMR_Linear(output)
        #两个数据地址不同
        _mask = self.Sigmoid(_mask)
        _mask = _mask.view(b,7,7)

        #Hn is the last hidden,output is patch sequence
        return Hn,_mask

