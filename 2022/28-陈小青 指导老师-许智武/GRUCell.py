import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def gru_forward(input,gru_output,h0, W1, W2,W3,b1,b2,input_size,hidden_size):
    # initial_state在GRU里是指h0，在LSTM包含h0和c0
    prev_h = h0 #初始状态
    
    batch_size_sum = 0
    
    for j in range(len(input[1])):
        batch_size = input[1][j]
        x = input[0][batch_size_sum:batch_size_sum+batch_size]
        prev_h_temp = prev_h[0:batch_size]
        
        batch_W1 = W1.unsqueeze(0).tile(batch_size, 1, 1)
        batch_W2 = W2.unsqueeze(0).tile(batch_size, 1, 1)
        batch_W3 = W3.unsqueeze(0).tile(batch_size, 1, 1)
        
        # x = input.data[:, t, :] # t时刻GRU cell的输入特征向量, 大小(batch_size, input_size)
        # print(batch_W1.size())
        # print(x.size())
        w_times_x1 = torch.bmm(batch_W1, x.unsqueeze(-1)) # [batch_size, 3*h_size, 1]
        # print(w_times_x1 .size())
        w_times_x1 = w_times_x1.squeeze(-1) #[batch_size, 3*hidden_size]

        w_times_x2 = torch.bmm(batch_W2, x.unsqueeze(-1)) # [batch_size, 3*h_size, 1]
        # print(w_times_x2 .size())
        w_times_x2 = w_times_x2.squeeze(-1) #[batch_size, 3*hidden_size]
        
        
        # 重置门和更新门
        r_t = torch.sigmoid(w_times_x1 + b1)
        z_t = torch.sigmoid(w_times_x2 + b2)
        # 候选状态
        
        temp = torch.cat(((prev_h_temp*r_t),x[:,:input_size//3],x[:,(input_size//3):(input_size//3)*2]),1).unsqueeze(-1)
        
        n_t = torch.tanh(torch.bmm(batch_W3,torch.cat(((prev_h_temp*r_t),x[:,:input_size//3],x[:,(input_size//3):(input_size//3)*2]),1).unsqueeze(-1))).squeeze(-1)
        # print(prev_h.size())
        prev_h_temp = (1-z_t) * prev_h_temp + z_t * n_t    # 增量更新得到当前时刻最新的隐含状态
        # print(prev_h.size())
        prev_h_temp = prev_h_temp/torch.norm(prev_h_temp)
        
        # prev_h[0:batch_size] = prev_h_temp
        gru_output[0][batch_size_sum:batch_size_sum+batch_size]  = prev_h_temp
        # input[0][batch_size_sum:batch_size_sum+batch_size] = prev_h

        batch_size_sum = batch_size_sum + batch_size
        
    
    return gru_output, prev_h # 整个输出状态序列，最后的隐藏状态
        

    