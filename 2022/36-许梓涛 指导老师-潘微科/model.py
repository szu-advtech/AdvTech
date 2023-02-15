import numpy as np
import torch


class PointWiseFeedForward(torch.nn.Module):   
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs   
        return outputs


class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.user_emb = torch.nn.Embedding(self.user_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()

        self.cross_attention_layernorms = torch.nn.ModuleList()
        self.cross_attention_layers = torch.nn.ModuleList()

        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.last_cross_layernorm = torch.nn.LayerNorm(3*args.hidden_units, eps=1e-8)
        self.last_user_cross_layernorm = torch.nn.LayerNorm(2*args.hidden_units, eps=1e-8)
        self.cross_forward_layernorms = torch.nn.ModuleList()
        self.cross_forward_layers = torch.nn.ModuleList()
        
        self.attention_layernorms2 = torch.nn.ModuleList()

        self.forward_layernorms2 = torch.nn.ModuleList()
        self.forward_layers2 = torch.nn.ModuleList()
        
        self.gate_layernorms = torch.nn.ModuleList()
        self.gate_layers = torch.nn.ModuleList()
        
        for _ in range(args.num_blocks):    
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)  # attention
            self.attention_layers.append(new_attn_layer)


            new_cross_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.cross_attention_layernorms.append(new_cross_attn_layernorm)

            new_attn_layernorm2 = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms2.append(new_attn_layernorm2)
            
            new_cross_attn_layer = torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)  # attention
            self.cross_attention_layers.append(new_cross_attn_layer)


            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)   
            self.forward_layers.append(new_fwd_layer)
            
            new_fwd_layernorm2 = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms2.append(new_fwd_layernorm2)

            new_fwd_layer2 = PointWiseFeedForward(args.hidden_units, args.dropout_rate)   
            self.forward_layers2.append(new_fwd_layer2)
            
            new_cross_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.cross_forward_layernorms.append(new_cross_fwd_layernorm)

            new_cross_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)   
            self.cross_forward_layers.append(new_cross_fwd_layer)
            
        self.dropout = torch.nn.Dropout(p=args.dropout_rate)   
        #self.gating = torch.nn.Linear(2*args.hidden_units, 2)
        self.gating = torch.nn.Linear(3*args.hidden_units, args.hidden_units)
        self.gating2 = torch.nn.Linear(2*args.hidden_units, args.hidden_units)
        self.user1_attention_layernorms = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        self.user1_attention_layers = torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)  # attention
                                                            
        self.user2_attention_layernorms = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        self.user2_attention_layers = torch.nn.MultiheadAttention(args.hidden_units,
                                                            args.num_heads,
                                                            args.dropout_rate)  # attention
                              
        self.user_forward_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        self.user_forward_layers = PointWiseFeedForward(args.hidden_units, args.dropout_rate)   
                                  
        self.user_last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def log2user(self, user_ids, log_seqs1, log_seqs2, mask2):
        
        seqs1 = self.item_emb(torch.LongTensor(log_seqs1).to(self.dev))   
        seqs1 *= self.item_emb.embedding_dim ** 0.5  
        seqs1 = self.emb_dropout(seqs1)
        timeline_mask1 = torch.BoolTensor(log_seqs1 == 0).to(self.dev)
        seqs1 *= ~timeline_mask1.unsqueeze(-1) # broadcast in last dim

        tl1 = seqs1.shape[1] # time dim len for enforce causality
        attention_mask1 = ~torch.tril(torch.ones((tl1, tl1), dtype=torch.bool, device=self.dev))
        
        user1 = self.user_emb(torch.LongTensor(user_ids).to(self.dev))   
        user1 *= self.user_emb.embedding_dim ** 0.5  
        user1 = self.emb_dropout(user1)
        user1 = user1.unsqueeze(1)
        user1 = user1.expand_as(seqs1)
        
        seqs1 = torch.transpose(seqs1, 0, 1)  
        user1 = torch.transpose(user1, 0, 1)  
        Q = self.user1_attention_layernorms(user1) 
        mha_outputs1, _ = self.user1_attention_layers(Q, seqs1, seqs1, attn_mask=attention_mask1) # len,batch,hidden
        user1 = mha_outputs1
        # seqs2 = Q2 + mha_outputs2
        user1 = torch.transpose(user1, 0, 1)
        
        #log_feats = self.user_last_layernorm(user)  # (U, T, C) -> (U, -1, C) # LN (128,50,100)(batch, len, hidden)

        return user1


    def log2feats(self, user_ids, log_seqs1, log_seqs2, mask):
        seqs1 = self.item_emb(torch.LongTensor(log_seqs1).to(self.dev))  
        seqs1 *= self.item_emb.embedding_dim ** 0.5  
        positions1 = np.tile(np.array(range(log_seqs1.shape[1])), [log_seqs1.shape[0], 1])
        seqs1 += self.pos_emb(torch.LongTensor(positions1).to(self.dev))  
        seqs1 = self.emb_dropout(seqs1)

        timeline_mask1 = torch.BoolTensor(log_seqs1 == 0).to(self.dev)
        seqs1 *= ~timeline_mask1.unsqueeze(-1) # broadcast in last dim

        tl1 = seqs1.shape[1] # time dim len for enforce causality
        attention_mask1 = ~torch.tril(torch.ones((tl1, tl1), dtype=torch.bool, device=self.dev))

        seqs2 = self.item_emb(torch.LongTensor(log_seqs2).to(self.dev))   
        seqs2 *= self.item_emb.embedding_dim ** 0.5  
        positions2 = np.tile(np.array(range(log_seqs2.shape[1])), [log_seqs2.shape[0], 1])
        seqs2 += self.pos_emb(torch.LongTensor(positions2).to(self.dev))  
        seqs2 = self.emb_dropout(seqs2)

        timeline_mask2 = torch.BoolTensor(log_seqs2 == 0).to(self.dev)
        seqs2 *= ~timeline_mask2.unsqueeze(-1) # broadcast in last dim

        tl2 = seqs2.shape[1] # time dim len for enforce causality
        batch_size = seqs2.shape[0]
        # attention_mask2 = ~torch.tril(torch.ones((tl2, tl2), dtype=torch.bool, device=self.dev))
        attention_mask2 = torch.ones((batch_size, tl2, tl2), dtype=torch.bool, device=self.dev)
        #print(attention_mask2)
       
        att_seq1 = seqs1
        att_seq2 = seqs1
        seqs_2 = seqs2
        for i in range(len(self.attention_layers)):  
            att_seq1 = torch.transpose(att_seq1, 0, 1)  

            Q = self.attention_layernorms[i](att_seq1)  
            mha_outputs1, _ = self.attention_layers[i](Q, att_seq1, att_seq1,  
                                                       attn_mask=attention_mask1) # len,batch,hidden
            att_seq1 = Q + mha_outputs1
            # seqs2 = Q2 + mha_outputs2
            att_seq1 = torch.transpose(att_seq1, 0, 1)
            
            att_seq1 = self.forward_layernorms[i](att_seq1)  # LN
            att_seq1 = self.forward_layers[i](att_seq1)  
            att_seq1 *= ~timeline_mask1.unsqueeze(-1)

        
        for i in range(len(self.cross_attention_layers)):  
            att_seq2 = torch.transpose(att_seq2, 0, 1)  
            seqs_2 = torch.transpose(seqs_2, 0, 1)  

            Q = self.attention_layernorms2[i](att_seq2)  
            Q2 = self.cross_attention_layernorms[i](seqs_2)
            mha_outputs2, _ = self.cross_attention_layers[i](Q, seqs_2, seqs_2,  # cross_attention Q,K,V
                                                             attn_mask=attention_mask2)
            att_seq2 = Q + mha_outputs2
            
            seqs_2 = torch.transpose(seqs_2, 0, 1)  
            att_seq2 = torch.transpose(att_seq2, 0, 1)
            
            att_seq2 = self.forward_layernorms2[i](att_seq2)  # LN
            att_seq2 = self.forward_layers2[i](att_seq2)  
            att_seq2 *= ~timeline_mask1.unsqueeze(-1)

        log_feats = self.last_layernorm(att_seq1)  # (U, T, C) -> (U, -1, C) # LN (128,50,100)(batch, len, hidden)

        return log_feats
        
        
    def forward(self, user_ids, log_seqs, log_seqs2, pos_seqs, neg_seqs, mask): # for training
        log_feats = self.log2feats(user_ids, log_seqs, log_seqs2, mask) # user_ids hasn't been used yet    
        
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)     
        neg_logits = (log_feats * neg_embs).sum(dim=-1)     

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, log_seqs2, item_indices, mask): # for inference
        log_feats = self.log2feats(user_ids, log_seqs, log_seqs2, mask) # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste
               
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)
