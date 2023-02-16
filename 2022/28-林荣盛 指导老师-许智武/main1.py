
import time
import sys
import argparse
import random
import copy
import torch
import gc
import pickle
import os
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from utils.metric import get_ner_fmeasure
from model.gazlstm import GazLSTM as SeqModel
from utils.data import Data


def data_initialization(data,test_file):

    return data

def recover_label1(pred_variable,  mask_variable, label_alphabet):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    seq_len = pred_variable.size(1)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    for idx in range(batch_size):
        pred = [label_alphabet.get_instance(int(pred_tag[idx][idy])) for idy in range(seq_len) if mask[idx][idy] != 0]
        pred_label.append(pred)

    return pred_label


def batchify_with_label1(input_batch_list, gpu, num_layer, volatile_flag=False):

    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list]
    biwords = [sent[1] for sent in input_batch_list]
    gazs = [sent[3] for sent in input_batch_list]
    # labels = [sent[4] for sent in input_batch_list]
    layer_gazs = [sent[4] for sent in input_batch_list]
    gaz_count = [sent[5] for sent in input_batch_list]
    gaz_chars = [sent[6] for sent in input_batch_list]
    gaz_mask = [sent[7] for sent in input_batch_list]
    gazchar_mask = [sent[8] for sent in input_batch_list]
    ### bert tokens
    bert_ids = [sent[9] for sent in input_batch_list]

    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    max_seq_len = word_seq_lengths.max()
    word_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len))).long()
    biword_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len))).long()
    # label_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len))).long()
    mask = autograd.Variable(torch.zeros((batch_size, max_seq_len))).byte()
    ### bert seq tensor
    bert_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len+2))).long()
    bert_mask = autograd.Variable(torch.zeros((batch_size, max_seq_len+2))).long()

    gaz_num = [len(layer_gazs[i][0][0]) for i in range(batch_size)]
    max_gaz_num = max(gaz_num)
    layer_gaz_tensor = torch.zeros(batch_size, max_seq_len, 4, max_gaz_num).long()
    gaz_count_tensor = torch.zeros(batch_size, max_seq_len, 4, max_gaz_num).float()
    gaz_len = [len(gaz_chars[i][0][0][0]) for i in range(batch_size)]
    max_gaz_len = max(gaz_len)
    gaz_chars_tensor = torch.zeros(batch_size, max_seq_len, 4, max_gaz_num, max_gaz_len).long()
    gaz_mask_tensor = torch.ones(batch_size, max_seq_len, 4, max_gaz_num).byte()
    gazchar_mask_tensor = torch.ones(batch_size, max_seq_len, 4, max_gaz_num, max_gaz_len).byte()

    # 变成tensor
    for b, (seq, bert_id, biseq,seqlen, layergaz, gazmask, gazcount, gazchar, gazchar_mask, gaznum, gazlen) in enumerate(zip(words, bert_ids, biwords, word_seq_lengths, layer_gazs, gaz_mask, gaz_count, gaz_chars, gazchar_mask, gaz_num, gaz_len)):

        word_seq_tensor[b, :seqlen] = torch.LongTensor(seq)
        biword_seq_tensor[b, :seqlen] = torch.LongTensor(biseq)
        # label_seq_tensor[b, :seqlen] = torch.LongTensor(label)
        layer_gaz_tensor[b, :seqlen, :, :gaznum] = torch.LongTensor(layergaz)
        mask[b, :seqlen] = torch.Tensor([1]*int(seqlen))
        bert_mask[b, :seqlen+2] = torch.LongTensor([1]*int(seqlen+2))
        gaz_mask_tensor[b, :seqlen, :, :gaznum] = torch.ByteTensor(gazmask)
        gaz_count_tensor[b, :seqlen, :, :gaznum] = torch.FloatTensor(gazcount)
        gaz_count_tensor[b, seqlen:] = 1
        gaz_chars_tensor[b, :seqlen, :, :gaznum, :gazlen] = torch.LongTensor(gazchar)
        gazchar_mask_tensor[b, :seqlen, :, :gaznum, :gazlen] = torch.ByteTensor(gazchar_mask)

        ##bert
        bert_seq_tensor[b, :seqlen+2] = torch.LongTensor(bert_id)


    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        biword_seq_tensor = biword_seq_tensor.cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        # label_seq_tensor = label_seq_tensor.cuda()
        layer_gaz_tensor = layer_gaz_tensor.cuda()
        gaz_chars_tensor = gaz_chars_tensor.cuda()
        gaz_mask_tensor = gaz_mask_tensor.cuda()
        gazchar_mask_tensor = gazchar_mask_tensor.cuda()
        gaz_count_tensor = gaz_count_tensor.cuda()
        mask = mask.cuda()
        bert_seq_tensor = bert_seq_tensor.cuda()
        bert_mask = bert_mask.cuda()

    # print(bert_seq_tensor.type())
    return gazs, word_seq_tensor, biword_seq_tensor, word_seq_lengths, layer_gaz_tensor, gaz_count_tensor,gaz_chars_tensor, gaz_mask_tensor, gazchar_mask_tensor, mask, bert_seq_tensor, bert_mask

def predict_label(data, model):

    instances = data.raw_Ids

    right_token = 0
    whole_token = 0
    pred_results = []
    gold_results = []
    ## set model in eval model
    model.eval()
    batch_size = 1
    train_num = len(instances)
    total_batch = train_num//batch_size+1
    gazes = []
    for batch_id in range(total_batch):
        with torch.no_grad():
            start = batch_id*batch_size
            end = (batch_id+1)*batch_size
            if end >train_num:
                end =  train_num
            instance = instances[start:end]
            if not instance:
                continue
            gaz_list,batch_word, batch_biword, batch_wordlen, layer_gaz, gaz_count, gaz_chars, gaz_mask, gazchar_mask, mask, batch_bert, bert_mask  = batchify_with_label1(instance, data.HP_gpu, data.HP_num_layer, True)
            tag_seq, gaz_match = model(gaz_list,batch_word, batch_biword, batch_wordlen, layer_gaz, gaz_count,gaz_chars, gaz_mask, gazchar_mask, mask, batch_bert, bert_mask)

            gaz_list = [data.gaz_alphabet.get_instance(id) for batchlist in gaz_match if len(batchlist)>0 for id in batchlist ]
            gazes.append( gaz_list)

            pred_label = recover_label1(tag_seq, mask, data.label_alphabet)

            pred_results += pred_label
    return pred_results, gazes


if __name__=='__main__':

    # instance包括
    # 词（字）的id, bi词的id, 字的id, gaz_id, label的id, gazs , gazs_count, gaz_char_Id,...
    # gaz_Ids每一行由两部分组成，当前字匹配的所有词的id，和匹配的词的长度（每个字一行）
    #  gazs的确记录的是匹配的词的id    高勇2
    # gazs_count的确记录的是词的频率   高勇的频率是1
    # 初始化data

    # gaz_list, batch_word, batch_biword, batch_wordlen, batch_label, layer_gaz,gaz_count,
    # gaz_chars, gaz_mask, gazchar_mask, mask, batch_bert, bert_mask = batchify_with_label(
    #     instance, data.HP_gpu, data.HP_num_layer, True)

    # model(gaz_list, batch_word, batch_biword, batch_wordlen, layer_gaz, gaz_count, gaz_chars,
    #       gaz_mask, gazchar_mask,mask, batch_bert, bert_mask)
    model_dir='./save_model/model_GAZLSTM'
    # text='屠呦呦，女，1930年12月30日出生于浙江宁波 [78]  ，汉族，中共党员，药学家。1951年考入北京大学医学院药学系生药专业。 [1-2]  1955年毕业于北京医学院（今北京大学医学部）。毕业后接受中医培训两年半，并一直在中国中医研究院（2005年更名为中国中医科学院）工作，期间晋升为硕士生导师、博士生导师。现为中国中医科学院首席科学家， [3-5]   终身研究员兼首席研究员 [6]  ，青蒿素研究开发中心主任，博士生导师，共和国勋章获得者。 [7] '

    data_dir='./data/save.dset'
    with open(data_dir, 'rb') as fp:
        data = pickle.load(fp)
    # data.HP_use_gaz = args.use_gaz
    data.use_bigram = False
    data.HP_use_char = False
    data.use_bert = False
    data.HP_gpu = False
    data.HP_use_count = True
    data.model_type = 'lstm'
    model = SeqModel(data)
    model.load_state_dict(torch.load(model_dir,map_location='cpu'))

    text = '屠呦呦，女，1930年12月30日出生于浙江宁波  ，汉族，中共党员，药学家。' \
           '1951年考入北京大学医学院药学系生药专业。 1955年毕业于北京医学院（今北京大学医学部）。' \
           '毕业后接受中医培训两年半，并一直在中国中医研究院（2005年更名为中国中医科学院）工作，' \
           '期间晋升为硕士生导师、博士生导师。现为中国中医科学院首席科学家，  ' \
           '终身研究员兼首席研究员，青蒿素研究开发中心主任，博士生导师，共和国勋章获得者。'

    text = list(text)
    text = [s for s in text if s.strip() != '']
    with open('./data/test1.char', 'w', encoding='utf8') as fp:
        for s in text:
            fp.write(s + '\n')
        fp.write(' ')

    test_file='./data/test1.char'

    data.build_alphabet1(test_file)
    data.build_gaz_alphabet1(test_file, count=True)
    data.fix_alphabet()
    data.generate_instance_with_gaz(test_file, 'raw')

    result,gazs=predict_label(data,model)
    print(result)

    with open('./data/test1.char', 'w', encoding='utf8') as fp:
        for i in range(len(text)):
            l = text[i].strip()
            r = result[0][i]
            fp.write(l + ',' + r + '\n')




