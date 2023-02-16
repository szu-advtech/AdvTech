# -*- coding: utf-8 -*-

import sys
import numpy as np
import re
from utils.alphabet import Alphabet
from transformers.tokenization_bert import BertTokenizer
NULLKEY = "-null-"

def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word


def read_instance_with_gaz(num_layer, input_file, gaz, word_alphabet, biword_alphabet, biword_count, char_alphabet, gaz_alphabet, gaz_count, gaz_split, label_alphabet, number_normalized, max_sent_length, char_padding_size=-1, char_padding_symbol = '</pad>'):

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)

    in_lines = open(input_file,'r',encoding="utf-8").readlines()
    instence_texts = []
    instence_Ids = []
    words = []
    biwords = []
    chars = []
    labels = []
    word_Ids = []
    biword_Ids = []
    char_Ids = []
    label_Ids = []
    for idx in range(len(in_lines)):
        line = in_lines[idx]
        if len(line) > 2:
            pairs = line.strip().split()
            word = pairs[0]
            if number_normalized:
                word = normalize_word(word)
            label = pairs[-1]
            if idx < len(in_lines) -1 and len(in_lines[idx+1]) > 2:
                biword = word + in_lines[idx+1].strip().split()[0]
            else:
                biword = word + NULLKEY
            biwords.append(biword)
            words.append(word)
            labels.append(label)
            word_Ids.append(word_alphabet.get_index(word))
            biword_index = biword_alphabet.get_index(biword)
            biword_Ids.append(biword_index)
            label_Ids.append(label_alphabet.get_index(label))
            char_list = []
            char_Id = []
            for char in word:
                char_list.append(char)
            if char_padding_size > 0:
                char_number = len(char_list)
                if char_number < char_padding_size:
                    char_list = char_list + [char_padding_symbol]*(char_padding_size-char_number)
                assert(len(char_list) == char_padding_size)
            else:
                ### not padding
                pass
            for char in char_list:
                char_Id.append(char_alphabet.get_index(char))
            chars.append(char_list)
            char_Ids.append(char_Id)

        else:
            if ((max_sent_length < 0) or (len(words) < max_sent_length)) and (len(words)>0):
                gaz_Ids = []
                layergazmasks = []
                gazchar_masks = []
                w_length = len(words)

                # 根据句子中的字创建BMES四个数组用来装对应的匹配的词？
                gazs = [ [[] for i in range(4)] for _ in range(w_length)]  # gazs:[c1,c2,...,cn]  ci:[B,M,E,S]  B/M/E/S :[w_id1,w_id2,...]  None:0
                # 对四个数组中匹配的词的频率进行记录
                gazs_count = [ [[] for i in range(4)] for _ in range(w_length)]
                # 对四个数组里面的每个词语中的字记录id
                gaz_char_Id = [ [[] for i in range(4)] for _ in range(w_length)]  ## gazs:[c1,c2,...,cn]  ci:[B,M,E,S]  B/M/E/S :[[w1c1,w1c2,...],[],...]

                max_gazlist = 0
                max_gazcharlen = 0
                for idx in range(w_length):

                    matched_list = gaz.enumerateMatchList(words[idx:])
                    matched_length = [len(a) for a in matched_list]
                    matched_Id  = [gaz_alphabet.get_index(entity) for entity in matched_list]

                    if matched_length:
                        max_gazcharlen = max(max(matched_length),max_gazcharlen)


                    for w in range(len(matched_Id)):
                        gaz_chars = []
                        g = matched_list[w]
                        for c in g:
                            gaz_chars.append(word_alphabet.get_index(c))

                        if matched_length[w] == 1:  ## Single
                            # 高 3
                            gazs[idx][3].append(matched_Id[w])
                            # 单字直接给1
                            gazs_count[idx][3].append(1)
                            gaz_char_Id[idx][3].append(gaz_chars)
                        else:
                            # gazs的确记录的是匹配的词的id    高勇2
                            gazs[idx][0].append(matched_Id[w])   ## Begin
                            # gazscount的确记录的是词的频率   高勇的频率是1
                            gazs_count[idx][0].append(gaz_count[matched_Id[w]])
                            # gaz_char_id记录的是匹配的词里的字符的id
                            gaz_char_Id[idx][0].append(gaz_chars)
                            wlen = matched_length[w]
                            gazs[idx+wlen-1][2].append(matched_Id[w])  ## End
                            gazs_count[idx+wlen-1][2].append(gaz_count[matched_Id[w]])
                            gaz_char_Id[idx+wlen-1][2].append(gaz_chars)
                            for l in range(wlen-2):
                                gazs[idx+l+1][1].append(matched_Id[w])  ## Middle
                                gazs_count[idx+l+1][1].append(gaz_count[matched_Id[w]])
                                gaz_char_Id[idx+l+1][1].append(gaz_chars)


                    # 对空的BMES数组进行补充一个0，后续在根据最长数组长度进行补充
                    for label in range(4):
                        if not gazs[idx][label]:
                            gazs[idx][label].append(0)
                            gazs_count[idx][label].append(1)
                            gaz_char_Id[idx][label].append([0])

                        max_gazlist = max(len(gazs[idx][label]),max_gazlist)

                    matched_Id  = [gaz_alphabet.get_index(entity) for entity in matched_list]  #词号
                    if matched_Id:
                        # 每个字一行？
                        # gaz_Ids每一行由两部分组成，当前字匹配的所有词的id，和匹配的词的长度
                        gaz_Ids.append([matched_Id, matched_length])
                    else:
                        gaz_Ids.append([])

                ## batch_size = 1

                # 以下都是在对上述的所有数组生成对应的mask数组
                for idx in range(w_length):
                    gazmask = []
                    gazcharmask = []

                    for label in range(4):
                        label_len = len(gazs[idx][label])
                        count_set = set(gazs_count[idx][label])
                        if len(count_set) == 1 and 0 in count_set:
                            gazs_count[idx][label] = [1]*label_len

                        # 生成mask
                        mask = label_len*[0]
                        mask += (max_gazlist-label_len)*[1]

                        # 根据最长的BMSE数组的长度对每一个数组进行补充padding
                        gazs[idx][label] += (max_gazlist-label_len)*[0]  ## padding
                        gazs_count[idx][label] += (max_gazlist-label_len)*[0]  ## padding

                        char_mask = []
                        for g in range(len(gaz_char_Id[idx][label])):
                            glen = len(gaz_char_Id[idx][label][g])
                            charmask = glen*[0]
                            charmask += (max_gazcharlen-glen) * [1]
                            char_mask.append(charmask)
                            gaz_char_Id[idx][label][g] += (max_gazcharlen-glen) * [0]
                        gaz_char_Id[idx][label] += (max_gazlist-label_len)*[[0 for i in range(max_gazcharlen)]]
                        char_mask += (max_gazlist-label_len)*[[1 for i in range(max_gazcharlen)]]

                        gazmask.append(mask)
                        gazcharmask.append(char_mask)
                    layergazmasks.append(gazmask)
                    gazchar_masks.append(gazcharmask)

                texts = ['[CLS]'] + words + ['[SEP]']
                bert_text_ids = tokenizer.convert_tokens_to_ids(texts)

                # words是词（文本）向量 biwords是2个字（文本）向量  chars是词里的字(文本)二维向量  gazs是BMES数组四个数组（数组里是代表词的编号，并根据最长长度补充好了0）
                # 则instance_texts中一行代表一句话，话里的信息由5个数组保存
                instence_texts.append([words, biwords, chars, gazs, labels])
                #  word_Ids是词对应的id数组，biword_Ids是双字对应的id数组，char_Ids则是词里的字对应的id数组（二维）
                #  gazs的确记录的是匹配的词的id    高勇2
                # gazs_count的确记录的是词的频率   高勇的频率是1
                # gaz_char_id记录的是匹配的词里的字符的id
                # gaz_Ids每一行由两部分组成，当前字匹配的所有词的id，和匹配的词的长度（每个字一行）
                instence_Ids.append([word_Ids, biword_Ids, char_Ids, gaz_Ids, label_Ids, gazs, gazs_count, gaz_char_Id, layergazmasks,gazchar_masks, bert_text_ids])

            words = []
            biwords = []
            chars = []
            labels = []
            word_Ids = []
            biword_Ids = []
            char_Ids = []
            label_Ids = []

    return instence_texts, instence_Ids


def build_pretrain_embedding(embedding_path, word_alphabet, embedd_dim=100, norm=True):    
    embedd_dict = dict()
    if embedding_path != None:
        # 加载gaz词典里的所有embedding
        embedd_dict, embedd_dim = load_pretrain_emb(embedding_path)

    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.empty([word_alphabet.size(), embedd_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0
    pretrain_emb[0,:] = np.random.uniform(-scale, scale, [1, embedd_dim])
    for word, index in word_alphabet.instance2index.items():
        if word in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word])
            else:
                pretrain_emb[index,:] = embedd_dict[word]
            perfect_match += 1
        elif word.lower() in embedd_dict:
            if norm:
                pretrain_emb[index,:] = norm2one(embedd_dict[word.lower()])
            else:
                pretrain_emb[index,:] = embedd_dict[word.lower()]
            case_match += 1
        else:
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1
    pretrained_size = len(embedd_dict)
    print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s"%(pretrained_size, perfect_match, case_match, not_match, (not_match+0.)/word_alphabet.size()))
    return pretrain_emb, embedd_dim


def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec/root_sum_square

def load_pretrain_emb(embedding_path):
    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path, 'r',encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            else:
                assert (embedd_dim + 1 == len(tokens))
            embedd = np.empty([1, embedd_dim])
            embedd[:] = tokens[1:]
            embedd_dict[tokens[0]] = embedd
    return embedd_dict, embedd_dim

def read_instance_with_gaz1(num_layer, input_file, gaz, word_alphabet, biword_alphabet, biword_count, char_alphabet, gaz_alphabet, gaz_count, gaz_split, label_alphabet, number_normalized, max_sent_length, char_padding_size=-1, char_padding_symbol = '</pad>'):

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)

    in_lines = open(input_file,'r',encoding="utf-8").readlines()
    instence_texts = []
    instence_Ids = []
    words = []
    biwords = []
    chars = []
    # labels = []
    word_Ids = []
    biword_Ids = []
    char_Ids = []
    # label_Ids = []
    for idx in range(len(in_lines)):
        line = in_lines[idx].strip()
        if len(line) >= 1:
            word = line.strip()
            if number_normalized:
                word = normalize_word(word)
            if idx < len(in_lines) -1 and len(in_lines[idx+1]) == 2:
                biword = word + in_lines[idx+1].strip()
            else:
                biword = word + NULLKEY
            biwords.append(biword)
            words.append(word)
            # labels.append(label)
            word_Ids.append(word_alphabet.get_index(word))
            biword_index = biword_alphabet.get_index(biword)
            biword_Ids.append(biword_index)
            # label_Ids.append(label_alphabet.get_index(label))
            char_list = []
            char_Id = []
            for char in word:
                char_list.append(char)
            if char_padding_size > 0:
                char_number = len(char_list)
                if char_number < char_padding_size:
                    char_list = char_list + [char_padding_symbol]*(char_padding_size-char_number)
                assert(len(char_list) == char_padding_size)
            else:
                ### not padding
                pass
            for char in char_list:
                char_Id.append(char_alphabet.get_index(char))
            chars.append(char_list)
            char_Ids.append(char_Id)

        else:
            if ((max_sent_length < 0) or (len(words) < max_sent_length)) and (len(words)>0):
                gaz_Ids = []
                layergazmasks = []
                gazchar_masks = []
                w_length = len(words)

                # 根据句子中的字创建BMES四个数组用来装对应的匹配的词？
                gazs = [ [[] for i in range(4)] for _ in range(w_length)]  # gazs:[c1,c2,...,cn]  ci:[B,M,E,S]  B/M/E/S :[w_id1,w_id2,...]  None:0
                # 对四个数组中匹配的词的频率进行记录
                gazs_count = [ [[] for i in range(4)] for _ in range(w_length)]
                # 对四个数组里面的每个词语中的字记录id
                gaz_char_Id = [ [[] for i in range(4)] for _ in range(w_length)]  ## gazs:[c1,c2,...,cn]  ci:[B,M,E,S]  B/M/E/S :[[w1c1,w1c2,...],[],...]

                max_gazlist = 0
                max_gazcharlen = 0
                for idx in range(w_length):

                    matched_list = gaz.enumerateMatchList(words[idx:])
                    matched_length = [len(a) for a in matched_list]
                    matched_Id  = [gaz_alphabet.get_index(entity) for entity in matched_list]

                    if matched_length:
                        max_gazcharlen = max(max(matched_length),max_gazcharlen)


                    for w in range(len(matched_Id)):
                        gaz_chars = []
                        g = matched_list[w]
                        for c in g:
                            gaz_chars.append(word_alphabet.get_index(c))

                        if matched_length[w] == 1:  ## Single
                            # 高 3
                            gazs[idx][3].append(matched_Id[w])
                            # 单字直接给1
                            gazs_count[idx][3].append(1)
                            gaz_char_Id[idx][3].append(gaz_chars)
                        else:
                            # gazs的确记录的是匹配的词的id    高勇2
                            gazs[idx][0].append(matched_Id[w])   ## Begin
                            # gazscount的确记录的是词的频率   高勇的频率是1
                            gazs_count[idx][0].append(gaz_count[matched_Id[w]])
                            # gaz_char_id记录的是匹配的词里的字符的id
                            gaz_char_Id[idx][0].append(gaz_chars)
                            wlen = matched_length[w]
                            gazs[idx+wlen-1][2].append(matched_Id[w])  ## End
                            gazs_count[idx+wlen-1][2].append(gaz_count[matched_Id[w]])
                            gaz_char_Id[idx+wlen-1][2].append(gaz_chars)
                            for l in range(wlen-2):
                                gazs[idx+l+1][1].append(matched_Id[w])  ## Middle
                                gazs_count[idx+l+1][1].append(gaz_count[matched_Id[w]])
                                gaz_char_Id[idx+l+1][1].append(gaz_chars)


                    # 对空的BMES数组进行补充一个0，后续在根据最长数组长度进行补充
                    for label in range(4):
                        if not gazs[idx][label]:
                            gazs[idx][label].append(0)
                            gazs_count[idx][label].append(1)
                            gaz_char_Id[idx][label].append([0])

                        max_gazlist = max(len(gazs[idx][label]),max_gazlist)

                    matched_Id  = [gaz_alphabet.get_index(entity) for entity in matched_list]  #词号
                    if matched_Id:
                        # 每个字一行？
                        # gaz_Ids每一行由两部分组成，当前字匹配的所有词的id，和匹配的词的长度
                        gaz_Ids.append([matched_Id, matched_length])
                    else:
                        gaz_Ids.append([])

                ## batch_size = 1

                # 以下都是在对上述的所有数组生成对应的mask数组
                for idx in range(w_length):
                    gazmask = []
                    gazcharmask = []

                    for label in range(4):
                        label_len = len(gazs[idx][label])
                        count_set = set(gazs_count[idx][label])
                        if len(count_set) == 1 and 0 in count_set:
                            gazs_count[idx][label] = [1]*label_len

                        # 生成mask
                        mask = label_len*[0]
                        mask += (max_gazlist-label_len)*[1]

                        # 根据最长的BMSE数组的长度对每一个数组进行补充padding
                        gazs[idx][label] += (max_gazlist-label_len)*[0]  ## padding
                        gazs_count[idx][label] += (max_gazlist-label_len)*[0]  ## padding

                        char_mask = []
                        for g in range(len(gaz_char_Id[idx][label])):
                            glen = len(gaz_char_Id[idx][label][g])
                            charmask = glen*[0]
                            charmask += (max_gazcharlen-glen) * [1]
                            char_mask.append(charmask)
                            gaz_char_Id[idx][label][g] += (max_gazcharlen-glen) * [0]
                        gaz_char_Id[idx][label] += (max_gazlist-label_len)*[[0 for i in range(max_gazcharlen)]]
                        char_mask += (max_gazlist-label_len)*[[1 for i in range(max_gazcharlen)]]

                        gazmask.append(mask)
                        gazcharmask.append(char_mask)
                    layergazmasks.append(gazmask)
                    gazchar_masks.append(gazcharmask)

                texts = ['[CLS]'] + words + ['[SEP]']
                bert_text_ids = tokenizer.convert_tokens_to_ids(texts)

                # words是词（文本）向量 biwords是2个字（文本）向量  chars是词里的字(文本)二维向量  gazs是BMES数组四个数组（数组里是代表词的编号，并根据最长长度补充好了0）
                # 则instance_texts中一行代表一句话，话里的信息由5个数组保存
                instence_texts.append([words, biwords, chars, gazs])
                #  word_Ids是词对应的id数组，biword_Ids是双字对应的id数组，char_Ids则是词里的字对应的id数组（二维）
                #  gazs的确记录的是匹配的词的id    高勇2
                # gazs_count的确记录的是词的频率   高勇的频率是1
                # gaz_char_id记录的是匹配的词里的字符的id
                # gaz_Ids每一行由两部分组成，当前字匹配的所有词的id，和匹配的词的长度（每个字一行）
                # label_id在gaz_ids后面
                instence_Ids.append([word_Ids, biword_Ids, char_Ids, gaz_Ids, gazs, gazs_count, gaz_char_Id, layergazmasks,gazchar_masks, bert_text_ids])

            words = []
            biwords = []
            chars = []
            word_Ids = []
            biword_Ids = []
            char_Ids = []

    return instence_texts, instence_Ids
