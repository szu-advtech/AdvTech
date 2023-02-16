#!/usr/bin/python
# -*- coding: UTF-8 -*-

import jieba
import sacrebleu
import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel,CpmTokenizer,BertTokenizer, BertForMaskedLM, OpenAIGPTLMHeadModel
import os
import json
# Load pre-trained model (weights)
from torch.nn import CrossEntropyLoss





#badwords ratio

def get_ngrams_from_sentence(sent, n=1, lowercase=True):
    # words = sent.strip().split(" ")
    # print(words)
    words = sent
    if lowercase:
        words = [word.lower() for word in words]
        ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
        return ngrams


def get_words(words_path):
    with open(words_path, "r") as f:
        words = []
        for line in f.readlines():
            line = line.strip()
            words.append(line)
    return words


def get_dist(output_path):
    with open(output_path, "r", encoding = 'utf-8') as f:
        all_distinct1 = []
        all_distinct2 = []
        for line in f.readlines():
            line = line.strip()
            all_unigrams = get_ngrams_from_sentence(line, n=1, lowercase=True)
            distinct1 = len(set(all_unigrams)) / len(all_unigrams)
            if len(line) > 1:
                all_bigrams = get_ngrams_from_sentence(line, n=2, lowercase=True)
                if all_bigrams == 0:
                    print('len(all_bigrams)为0')
                    continue
                distinct2 = len(set(all_bigrams)) / len(all_bigrams)
            else:
                distinct2 = 1.0

            all_distinct1.append(distinct1)
            all_distinct2.append(distinct2)

        distinct1_avg = sum(all_distinct1)/len(all_distinct1)
        distinct2_avg = sum(all_distinct2)/len(all_distinct2)

    return distinct1_avg, distinct2_avg


def get_ppl(STC_test_path, pretrain_model):
    print('0_0')

    # sens = ["今天是个好日子。", "天今子日。个是好", "这个婴儿有900000克呢。", "我不会忘记和你一起奋斗的时光。",
    #         "我不会记忘和你一起奋斗的时光。", "会我记忘和你斗起一奋的时光。"]
    f = open(STC_test_path, "r", encoding='utf-8')
    sens = []
    for line in f:
        post = line.find('标题：')
        if post != -1:
            continue
        sens.append(line.replace('\n', '').replace('“', '').replace('”', ''))
    print('1')
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 此处设置程序使用哪些显卡
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('2')
    model = GPT2LMHeadModel.from_pretrained(pretrain_model)
    model.eval()
    model = model.to(device)
    tokenizer = CpmTokenizer(vocab_file="../vocab/chinese_vocab.model")
    # tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
    # model = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
    a=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    for i in range(20):
        inputs = tokenizer(sens[50*i:50*(i+1)], padding='max_length', max_length=50, truncation=True, return_tensors="pt")
        inputs =inputs.to(device)
        print('5')
        bs, sl = inputs['input_ids'].size()
        outputs = model(**inputs, labels=inputs['input_ids'])
        print('6')
        logits = outputs[1]
        # Shift so that tokens < n predict n
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = inputs['input_ids'][:, 1:].contiguous()
        shift_attentions = inputs['attention_mask'][:, 1:].contiguous()
        print('7')
        # Flatten the tokens
        loss_fct = CrossEntropyLoss(ignore_index=0, reduction="none")
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).detach().reshape(bs, -1)
        meanloss = loss.sum(1) / shift_attentions.sum(1)
        ppl = torch.exp(meanloss).cpu().numpy().tolist()
        ppl = sum(ppl)/len(ppl)
        a[i]=ppl
        print(a)
    ppl=sum(a)/20
    return ppl

    # with torch.no_grad():
    #     # model = OpenAIGPTLMHeadModel.from_pretrained(pretrain_model)
    #     # model.eval()
    #     os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 此处设置程序使用哪些显卡
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     model = GPT2LMHeadModel.from_pretrained(pretrain_model)
    #     model.eval()
    #     model = model.to(device)
    #     # Load pre-trained model tokenizer (vocabulary)
    #     tokenizer = CpmTokenizer(vocab_file="../vocab/chinese_vocab.model")
    #     #tokenizer = BertTokenizer.from_pretrained(pretrain_model)
    #     print('1')
    #     with open(STC_test_path, "r", encoding='utf-8') as f:
    #         print('2')
    #         all_ppl = []
    #         for line in f.readlines():
    #             post = line.find('标题：')
    #             if post != -1:
    #                 continue
    #             ref = line.strip()[15:]
    #             print(ref)
    #         # dict = json.load(f)
    #         # utterances = dict['test']
    #         # for utterance in utterances:
    #         #     ref = utterance[0]
    #             sentence = ref
    #             tokenize_input = tokenizer.tokenize(sentence)
    #             tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    #             sen_len = len(tokenize_input)
    #             sentence_loss = 0.
    #
    #             for i, word in enumerate(tokenize_input):
    #                 # add mask to i-th character of the sentence
    #                 tokenize_input[i] = '[MASK]'
    #                 mask_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    #                 mask_input = mask_input.to(device)
    #                 output = model(mask_input)
    #
    #                 prediction_scores = output[0]
    #                 softmax = nn.Softmax(dim=0)
    #                 ps = softmax(prediction_scores[0, i]).log()
    #                 word_loss = ps[tensor_input[0, i]]
    #                 sentence_loss += word_loss.item()
    #
    #                 tokenize_input[i] = word
    #                 print(i)
    #             ppl = np.exp(-sentence_loss / sen_len)
    #             ppl = ppl / sen_len
    #             all_ppl.append(ppl)
    #
    # all_ppl.append(ppl)
    # ppl_avg = sum(all_ppl) / len(all_ppl)
    # return ppl_avg

def get_post_com (result_path, is_post):
    with open(result_path, "r") as f:
        post_refs = []
        post_hyps = []
        com_refs = []
        com_hyps = []
        for line in f.readlines():
            line = line.strip().split(" | ")
            if is_post:
                post = line[0]
                hyp = line[1]
                ref = line[2]
                post_hyps.append(hyp)
                post_refs.append(ref)

            else:
                coms = line[0]
                refs = line[1]
                com_refs.append(ref)

    post_refs_all = post_refs
    post_hyps_all = post_hyps
    com_refs_all = com_refs
    return post_refs_all, post_hyps_all, com_refs_all

def get_hyp_ref(hyp_file, ref_file):
    with open(hyp_file, 'r', encoding='utf-8') as fhyp:
        hyps= []
        for line in fhyp.readlines():
            post = line.find('title:')
            if post != -1:
                continue
            #line = line.strip()[22:322]
            hyps.append(line)
    fhyp.close()

    with open(ref_file, 'r', encoding='utf-8') as fref:
        refs = []
        for line in fref.readlines():
            post = line.find('标题：')
            if post != -1:
                continue
            line = line.strip()[15:315]
            refs.append(line)
    fref.close()
        # dict = json.load(fref)
        # utterances = dict['test']
        # for utterance in utterances:
        #     ref = utterance[1]
        #     refs.append(ref)

    return refs, hyps


def get_sacrebleu(refs, hyps):
    refs = [refs]
    bleu = sacrebleu.corpus_bleu(hyps, refs)
    return bleu.score

# def get_humen_evaluation(file):
#     with open(file, "r", encoding='utf-8') as f:
#         count_0 = 0
#         count_1 = 0
#         count_2 = 0
#         # count = 0
#         for line in f.readlines():
#             # count += 1
#             # print(count)
#             line = line.strip().split("$$")
#             score = int(line[1])
#             if score == 0:
#                 count_0 += 1
#             elif score == 1:
#                 count_1 += 1
#             elif score == 2:
#                 count_2 += 1
#
#         ratio_0 = count_0/200
#         ratio_1 = count_1/200
#         ratio_2 = count_2/200
#
#     return ratio_0, ratio_1, ratio_2





# # txt_path = 'eg.txt'
# post_merge_path = 'post_merge.txt'
# com_merge_path = 'com_merge.txt'
# # com_merge_path = 'ppltest.txt'

hyp_file = 'text_pangu_epoch2.txt'
ref_file = 'test_human.txt'
pretrain_model = '../model/large_all/epoch3/epoch8'

refs_list, hyps_list = get_hyp_ref(hyp_file, ref_file)
print(len(hyps_list))
print(len(refs_list))

bleu = get_sacrebleu(refs_list, hyps_list)

distinct1, distinct2 = get_dist(hyp_file)

ppl = get_ppl('test_human.txt', pretrain_model)
print('ppl:',ppl)
# print('/distinct-1:', distinct1, '/distinct-2:', distinct2, 'BLEU:', bleu)

# human_com_ratio_0, human_com_ratio_1, human_com_ratio_2 = get_humen_evaluation(human_eval_com_path)
# human_post_ratio_0, human_post_ratio_1, human_post_ratio_2 = get_humen_evaluation(human_eval_post_path)
# print('[post]', 'human_evaluation:', 'ratio_:0', human_post_ratio_0, 'ratio_1:', human_post_ratio_1, 'ratio_2:', human_post_ratio_2)
# print('[com]', 'human_evaluation:', 'ratio_:0', human_com_ratio_0, 'ratio_1:', human_com_ratio_1, 'ratio_2:', human_com_ratio_2)
#
# post_refs_all, post_hyps_all, com_refs_all = get_post_com(post_merge_path, is_post=True)
# bleu = get_sacrebleu(post_refs_all, post_hyps_all)
# print('post BLEU:', bleu)
# bert-score -r /Users/zhaojiaxu/PycharmProjects/toxicdial/evaluation/refs_post.txt -c /Users/zhaojiaxu/PycharmProjects/toxicdial/evaluation/hyps_post.txt --lang zh

# pip install --user fast-bleu --install-option="--CC=/opt/homebrew/bin/aarch64-apple-darwin20-gcc-11/" --install-option="--CXX=/opt/homebrew/bin/aarch64-apple-darwin20-g++-11/"
# aarch64-apple-darwin20-gcc-11
# pip install --user fast-bleu --install-option="--CC=<path-to-gcc-11>" --install-option="--CXX=<path-to-g++-11>"

