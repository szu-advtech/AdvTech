import numpy as np
import os
import math

splits = [1, 2, 3]

def count_word_and_class(sentences, total_cnt, word_cnt, class_cnt, class_word_cnt):
    for sentence in sentences:
        # total sentence N
        total_cnt += 1

        # class cnt N(a)
        class_cnt += 1

        words = sentence.strip().split()
        words = list(set(words))

        for word in words:

            # single word cnt N(w)
            if word not in word_cnt:
                word_cnt[word] = 1
            else:
                word_cnt[word] += 1

            if word not in class_word_cnt:
                class_word_cnt[word] = 1
            else:
                class_word_cnt[word] += 1

    return  total_cnt, word_cnt, class_cnt, class_word_cnt


def calculate_pmi(pmi_dict, total_cnt, word_cnt, class_cnt, class_word_cnt, domain):

    if domain == 'res':
        index = 0
    elif domain == 'lap':
        index = 1
    elif domain == 'dev':
        index = 2

    for word in pmi_dict:
        N = total_cnt
        N_w = word_cnt[word]
        N_a = class_cnt

        if word not in class_word_cnt:
            continue
        else:
            N_wa = class_word_cnt[word]
            pmi = np.log2( (N_wa * N) / (N_w * N_a))
            pmi_dict[word][index] = max(0., round(pmi, 4))

    return pmi_dict




    pass




for split in splits:
    dev_sentences = open('data/device/train{}/sentence.txt'.format(split), 'r', encoding='utf-8').readlines()
    res_sentences = open('data/restaurant/train{}/sentence.txt'.format(split), 'r', encoding='utf-8').readlines()
    lap_sentences = open('data/laptop/train{}/sentence.txt'.format(split), 'r', encoding='utf-8').readlines()

    total_cnt = 0 # N

    word_cnt = {} # N(w)

    dev_cnt = 0 # N(a)
    lap_cnt = 0
    res_cnt = 0

    dev_word_cnt = {} # N(a,w)
    lap_word_cnt = {}
    res_word_cnt = {}

    total_cnt, word_cnt, dev_cnt, dev_word_cnt = count_word_and_class(dev_sentences, total_cnt, word_cnt, dev_cnt, dev_word_cnt)
    total_cnt, word_cnt, res_cnt, res_word_cnt = count_word_and_class(res_sentences, total_cnt, word_cnt, res_cnt, res_word_cnt)
    total_cnt, word_cnt, lap_cnt, lap_word_cnt = count_word_and_class(lap_sentences, total_cnt, word_cnt, lap_cnt, lap_word_cnt)

    pmi_dict = {}
    frq_dict = {}
    for word in word_cnt:
        pmi_dict[word] = [0., 0., 0.]
        frq_dict[word] = [0, 0, 0]

    for word in frq_dict:
        'res lap dev'
        if word in res_word_cnt:
            frq_dict[word][0] = res_word_cnt[word]
        if word in lap_word_cnt:
            frq_dict[word][1] = lap_word_cnt[word]
        if word in dev_word_cnt:
            frq_dict[word][2] = dev_word_cnt[word]



    pmi_dict = calculate_pmi(pmi_dict, total_cnt, word_cnt, dev_cnt, dev_word_cnt, 'dev')
    pmi_dict = calculate_pmi(pmi_dict, total_cnt, word_cnt, res_cnt, res_word_cnt, 'res')
    pmi_dict = calculate_pmi(pmi_dict, total_cnt, word_cnt, lap_cnt, lap_word_cnt, 'lap')

    res_pmi_dict = {}
    lap_pmi_dict = {}
    dev_pmi_dict = {}

    for word in pmi_dict:
        res_pmi_dict[word] = pmi_dict[word][0]
        lap_pmi_dict[word] = pmi_dict[word][1]
        dev_pmi_dict[word] = pmi_dict[word][2]
        # print('{}\t{:.2f}\t{:.2f}\t{:.2f}'.format(word, pmi_dict[word][0], pmi_dict[word][1], pmi_dict[word][2]))

    with open('./data/pmi_dict_split{}.txt'.format(split), 'w', encoding='utf-8') as f:
        'detailed dictionary'
        # f.write('PMI DICTIONARY\n')
        # f.write('WORD RES LAP DEV\n')
        # for word in pmi_dict:
        #     pmis = list(map(str, pmi_dict[word]))
        #     f.write(word + ' ')
        #     f.write(' '.join(pmis))
        #     f.write('\n')

        'direct dictionary'
        f.write(str(pmi_dict))
        # with open('./data/pmi_dict_split{}.txt'.format(split), 'r', encoding='utf-8') as f:
        #     pmi_dict = eval(f.read())

    with open('./data/frq_dict_split{}.txt'.format(split), 'w', encoding='utf-8') as f:
        f.write(str(frq_dict))




    pass





