from nltk.translate.bleu_score import corpus_bleu
import codecs
import os
from rouge import rouge
import numpy as np
from sentence_transformers import SentenceTransformer

bert_model = SentenceTransformer('symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli').to("cuda")


def bert_s(gen, true):
    gen_emb = np.array(bert_model.encode(gen))
    ture_emb = np.array(bert_model.encode(true))
    dot = np.sum(gen_emb*ture_emb, axis=1)
    gen_norm = np.sqrt(np.sum(np.square(gen_emb), axis=1))
    ture_norm = np.sqrt(np.sum(np.square(ture_emb), axis=1))

    return np.mean(dot/(gen_norm*ture_norm))


log_path = '../City4_log/dianping/RAW_MSE_CAML_FN_FM/'

# 获取gen和true文件
for exp in os.listdir(log_path):
    if '0' <= exp[0] <= '9':
        continue
    dir_path = '../City4_log/dianping/RAW_MSE_CAML_FN_FM/{}/expalanation/'.format(exp)
    gen_sentence = []
    true_sentence = []
    gen_sentence_rouge = []
    true_sentence_rouge = []
    for i in range(9999999):
        gen_filename = dir_path + 'gen_review.' + str(i) + '.txt'
        true_filename = dir_path + 'true_review.A.' + str(i) + '.txt'
        if not os.path.exists(gen_filename):
            break
        gen_file = codecs.open(gen_filename, 'rb', 'utf-8').readlines()
        true_file = codecs.open(true_filename, 'rb', 'utf-8').readlines()

        for j in range(len(gen_file)):
            gen_str = gen_file[j]
            gen_sentence_rouge.append(gen_str)
            gen_split = gen_str.split(' ')
            new_sentence = []
            for k in range(len(gen_split)):
                new_sentence.append(gen_split[k])
            gen_sentence.append(new_sentence)
            true_str = true_file[j]
            true_sentence_rouge.append(true_str)
            true_split = true_str.split(' ')
            new_sentence = []
            for k in range(len(true_split)):
                new_sentence.append(true_split[k])
            true_sentence.append([new_sentence])

    score = corpus_bleu(true_sentence, gen_sentence, weights=(0.25, 0.25, 0.25, 0.25))
    print('*********************************{}*********************************'.format(exp))
    print('BLEU score:' + str(score * 100))
    print(rouge(gen_sentence_rouge, true_sentence_rouge))
    print('Bert_s score:' + str(bert_s(gen_sentence_rouge, true_sentence_rouge)))

