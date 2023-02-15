

import pickle
import numpy
from nltk import word_tokenize
from torch.utils.data import Dataset
import collections
import codecs
import re, os
from unicodedata import normalize
import string
dir = 'dataset/europarl-v7.fr-en.en'
clean_dir='dataset/en.pkl'
vocab_output = 'dataset/vocab.pkl'
final_dir='dataset/final.pkl'

def clean_lines(lines):
    cleaned = list()
    # prepare regex for char filtering
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    for line in lines:
        # normalize unicode characters
        line = normalize('NFD', line).encode('ascii', 'ignore')
        line = line.decode('UTF-8')
        # tokenize on white space
        line = line.split()
        # convert to lower case
        line = [word.lower() for word in line]
        # remove punctuation from each token
        line = [word.translate(table) for word in line]
        # remove non-printable chars form each token
        line = [re_print.sub('', w) for w in line]
        # remove tokens with numbers in them
        line = [word for word in line if word.isalpha()]
        # store as string
        cleaned.append(' '.join(line))
    return cleaned

def update_dataset(lines, vocab):
    new_lines = list()
    for line in lines:
        new_tokens = list()
        for token in line.split():
            if token in vocab:
                new_tokens.append(token)
            else:
                new_tokens.append('unk')
        new_line = ' '.join(new_tokens)
        new_lines.append(new_line)
    return new_lines

class CorpusData(Dataset):
    def __init__(self,dir,vocab_output):
        with open(dir, 'rb') as file:  # attention the way of file is opened
            self.text = pickle.load(file)

        with open(vocab_output, 'rb') as file:
            self.id_dic = pickle.load(file)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        sen = self.text[index]  # 根据index取句子
        sen_split = word_tokenize(sen)  # 将句子转换成list 每个元素对应一个单词或符号？
        inputs = numpy.zeros((1, 30))  # used to pad sentence
        num = 0
        for word in sen_split:
            inputs[0, num] = self.id_dic[word]
            num += 1
            if(num >= 30):  # at most store 30 words
                break

        return inputs, num

dir = 'dataset/europarl-v7.fr-en.en'
clean_dir='dataset/en.pkl'
vocab_output = 'dataset/vocab.pkl'
final_dir='dataset/final.pkl'
# f=open(en_dir,'wb')
dataset=CorpusData(final_dir,vocab_output)
print(type(dataset.text))
print(len(dataset.text))
print(dataset.__getitem__(0))
print(word_tokenize(dataset.text[0]))


# sentences=dataset.text
# clean_sentences=clean_lines(sentences)
# pickle.dump(clean_sentences,f)
# counter = collections.Counter()
# # 读取处理过的句子
# with open(clean_dir, 'rb') as file:
#    clean_sentences=pickle.load(file)

# counter = collections.Counter()

# for line in clean_sentences:
#     for word in line.strip().split():
#         counter[word]+=1
#         # print(word)
# print(type(counter.items()))
# print(len(counter.items()))
# min_occurance=7
# # 出现次数少于min_occurance的被丢掉
# tokens = [k for k, c in counter.items() if c >= min_occurance]
# print(type(tokens))
# print(len(tokens))

# vocab={word:i+1 for i,word in enumerate(tokens)}


# # with open(vocab_output, 'rb') as file:
# #     vocab = pickle.load(file)
# #     # print(id_dic)
# #     print(type(vocab))
# #     print(len(vocab))

# final_snetences=update_dataset(clean_sentences,tokens)

# # 保持最终语料库
# f=open(final_dir,'wb')

# pickle.dump(final_snetences,f)
# # 不在字典外的字看成'unk'
# vocab['unk'] = 0
# # 保存字典
# f=open(vocab_output,'wb')
# pickle.dump(vocab,f)

# print(id_dic['Socrates'])
# items=id_dic.items()
# print(type(items))
# dataset.__getitem__(1)
# vocab = to_vocab(lines)
# print('English Vocabulary: %d' % len(vocab))