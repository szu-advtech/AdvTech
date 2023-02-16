from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertConfig, BertTokenizer
import jieba
import random
import jsonlines
from typing import List

def load_sts_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data_source = list()
        for line in f:
            line_split = line.strip().split("||")
            data_source.append((line_split[1], line_split[2], line_split[3]))
        return data_source

def random_swap_word(sentence, prob):
    if random.random() > prob:
        return sentence
    else:
        words = list(jieba.cut(sentence))
        if len(words) == 1:
            return sentence
        index1 = random.randint(0, len(words)-1)
        index2 = random.randint(0, len(words)-1)
        while index2 == index1:
            index2 = random.randint(0, len(words)-1)
        words[index1], words[index2] = words[index2], words[index1]
        sentence = "".join(words)
    return sentence

def random_delete_word(sentence, prob):
    if random.random() > prob:
        return sentence
    else:
        words = list(jieba.cut(sentence))
        delete_index = random.randint(0, len(words)-1)
        del words[delete_index]
        sentence = "".join(words)
    return sentence

def load_sts_data_unsup(path):
    with open(path, 'r', encoding='utf-8') as f:
        data_source = list()
        for line in f:
            line_split = line.strip().split("\n")
            data_source.append(line_split)
        return data_source


class TrainDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]
        # 同一个text重复两次， 通过bert编码互为正样本
        tokens = self.tokenizer([text, text], max_length=self.max_len,
                                truncation=True, padding='max_length', return_tensors='pt')
        return tokens


class TestDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def text2id(self, text):
        return self.tokenizer(text, max_length=self.max_len,
                              truncation=True, padding='max_length', return_tensors='pt')

    def __getitem__(self, index):
        da = self.data[index]
        return self.text2id([da[0]]), self.text2id([da[1]]), int(da[2])

class DevDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def text2id(self, text):
        return self.tokenizer(text, max_length=self.max_len,
                              truncation=True, padding='max_length', return_tensors='pt')

    def __getitem__(self, index):
        da = self.data[index]
        return self.text2id([da[0]]), self.text2id([da[1]]), int(da[2])


def load_data_sup(path):
    """根据名字加载不同的数据集
    """
    def load_sts_data(path):
        with open(path, 'r', encoding='utf8') as f:
            return [(line.split("||")[1], line.split("||")[2], line.split("||")[3]) for line in f]

    return load_sts_data(path)


class TrainDataset_sup(Dataset):
    """训练数据集, 重写__getitem__和__len__方法
    """

    def __init__(self, data: List):
        self.data = data

    def __len__(self):
        return len(self.data)

    def text_2_id(self, text: str):
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        return tokenizer([text[0], text[1], text[2]], max_length=64,
                         truncation=True, padding='max_length', return_tensors='pt')

    def __getitem__(self, index: int):
        return self.text_2_id(self.data[index])


