from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from tqdm import tqdm

def read_data(fname, word2idx, opt, data, data_type):
    max_length = opt.max_sentence_len
    source = opt.source
    target = opt.target
    topk = opt.topk

    assert data_type in ['train', 'test', 'unlabel']

    with open('./data/dep.dict', 'r', encoding='utf-8') as f:
        dep_dict = eval(f.read())
    with open('./data/pos.dict', 'r', encoding='utf-8') as f:
        pos_dict = eval(f.read())

    'SOURCE TRAINING DATA'
    review = open(fname + r'sentence.txt', 'r', encoding='utf-8').readlines()
    pos_data = open(fname + r'pos.txt', 'r', encoding='utf-8').readlines()
    graph_data = pickle.load(open(fname + r'dep.graph', 'rb'))
    if data_type in ['train', 'test']:
        ae_data = open(fname + r'aspect.txt', 'r', encoding='utf-8').readlines()
        oe_data = open(fname + r'opinion.txt', 'r', encoding='utf-8').readlines()
    if data_type in ['train']:
        lm_data = open(fname + r'{}-to-{}-prototype.txt'.format(source, target), 'r', encoding='utf-8').readlines()
    else:
        lm_data = open(fname + r'{}-to-{}-prototype.txt'.format(target, target), 'r', encoding='utf-8').readlines()

    for index, _ in enumerate(review):

        '''WORD INDEX & MASK'''
        sptoks = review[index].strip().split()

        idx = []
        mask = []
        len_cnt = 0
        for sptok in sptoks:
            if len_cnt < max_length:
                idx.append(word2idx[sptok.lower()])
                mask.append(1.)
                len_cnt += 1
            else:
                break

        data_per = (idx + [0] * (max_length - len(idx)))
        mask_per = (mask + [0.] * (max_length - len(idx)))

        '''POS & DEP'''
        dep_graph = graph_data[index]
        dep_tags = []
        for i in range(len(dep_graph)):
            dep_multihot = [0.] * 40
            dep_slice = dep_graph[i]
            for dep in dep_slice:
                if dep != 0:
                    dep_multihot[dep - 1] = 1.
            dep_tags.append(dep_multihot)

        self_circle = dep_graph
        for i in range(len(self_circle)):
            self_circle[i][i] = 1.
        adj_per = (np.pad(self_circle, ((0, max_length - len(idx)), (0, max_length - len(idx))), 'constant'))

        dep_per = np.pad(np.array(dep_tags), ((0, max_length - len(idx)), (0, 0)), 'constant', constant_values=0)

        pos_line = pos_data[index].strip().split()
        pos_tags = []
        for pos in (pos_line):
            pos_onehot = [0.] * 45
            pos_indice = pos_dict[pos] - 1
            pos_onehot[pos_indice] = 1.
            pos_tags.append(pos_onehot)

        pos_per = np.pad(np.array(pos_tags), ((0, max_length - len(idx)), (0, 0)), 'constant', constant_values=0)

        '''ASPECT & OPINION LABELS'''
        if data_type in ['train', 'test']:
            ae_labels = ae_data[index].strip().split()
            aspect_label = []
            for l in ae_labels:
                l = int(l)
                if l == 0:
                    aspect_label.append([1, 0, 0])
                elif l == 1:
                    aspect_label.append([0, 1, 0])
                elif l == 2:
                    aspect_label.append([0, 0, 1])
                else:
                    raise ValueError

            aspect_y_per = (aspect_label + [[0, 0, 0]] * (max_length - len(idx)))

            oe_labels = oe_data[index].strip().split()
            opinion_label = []
            for l in oe_labels:
                l = int(l)
                if l == 0:
                    opinion_label.append([1, 0, 0])
                elif l == 1:
                    opinion_label.append([0, 1, 0])
                elif l == 2:
                    opinion_label.append([0, 0, 1])
                else:
                    raise ValueError

            opinion_y_per = (opinion_label + [[0, 0, 0]] * (max_length - len(idx)))
        else:
            aspect_y_per = [[0, 0, 0]] * (len(data_per))
            opinion_y_per = [[0, 0, 0]] * (len(data_per))

        'DOMAIN LABELS'
        if data_type in ['train']:
            domain_y_per = [0, 1]
        elif data_type in ['unlabel', 'test']:
            domain_y_per = [1, 0]

        'PROTOTYPES'
        lmwords_list = []
        lmprobs_list = []

        if lm_data[index].strip() == 'NULL':
            raise ValueError
        else:
            segments = lm_data[index].strip().split('###')
            for segment in segments:
                lminfo = segment.split('@@@')
                try:
                    position = int(lminfo[0])
                except ValueError:
                    print(lm_data[index].strip())
                    print('debug')
                pairs = lminfo[1:]
                words = []
                probs = []
                topk_cnt = 0
                for pair in pairs:
                    if topk_cnt >= topk:
                        break
                    word = word2idx[pair.split()[0]]
                    prob = float(pair.split()[1])
                    words.append(word)
                    probs.append(prob)
                    topk_cnt += 1

                lmwords_list.append(words)
                lmprobs_list.append(softmax(probs))

        lmwords_per = (lmwords_list + [[0] * topk] * (max_length - len(idx)))
        lmprobs_per = (lmprobs_list + [[0.] * topk] * (max_length - len(idx)))


        data_per = {'x': np.array(data_per, dtype='int64'),
                    'mask': np.array(mask_per, dtype='float32'),
                    'dep': np.array(dep_per, dtype='float32'),
                    'pos': np.array(pos_per, dtype='float32'),
                    'adj': np.array(adj_per, dtype='float32'),
                    'lmwords':np.array(lmwords_per, dtype='int64'),
                    'lmprobs':np.array(lmprobs_per, dtype='float32'),
                    'aspect_y': np.array(aspect_y_per, dtype='int64'),
                    'domain_y': np.array(domain_y_per, dtype='int64')}

        if 'device' in fname and data_type == 'train':  # only use valid training samples in DEVICE
            print('Source Domain is DEVICE. Only use samples containing aspects for training.')
            if [0, 1, 0] in aspect_y_per:
                data.append(data_per)
            else:
                continue
        else:
            data.append(data_per)

class ABSADataset():
    def __init__(self, process, fname, word2idx, opt, unlabel_fname=None):
        data = []

        print('processing {} files: {}'.format(process, fname))
        read_data(fname, word2idx, opt, data, data_type=process)

        if unlabel_fname is not None:
            print('processing unlabeled files: {}'.format(unlabel_fname))
            read_data(unlabel_fname, word2idx, opt, data, data_type='unlabel')

        self.data = data
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

'GRL IMPLEMENTATION'

from torch.autograd import Function

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    batch_size, seq_length, prob_dim = logits.shape

    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y


    y = y.view(-1, prob_dim)

    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y)
    y_hard.scatter_(1, ind.view(-1, 1), 1)

    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard.view(batch_size, seq_length, prob_dim)

# def softmax(probs):
#     probs = np.array(probs)
#     mask = np.asarray(probs != 0, np.float32)
#     probs -= np.max(probs, axis=-1, keepdims=True)
#     probs_exp = np.exp(probs) * mask
#     return probs_exp / (np.sum(probs_exp, axis=-1, keepdims=True) + 1e-6)

def softmax(probs):
    probs = np.array(probs)
    probs -= np.max(probs, axis=-1, keepdims=True)
    return np.exp(probs) / (np.sum(np.exp(probs), axis=-1, keepdims=True) + 1e-6)

# def refine_softmax(probs):
#     refined_scores = probs ** (1 / 0.3)
#     refined_probs = refined_scores/(np.sum(refined_scores, -1, keepdims=True) + 1e-6)
#     return refined_probs