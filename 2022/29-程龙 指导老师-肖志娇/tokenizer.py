import collections
from keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np


class Tokenizer(object):
    def __init__(self):
        self.word_count = collections.Counter()
        self.w2i = {}
        self.i2w = {}
        self.oov_index = None
        self.vocab_size = None
        self.vectors = {}

    def save(self, path):
        pickle.dump(self, open(path, 'wb'))

    def load(self, path):
        return pickle.load(open(path, 'rb'))

    def train(self, texts, vocab_size):

        if len(self.word_count) != 0:
            raise Exception("To update existing tokenizer with new vocabulary, run update() or update_from_file()")

        
        for sent in texts:
            for w in sent.split():
                self.word_count[w] += 1

        self.vocab_size = vocab_size

        for count, w in enumerate(self.word_count.most_common(self.vocab_size-2)):
            self.w2i[w[0]] = count+1
            self.i2w[count+1] = w[0]
        self.oov_index = min([self.vocab_size-1, len(self.word_count)+1])
        self.vocab_size = self.oov_index+1
        self.w2i['<UNK>'] = self.oov_index
        self.w2i['<NULL>'] = 0
        self.i2w[0] = '<NULL>'
        self.i2w[self.oov_index] = '<UNK>'

    def train_from_file(self, path, vocab_size):
        if len(self.word_count) != 0:
            raise Exception("To update existing tokenizer with new vocabulary, run update() or update_from_file()")

        self.vocab_size = vocab_size

        for line in open(path):
            tmp = [x.strip() for x in line.split(',')]
            fid = tmp[0]
            sent = tmp[1]
            for w in sent.split():
                self.word_count[w] += 1

        for count, w in enumerate(self.word_count.most_common(self.vocab_size-2)):
            self.w2i[w[0]] = count+1
            self.i2w[count+1] = w[0]
        self.oov_index = min([self.vocab_size-1, len(self.word_count)+1])
        self.vocab_size = self.oov_index+1
        self.w2i['<UNK>'] = self.oov_index
        self.w2i['<NULL>'] = 0
        self.i2w[0] = '<NULL>'
        self.i2w[self.oov_index] = '<UNK>'
    
    def update(self, texts):
        # takes a list of strings that are space delimited into tokens
        for sent in texts:
            for w in sent.split():
                self.word_count[w] += 1

        self.w2i = {}
        self.i2w = {}


        for count, w in enumerate(self.word_count.most_common(self.vocab_size-2)):
            self.w2i[w[0]] = count+1
            self.i2w[count+1] = w[0]
        self.oov_index = min([self.vocab_size-1, len(self.word_count)+1])
        self.vocab_size = self.oov_index+1
        self.w2i['<UNK>'] = self.oov_index
        self.w2i['<NULL>'] = 0
        self.i2w[0] = '<NULL>'
        self.i2w[self.oov_index] = '<UNK>'      

    def update_from_file(self, path):
        for line in open(path):
            tmp = [x.strip() for x in line.split(',')]
            # takes a list of strings that are space delimited into tokens
            fid = tmp[0]
            sent = tmp[1]
            for w in sent.split():
                self.word_count[w] += 1

        self.w2i = {}
        self.i2w = {}
        
        for count, w in enumerate(self.word_count.most_common(self.vocab_size-2)):
            self.w2i[w[0]] = count+1
            self.i2w[count+1] = w[0]
        self.oov_index = min([self.vocab_size-1, len(self.word_count)+1])
        self.vocab_size = self.oov_index+1
        self.w2i['<UNK>'] = self.oov_index
        self.w2i['<NULL>'] = 0
        self.i2w[0] = '<NULL>'
        self.i2w[self.oov_index] = '<UNK>'

    def set_vocab_size(self, vocab_size):
        self.vocab_size = vocab_size
        self.w2i = {}
        self.i2w = {}
        
        for count, w in enumerate(self.word_count.most_common(self.vocab_size-2)):
            self.w2i[w[0]] = count+1
            self.i2w[count+1] = w[0]
        self.oov_index = min([self.vocab_size-1, len(self.word_count)+1])
        self.w2i['<UNK>'] = self.oov_index
        self.w2i['<NULL>'] = 0
        self.i2w[0] = '<NULL>'
        self.i2w[self.oov_index] = '<UNK>'

    def texts_to_sequences(self, texts, maxlen=None, padding='post', truncating='post', value=0):

        if len(self.word_count) == 0:
            raise Exception("Tokenizer has not been trained, no words in vocabulary.")

        all_seq = list()
        for sent in texts:
            seq = []
            for w in sent.split():
                try:
                    seq.append(self.w2i[w])
                except:
                    seq.append(self.oov_index)

                if maxlen is not None:
                    if len(seq) == maxlen:
                        break

            all_seq.append(seq)
            
        return pad_sequences(all_seq, maxlen=maxlen, padding=padding, truncating=truncating, value=value)

    def texts_to_sequences_from_file(self, path, maxlen=50, padding='post', truncating='post', value=0):

        if len(self.word_count) == 0:
            raise Exception("Tokenizer has not been trained, no words in vocabulary.")

        all_seq = {}
        for line in open(path):
            tmp = [x.strip() for x in line.split(',')]
            fid = int(tmp[0])
            sent = tmp[1]
            seq = []
            for w in sent.split():
                try:
                    seq.append(self.w2i[w])
                except:
                    seq.append(self.oov_index)

                if maxlen is not None:
                    if len(seq) == maxlen:
                        break

            all_seq[fid] = seq
        return {key: newval for key, newval in zip(all_seq.keys(), pad_sequences(all_seq.values(), maxlen=maxlen, padding=padding, truncating=truncating, value=value))}
        

    def seq_to_text(self, seq):
        return [self.i2w[x] for x in seq]

    def forw2v(self, seq):
        return [self.i2w[x] for x in seq if self.i2w[x] not in ['<NULL>', '<s>', '</s>']]
