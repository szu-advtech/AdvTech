'''
Generate processed data: filtering, combining ...
'''

import numpy as np
from sklearn import preprocessing
import json
import pickle
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

def add_set(x, s):
    for x_ in x.split(','):
        s.add(x_)

def get_allFile(path):
    data = pd.DataFrame()
    # dirFil = 列出当前文件夹下的所有文件夹及文件
    dirFil = os.listdir(path)
    for item in dirFil:
        # 目录连接
        nowPath = path + '/' + item
        print(nowPath)

        # 判断是不是文件，是则读出
        if os.path.isfile(nowPath):
            data_ = pd.read_csv(nowPath, usecols=['Name', 'Authors', 'ISBN', 'Rating', 'CountsOfReview', 'Description'])
            data = pd.concat([data, data_], ignore_index=False, axis=0)
        else:
            print('文件类型未知，pass', nowPath)

    data.dropna(inplace=True)
    data.drop_duplicates(subset=['ISBN'], keep='first', inplace=True)

    data['Authors'] = data['Authors'].apply(lambda s:s.split('/'))

    # filter review>=3
    data['CountsOfReview'] = data['CountsOfReview'].apply(lambda s: int(s))
    data = data[data['CountsOfReview']>=3]

    # filter author>=2 and <=50
    author_book = {}
    for index, row in data.iterrows():
        for au in row['Authors']:
            if(au in author_book.keys()):
                author_book[au].append(row['ISBN'])
            else:
                author_book[au] = [row['ISBN']]

    select_author = set()
    select_book = set()
    del_key = []
    for key, value in author_book.items():
        if(len(value)>=2 and len(value)<=50):
            select_book = set.union(select_book, set(value))
            select_author.add(key)
        else:
            del_key.append(key)
    for k in del_key:
        del author_book[k]

    print('len of author:', len(select_author), '   len of book:', len(select_book))

    author_map = dict(enumerate(select_author))
    book_map = dict(enumerate(select_book))
    author_map = dict(zip(author_map.values(), author_map.keys()))
    book_map = dict(zip(book_map.values(), book_map.keys()))

    data = data[data['ISBN'].isin(select_book)]
    data['ISBN'] = data['ISBN'].apply(lambda s:book_map[s])
    data.sort_values('ISBN', inplace=True)

    treatment_binary = list(data['Rating'].apply(lambda x:1 if x>3 else 0))

    data['asin'] = data['Description'] + ' ' + data['Name']

    # bag of words
    corpus = list(data['asin'])
    cv = CountVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 1), max_features=500)  # tokenizer=LemmaTokenizer()
    cv_fit = cv.fit_transform(corpus)  # top 500 x word num
    word_name = cv.get_feature_names()  # dictionary
    print('word num: ', len(word_name), ' ', word_name)
    # normalize
    cv_fit = cv_fit.toarray()
    features = preprocessing.normalize(cv_fit)
    features += np.random.normal(0, 1, size=(features.shape[0], features.shape[1]))
    print('feature mean/std: ', np.mean(features), np.std(features))

    # hypergraph
    hyperedge_index = [[], []]
    for key, value in author_book.items():
        for v in value:
            hyperedge_index[0].append(book_map[v])
            hyperedge_index[1].append(author_map[k])
    hyperedge_index = np.array(hyperedge_index)

    with open('goodreads.pickle', 'wb') as f:
        pickle.dump({'features':features, 'hyperedge_index':hyperedge_index,'treatments':treatment_binary}, f)


if __name__ == '__main__':
    path = 'archive'
    get_allFile(path)






