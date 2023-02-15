import networkx as nx
import random
import scanpy as sc
import codecs
import csv


def facebook_combined(G=nx.Graph()):
    with open('./dataset/facebook_combined.txt', 'r') as f:
        for line in f.readlines():
            e = line.strip().split()
            G.add_edge(int(e[0]), int(e[1]))

    return G


def email_univ(G=nx.Graph()):
    with open('./dataset/email_univ.txt', 'r') as f:
        for line in f.readlines():
            e = line.strip().split()
            G.add_edge(int(e[0]), int(e[1]))

    return G


def lastfm_asia(G=nx.Graph()):
    with codecs.open('./dataset/lastfm_asia_edges.csv', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f, skipinitialspace=True):
            G.add_edge(int(row['node_1']), int(row['node_2']))
    return G


def fb_pages_politician(G=nx.Graph()):
    with open('./dataset/fb-pages-politician.txt', 'r') as f:
        for line in f.readlines():
            e = line.strip().split(',')
            G.add_edge(int(e[0]), int(e[1]))

    return G


def fb_pages_company(G=nx.Graph()):
    with open('./dataset/fb-pages-company.txt', 'r') as f:
        for line in f.readlines():
            e = line.strip().split(',')
            G.add_edge(int(e[0]), int(e[1]))

    return G


def ca_GrQc(G=nx.Graph()):
    with open('./dataset/ca-GrQc.txt', 'r') as f:
        for line in f.readlines():
            e = line.strip().split()
            G.add_edge(int(e[0]), int(e[1]))

    return G


def ca_HepTh(G=nx.Graph()):
    with open('./dataset/ca-HepTh.txt', 'r') as f:
        for line in f.readlines():
            e = line.strip().split()
            G.add_edge(int(e[0]), int(e[1]))

    return G


def email_Enron(G=nx.Graph()):
    with open('./dataset/email-Enron.txt', 'r') as f:
        for line in f.readlines():
            e = line.strip().split()
            G.add_edge(int(e[0]), int(e[1]))

    return G


def cond_mat(G=nx.Graph()):
    adata = sc.read('./dataset/cond-mat.mtx')
    data = adata.X
    data = data.todense()
    print(len(data))
    return G


def astro_ph(G=nx.Graph()):
    with open('./dataset/astro-ph.txt', 'r') as f:
        for line in f.readlines():
            e = line.strip().split()
            G.add_edge(e[0], e[1])

    return G


def initG():
    G = nx.Graph()
    G.add_edge(1, 2)
    G.add_edge(1, 3)
    G.add_edge(1, 5)

    G.add_edge(2, 1)
    G.add_edge(2, 3)
    G.add_edge(2, 4)

    G.add_edge(3, 1)
    G.add_edge(3, 2)
    G.add_edge(3, 4)
    G.add_edge(3, 5)
    G.add_edge(3, 10)

    G.add_edge(5, 1)
    G.add_edge(5, 3)
    G.add_edge(5, 4)
    G.add_edge(5, 7)
    G.add_edge(5, 15)

    G.add_edge(4, 2)
    G.add_edge(4, 3)
    G.add_edge(4, 5)
    G.add_edge(4, 6)

    G.add_edge(6, 4)
    G.add_edge(6, 7)
    G.add_edge(6, 8)
    G.add_edge(6, 9)

    G.add_edge(7, 5)
    G.add_edge(7, 6)
    G.add_edge(7, 8)
    G.add_edge(7, 9)

    G.add_edge(8, 6)
    G.add_edge(8, 7)
    G.add_edge(8, 9)

    G.add_edge(9, 6)
    G.add_edge(9, 7)
    G.add_edge(9, 8)
    G.add_edge(9, 14)

    G.add_edge(10, 3)
    G.add_edge(10, 11)
    G.add_edge(10, 13)
    G.add_edge(10, 15)

    G.add_edge(11, 10)
    G.add_edge(11, 12)
    G.add_edge(11, 13)
    G.add_edge(11, 15)

    G.add_edge(12, 11)
    G.add_edge(12, 13)

    G.add_edge(13, 10)
    G.add_edge(13, 11)
    G.add_edge(13, 12)
    G.add_edge(13, 14)
    G.add_edge(13, 15)

    G.add_edge(14, 9)
    G.add_edge(14, 13)
    G.add_edge(14, 15)

    G.add_edge(15, 5)
    G.add_edge(15, 10)
    G.add_edge(15, 11)
    G.add_edge(15, 13)
    G.add_edge(15, 14)
    return G


def initBigG():
    G = nx.Graph()
    for i in range(1, 101):
        G.add_node(i)

    for i in range(0, 5):
        for k in range(0, 200):
            u = random.randint(1 + i * 20, 20 + i * 20)
            v = random.randint(1 + i * 20, 20 + i * 20)
            if u != v:
                G.add_edge(u, v)

    for i in range(0, 100):
        u = random.randint(1, 100)
        v = random.randint(1, 100)
        if u != v:
            G.add_edge(u, v)
    return G


def initBigG2():
    G = nx.Graph()
    for i in range(0, 1000):
        G.add_node(i)

    for i in range(0, 50):
        for k in range(0, 100):
            u = random.randint(1 + i * 20, 20 + i * 20)
            v = random.randint(1 + i * 20, 20 + i * 20)
            if u != v:
                G.add_edge(u, v)

    for i in range(0, 70):
        u = random.randint(1, 1000)
        v = random.randint(1, 1000)
        if u != v:
            G.add_edge(u, v)
    return G
