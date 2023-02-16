import datetime

import h5py
import numpy as np
from matplotlib import pylab
from timeit import default_timer as timer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

import utils.coreset as cs
from truncated_GMM import TruncatedGaussianMixture as GMM
from sklearn.mixture import GaussianMixture as TGMM

# used to draw
colors = {
'aliceblue':            '#F0F8FF',
'antiquewhite':         '#FAEBD7',
'aqua':                 '#00FFFF',
'aquamarine':           '#7FFFD4',
'azure':                '#F0FFFF',
'beige':                '#F5F5DC',
'bisque':               '#FFE4C4',
'black':                '#000000',
'blanchedalmond':       '#FFEBCD',
'blue':                 '#0000FF',
'blueviolet':           '#8A2BE2',
'brown':                '#A52A2A',
'burlywood':            '#DEB887',
'cadetblue':            '#5F9EA0',
'chartreuse':           '#7FFF00',
'chocolate':            '#D2691E',
'coral':                '#FF7F50',
'cornflowerblue':       '#6495ED',
'cornsilk':             '#FFF8DC',
'crimson':              '#DC143C',
'cyan':                 '#00FFFF',
'darkblue':             '#00008B',
'darkcyan':             '#008B8B',
'darkgoldenrod':        '#B8860B',
'darkgray':             '#A9A9A9',
'darkgreen':            '#006400',
'darkkhaki':            '#BDB76B',
'darkmagenta':          '#8B008B',
'darkolivegreen':       '#556B2F',
'darkorange':           '#FF8C00',
'darkorchid':           '#9932CC',
'darkred':              '#8B0000',
'darksalmon':           '#E9967A',
'darkseagreen':         '#8FBC8F',
'darkslateblue':        '#483D8B',
'darkslategray':        '#2F4F4F',
'darkturquoise':        '#00CED1',
'darkviolet':           '#9400D3',
'deeppink':             '#FF1493',
'deepskyblue':          '#00BFFF',
'dimgray':              '#696969',
'dodgerblue':           '#1E90FF',
'firebrick':            '#B22222',
'floralwhite':          '#FFFAF0',
'forestgreen':          '#228B22',
'fuchsia':              '#FF00FF',
'gainsboro':            '#DCDCDC',
'ghostwhite':           '#F8F8FF',
'gold':                 '#FFD700',
'goldenrod':            '#DAA520',
'gray':                 '#808080',
'green':                '#008000',
'greenyellow':          '#ADFF2F',
'honeydew':             '#F0FFF0',
'hotpink':              '#FF69B4',
'indianred':            '#CD5C5C',
'indigo':               '#4B0082',
'ivory':                '#FFFFF0',
'khaki':                '#F0E68C',
'lavender':             '#E6E6FA',
'lavenderblush':        '#FFF0F5',
'lawngreen':            '#7CFC00',
'lemonchiffon':         '#FFFACD',
'lightblue':            '#ADD8E6',
'lightcoral':           '#F08080',
'lightcyan':            '#E0FFFF',
'lightgoldenrodyellow': '#FAFAD2',
'lightgreen':           '#90EE90',
'lightgray':            '#D3D3D3',
'lightpink':            '#FFB6C1',
'lightsalmon':          '#FFA07A',
'lightseagreen':        '#20B2AA',
'lightskyblue':         '#87CEFA',
'lightslategray':       '#778899',
'lightsteelblue':       '#B0C4DE',
'lightyellow':          '#FFFFE0',
'lime':                 '#00FF00',
'limegreen':            '#32CD32',
'linen':                '#FAF0E6',
'magenta':              '#FF00FF',
'maroon':               '#800000',
'mediumaquamarine':     '#66CDAA',
'mediumblue':           '#0000CD',
'mediumorchid':         '#BA55D3',
'mediumpurple':         '#9370DB',
'mediumseagreen':       '#3CB371',
'mediumslateblue':      '#7B68EE',
'mediumspringgreen':    '#00FA9A',
'mediumturquoise':      '#48D1CC',
'mediumvioletred':      '#C71585',
'midnightblue':         '#191970',
'mintcream':            '#F5FFFA',
'mistyrose':            '#FFE4E1',
'moccasin':             '#FFE4B5',
'navajowhite':          '#FFDEAD',
'navy':                 '#000080',
'oldlace':              '#FDF5E6',
'olive':                '#808000',
'olivedrab':            '#6B8E23',
'orange':               '#FFA500',
'orangered':            '#FF4500',
'orchid':               '#DA70D6',
'palegoldenrod':        '#EEE8AA',
'palegreen':            '#98FB98',
'paleturquoise':        '#AFEEEE',
'palevioletred':        '#DB7093',
'papayawhip':           '#FFEFD5',
'peachpuff':            '#FFDAB9',
'peru':                 '#CD853F',
'pink':                 '#FFC0CB',
'plum':                 '#DDA0DD',
'powderblue':           '#B0E0E6',
'purple':               '#800080',
'red':                  '#FF0000',
'rosybrown':            '#BC8F8F',
'royalblue':            '#4169E1',
'saddlebrown':          '#8B4513',
'salmon':               '#FA8072',
'sandybrown':           '#FAA460',
'seagreen':             '#2E8B57',
'seashell':             '#FFF5EE',
'sienna':               '#A0522D',
'silver':               '#C0C0C0',
'skyblue':              '#87CEEB',
'slateblue':            '#6A5ACD',
'slategray':            '#708090',
'snow':                 '#FFFAFA',
'springgreen':          '#00FF7F',
'steelblue':            '#4682B4',
'tan':                  '#D2B48C',
'teal':                 '#008080',
'thistle':              '#D8BFD8',
'tomato':               '#FF6347',
'turquoise':            '#40E0D0',
'violet':               '#EE82EE',
'wheat':                '#F5DEB3',
'white':                '#FFFFFF',
'whitesmoke':           '#F5F5F5',
'yellow':               '#FFFF00',
'yellowgreen':          '#9ACD32'}
colors = list(colors.values())

# load CIFAR-10 dataset
import pickle
# Unpickle the data of CIFAR
# Get data like: dict[b'data'], dict[b'fine_labels']
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
CIFAR = unpickle("dataset/train")

# Full dataset
# X = np.loadtxt("dataset/birch1.txt")
c = cs.Coreset("dataset/birch1.txt")
# c.load_data(CIFAR[b'data'], CIFAR[b'fine_labels'])
c.load_UDF(np.loadtxt)
# c.construct(cluster=100, delta=0.1, epsilon=0.1)
c.construct(10000)
X = c.data
Xc = c.coreset

# init parameters
params = {
    'algorithm': 'var-GMM-S',       # using algorithm
    'C': 100,                       # number of clusters
    'Cprime': 5,                    # truncated covariance
    'G': 5,                         # number of 'neighboring' clusters
    'Niter': 25,                    # iteration times
    'Ninit': 5,                     # init Ninit times to get gain better K and G_c
    # 'dataset': 'BIRCH2-400',        # path to dataset
    'VERBOSE': True,                # output
}
user_params = {'algorithm' : 'var-GMM-S+1'}
params.update(user_params)
params['C'] = int(params['C'])
params['Cprime'] = int(params['Cprime'])
params['G'] = int(params['G'])
params['Niter'] = int(params['Niter'])
params['Ninit'] = int(params['Ninit'])
params['VERBOSE'] = True if (params['VERBOSE'] == True or params['VERBOSE'] == 'True') else False

if params['VERBOSE']:
    params['VERBOSE'] = {
        'll': True,  # loglikelihood
        'fe': True,  # free energy
        'qe': True,  # quantization error
        'cs': True,  # clustering scores (purity, NMI, AMI)
        'nd': True,  # number of distance evaluations
        'np': 1,  # picture output every n iterations
    }
else:
    params['VERBOSE'] = {
        'll': False,
        'fe': False,
        'qe': False,
        'cs': False,
        'nd': True,
        'np': np.inf,
    }

# test only init
# params.update({'Niter': 25})
# truncated GMM
gmm = GMM(params=params)    #154
tgmm = TGMM(n_init=params['Ninit'], max_iter=params['Niter'], n_components=params['C'])
params.update({'C':110})
gmm2 = GMM(params=params)   #151
tgmm2 = TGMM(n_init=params['Ninit'], max_iter=params['Niter'], n_components=params['C'])
params.update({'C':130})
gmm3 = GMM(params=params)   #162
tgmm3 = TGMM(n_init=params['Ninit'], max_iter=params['Niter'], n_components=params['C'])
params.update({'C':200})
gmm4 = GMM(params=params)   #164
tgmm4 = TGMM(n_init=params['Ninit'], max_iter=params['Niter'], n_components=params['C'])
params.update({'C':500})
gmm5 = GMM(params=params)   #202
tgmm5 = TGMM(n_init=params['Ninit'], max_iter=params['Niter'], n_components=params['C'])
params.update({'C':1000})
gmm6 = GMM(params=params)   #268
tgmm6 = TGMM(n_init=params['Ninit'], max_iter=params['Niter'], n_components=params['C'])
avt1 = datetime.datetime.now() - datetime.datetime.now()
avt2 = datetime.datetime.now() - datetime.datetime.now()
avt3 = datetime.datetime.now() - datetime.datetime.now()
avt4 = datetime.datetime.now() - datetime.datetime.now()
avt5 = datetime.datetime.now() - datetime.datetime.now()
avt6 = datetime.datetime.now() - datetime.datetime.now()
for i in range(10):
    print("vcgmm1:")
    gmm.fit_weight(c.coreset, c.coreset_label, filename="C100_{}".format(i), origin_X=c.data, weight=c.weight)
    print("vcgmm2:")
    gmm2.fit_weight(c.coreset, c.coreset_label, filename="C110_{}".format(i), origin_X=c.data, weight=c.weight)
    print("vcgmm3:")
    gmm3.fit_weight(c.coreset, c.coreset_label, filename="C150_{}".format(i), origin_X=c.data, weight=c.weight)
    print("vcgmm4:")
    gmm4.fit_weight(c.coreset, c.coreset_label, filename="C200_{}".format(i), origin_X=c.data, weight=c.weight)
    print("vcgmm5:")
    gmm5.fit_weight(c.coreset, c.coreset_label, filename="C500_{}".format(i), origin_X=c.data, weight=c.weight)
    print("vcgmm6:")
    gmm6.fit_weight(c.coreset, c.coreset_label, filename="C1000_{}".format(i), origin_X=c.data, weight=c.weight)
    st = datetime.datetime.now()
    tgmm.fit(c.coreset)
    avt1 += datetime.datetime.now() - st
    st = datetime.datetime.now()
    tgmm2.fit(c.coreset)
    avt2 += datetime.datetime.now() - st
    st = datetime.datetime.now()
    tgmm3.fit(c.coreset)
    avt3 += datetime.datetime.now() - st
    st = datetime.datetime.now()
    tgmm4.fit(c.coreset)
    avt4 += datetime.datetime.now() - st
    st = datetime.datetime.now()
    tgmm5.fit(c.coreset)
    avt5 += datetime.datetime.now() - st
    st = datetime.datetime.now()
    tgmm6.fit(c.coreset)
    avt6 += datetime.datetime.now() - st
#av1:0:00:03.978691,av2:0:00:03.898825,av3:0:00:05.201657,av4:0:00:14.038272,av5:0:01:35.378465,av6:0:03:20.876664
print("av1:{},av2:{},av3:{},av4:{},av5:{},av6:{}".format(avt1.seconds/10, avt2.seconds/10, avt3.seconds/10, avt4.seconds/10, avt5.seconds/10, avt6.seconds/10))
# params.update({'C':2000})
# gmm7 = GMM(params=params)   #cannot run
# traditional GMM

# # tgmm fit coreset
# print("tgmm model")
# tgmm = TGMM(n_init=params['Ninit'], max_iter=params['Niter'], n_components=params['C'])
# print("start fitting")
# st = datetime.datetime.now()
# tgmm.fit(c.coreset)
# print("cost {}".format((datetime.datetime.now() - st).seconds))
# print("end fitting")
# print("start predicting")
# st = datetime.datetime.now()
# labels_cc = tgmm.predict(c.coreset)
# print("cost {}".format((datetime.datetime.now() - st).seconds))
# print("end predicting")
# fig = plt.figure(figsize=(10, 10), dpi=80)
# pylab.xlim(10000, 1000000)
# pylab.ylim(10000, 1000000)
# print("start figuring")
# for i in range(c.coreset.shape[0]):
#     plt.scatter(c.coreset[i][0], c.coreset[i][1], color=colors[labels_cc[i]], s=1)
# plt.savefig("tgmm_coreset_coreset" + pylab.datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".png")
# plt.close(fig)
# print("end figuring")

# fig = plt.figure(figsize=(10, 10), dpi=80)
# pylab.xlim(10000, 1000000)
# pylab.ylim(10000, 1000000)
# print("start predicting")
# st = datetime.datetime.now()
# labels_cf = tgmm.predict(c.data)
# print("cost {}".format((datetime.datetime.now() - st).seconds))
# print("end predicting")
# print("start figuring")
# for i in range(c.data.shape[0]):
#     plt.scatter(c.data[i][0], c.data[i][1], color=colors[labels_cf[i]], s=1)
# plt.savefig("tgmm_coreset_full" + pylab.datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".png")
# plt.close(fig)
# print("end figuring")


# # tgmm fit full dataset
# tgmm = TGMM(n_init=params['Ninit'], max_iter=params['Niter'], n_components=params['C'])
# print("start fitting")
# st = datetime.datetime.now()
# tgmm.fit(c.data)
# print("cost {}".format((datetime.datetime.now() - st).seconds))
# print("end fitting")

# print("start predicting")
# st = datetime.datetime.now()
# labels_fc = tgmm.predict(c.coreset)
# print("end predicting")

# fig = plt.figure(figsize=(10, 10), dpi=80)
# pylab.xlim(10000, 1000000)
# pylab.ylim(10000, 1000000)
# print("start figuring")
# for i in range(c.coreset.shape[0]):
#     plt.scatter(c.coreset[i][0], c.coreset[i][1], color=colors[labels_fc[i]], s=1)
# plt.savefig("tgmm_full_coreset" + pylab.datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".png")
# plt.close(fig)
# print("end figuring")

# print("start predicting")
# st = datetime.datetime.now()
# labels_ff = tgmm.predict(c.data)
# print("end predicting")

# fig = plt.figure(figsize=(10, 10), dpi=80)
# pylab.xlim(10000, 1000000)
# pylab.ylim(10000, 1000000)
# print("start figuring")
# for i in range(c.data.shape[0]):
#     plt.scatter(c.data[i][0], c.data[i][1], color=colors[labels_ff[i]], s=1)
# plt.savefig("tgmm_full_full" + pylab.datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".png")
# plt.close(fig)
# print("end figuring")



# # data used for GMM cluster
# data = Xc
# # fit coreset
# st = datetime.datetime.now()
# gmm.fit_weight(data, None, filename="test_coreset_birch_C100", origin_X=X, plot=True, weight=c.weight)
# print("100 cost : {}".format((datetime.datetime.now() - st).seconds))

# c.construct(40000)
#
# st = datetime.datetime.now()
# gmm.fit_weight(c.coreset, None, filename="test_coreset_birch_10000", origin_X=X, plot=True, weight=c.weight)
# print("40000 cost : {}".format((datetime.datetime.now() - st).seconds))
#
# kmc = KMeans(n_clusters=100, max_iter=50)
# t = timer()
# kmc.fit(c.coreset, sample_weight=c.weight)
# print("K-means cluster 40000 fit time : {:.2f}s".format(timer()-t))
#
# c.construct(50000)
#
# st = datetime.datetime.now()
# gmm.fit_weight(c.coreset, None, filename="test_coreset_birch_10000", origin_X=X, plot=True, weight=c.weight)
# print("50000 cost : {}".format((datetime.datetime.now() - st).seconds))
#
# kmc = KMeans(n_clusters=100, max_iter=50)
# t = timer()
# kmc.fit(c.coreset, sample_weight=c.weight)
# print("K-means cluster 50000 fit time : {:.2f}s".format(timer()-t))
#
# c.construct(60000)
#
# st = datetime.datetime.now()
# gmm.fit_weight(c.coreset, None, filename="test_coreset_birch_10000", origin_X=X, plot=True, weight=c.weight)
# print("60000 cost : {}".format((datetime.datetime.now() - st).seconds))
#
# kmc = KMeans(n_clusters=100, max_iter=50)
# t = timer()
# kmc.fit(c.coreset, sample_weight=c.weight)
# print("K-means cluster 60000 fit time : {:.2f}s".format(timer()-t))
#
# c.construct(70000)
#
# st = datetime.datetime.now()
# gmm.fit_weight(c.coreset, None, filename="test_coreset_birch_10000", origin_X=X, plot=True, weight=c.weight)
# print("70000 cost : {}".format((datetime.datetime.now() - st).seconds))
#
# kmc = KMeans(n_clusters=100, max_iter=50)
# t = timer()
# kmc.fit(c.coreset, sample_weight=c.weight)
# print("K-means cluster 70000 fit time : {:.2f}s".format(timer()-t))
#
# c.construct(80000)
#
# st = datetime.datetime.now()
# gmm.fit_weight(c.coreset, None, filename="test_coreset_birch_10000", origin_X=X, plot=True, weight=c.weight)
# print("80000 cost : {}".format((datetime.datetime.now() - st).seconds))
#
# kmc = KMeans(n_clusters=100, max_iter=50)
# t = timer()
# kmc.fit(c.coreset, sample_weight=c.weight)
# print("K-means cluster 80000 fit time : {:.2f}s".format(timer()-t))

# st = datetime.datetime.now()
# gmm2.fit_weight(c.coreset, None, filename="test_coreset_birch_C110_I0", origin_X=X, plot=True, weight=c.weight)
# print("110 cost : {}".format((datetime.datetime.now() - st).seconds))
# st = datetime.datetime.now()
# gmm3.fit_weight(c.coreset, None, filename="test_coreset_birch_C130_I0", origin_X=X, plot=True, weight=c.weight)
# print("130 cost : {}".format((datetime.datetime.now() - st).seconds))
# st = datetime.datetime.now()
# gmm4.fit_weight(c.coreset, None, filename="test_coreset_birch_C200_I0", origin_X=X, plot=True, weight=c.weight)
# print("200 cost : {}".format((datetime.datetime.now() - st).seconds))
# st = datetime.datetime.now()
# gmm5.fit_weight(c.coreset, None, filename="test_coreset_CIFAR_C500_I0", origin_X=X, plot=False, weight=c.weight)
# print("500 cost : {}".format((datetime.datetime.now() - st).seconds))
# st = datetime.datetime.now()
# gmm6.fit_weight(c.coreset, None, filename="test_coreset_birch_C1000_I0", origin_X=X, plot=True, weight=c.weight)
# print("1000 cost : {}".format((datetime.datetime.now() - st).seconds))
# fit full dataset
# st = datetime.datetime.now()
# gmm.fit(X, None, filename="test_birch", origin_X=X, plot=True)
# print("cost : {}".format((datetime.datetime.now() - st).seconds))



# # Clustering via k-means on coreset
# kmc = KMeans(n_clusters=100, max_iter=50)
# t = timer()
# kmc.fit(Xc, sample_weight=c.weight)
# print("K-means cluster 100 fit time : {:.2f}s".format(timer()-t))
# kmc = KMeans(n_clusters=110, max_iter=50)
# t = timer()
# kmc.fit(Xc, sample_weight=c.weight)
# print("K-means cluster 110 fit time : {:.2f}s".format(timer()-t))
# kmc = KMeans(n_clusters=150, max_iter=50)
# t = timer()
# kmc.fit(Xc, sample_weight=c.weight)
# print("K-means cluster 150 fit time : {:.2f}s".format(timer()-t))
# kmc = KMeans(n_clusters=200, max_iter=50)
# t = timer()
# kmc.fit(Xc, sample_weight=c.weight)
# print("K-means cluster 200 fit time : {:.2f}s".format(timer()-t))
# kmc = KMeans(n_clusters=500, max_iter=50)
# t = timer()
# kmc.fit(Xc, sample_weight=c.weight)
# print("K-means cluster 500 fit time : {:.2f}s".format(timer()-t))
# kmc = KMeans(n_clusters=1000, max_iter=50)
# t = timer()
# kmc.fit(Xc, sample_weight=c.weight)
# print("K-means cluster 1000 fit time : {:.2f}s".format(timer()-t))

# # Clustering via k-means on full dataset
# km = KMeans(n_clusters=100, max_iter=50)
# t = timer()
# km.fit(X)
# print("K-means on full data set fit time : {:.2f}s".format(timer()-t))
# # predict the cluster
# t = timer()
# labels1 = kmc.predict(X)
# print("K-means coreset predict X : {:.2f}".format(timer()-t))

# t = timer()
# labels2 = km.predict(X)
# print("K-means predict X : {:.2f}".format(timer()-t))

# print("Start plotting k-means-coreset")
# # plot the k-means coreset clustering result
# fig = plt.figure(figsize=(10, 10), dpi=80)
# for i in range(X.shape[0]):
#     plt.scatter(X[i, 0], X[i, 1], s=1, color=colors[labels1[i]])
#
# pylab.ylim([0, 1000000])
# pylab.xlim([0, 1000000])
# ax = plt.gca()
# try:
#     ax.set_facecolor('gainsboro')
# except:
#     ax.set_axis_bgcolor('gainsboro')
# pylab.savefig('k-means-coreset' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") +'.png')
# plt.close(fig)
# print("k-means-coreset complete")

# print("Start plotting k-means-full")
# # plot the k-means full dataset clustering result
# fig = plt.figure(figsize=(10, 10), dpi=80)
# for i in range(X.shape[0]):
#     plt.scatter(X[i, 0], X[i, 1], s=1, color=colors[labels2[i]])
#
# pylab.ylim([0, 1000000])
# pylab.xlim([0, 1000000])
# ax = plt.gca()
# try:
#     ax.set_facecolor('gainsboro')
# except:
#     ax.set_axis_bgcolor('gainsboro')
# pylab.savefig('k-means-full' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") +'.png')
# plt.close(fig)
# print("k-means-full complete")
