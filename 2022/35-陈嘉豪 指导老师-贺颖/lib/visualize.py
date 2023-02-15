from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from options import args_parser
import torch
import random

def visualize(args, x, y):
    print("Begin visualization ...")
    colors = ['#000000', 'peru', '#FF8C00', 'gold', 'lightseagreen', 'royalblue', 'darkseagreen', 'violet', 'palevioletred', 'g']
    S_data = np.hstack((x, y))

    S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1], 'label': S_data[:, 2]})

    for class_index in range(args.num_classes):
        X = S_data.loc[S_data['label'] == class_index]['x']
        Y = S_data.loc[S_data['label'] == class_index]['y']
        plt.scatter(X, Y, marker='.', color=colors[class_index], alpha=0.08)

    plt.subplots_adjust(left=None, bottom=0.15, right=None, top=None, wspace=0.1, hspace=0.15)

    plt.savefig("./protos_"+args.alg+".pdf", format=' pdf', dpi=600)

args = args_parser()
args.alg = 'fedproto'

args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
if args.device == 'cuda':
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
else:
    torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

x = np.load('../exps/' + args.alg + '_protos.npy', allow_pickle=True)
y = np.load('../exps/' + args.alg + '_labels.npy', allow_pickle=True)

tsne = TSNE()
x = tsne.fit_transform(x)

y = y.reshape((-1, 1))
visualize(args, x, y)
