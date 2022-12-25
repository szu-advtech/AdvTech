import torch
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as dsets

# partial_rate = 0.1
#
# def generate_uniform_cv_candidate_labels(train_labels, partial_rate=0.1):
#     '''yjuny:统一概率均匀生成候选标签'''
#     if torch.min(train_labels) > 1:
#         raise RuntimeError('testError')
#     elif torch.min(train_labels) == 1:
#         train_labels = train_labels - 1
#
#     # yjuny:K为类别数量，n为样本数量
#     K = int(torch.max(train_labels) - torch.min(train_labels) + 1)
#     n = train_labels.shape[0]
#     print(f'K:{K}, n:{n}')
#
#     partialY = torch.zeros(n, K)
#     # yjuny:随机生成混淆标签
#     partialY[torch.arange(n), train_labels] = 1.0
#     print(f'partial_Y before:{partialY}')
#     # yjuny:转移矩阵
#     transition_matrix = np.eye(K)
#
#     # yjuny:~np.eye(K)表示反转，且dtype=bool把0/1变为False/True np.where(condition)返回满足条件的坐标
#     transition_matrix[np.where(~np.eye(transition_matrix.shape[0], dtype=bool))] = partial_rate
#     print(transition_matrix)
#
#     random_n = np.random.uniform(0, 1, size=(n, K))
#     print(f'random_n:{random_n}, random_n.shape:{random_n.shape}')
#
#     for j in range(n):  # for each instance
#         partialY[j, :] = torch.from_numpy((random_n[j, :] < transition_matrix[train_labels[j], :]) * 1)
#     print(f'11111111111:{transition_matrix[train_labels[0], :]}')
#
#     print("Finish Generating Candidate Label Sets!\n")
#     print(f'partialY:{partialY}')
#     return partialY
#
#
# batch_size = 256
# test_transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
#
# temp_train = dsets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
# data, labels = temp_train.data, torch.Tensor(temp_train.targets).long()
# print(f'labels:{labels}')
# # get original data and labels
#
# # test_dataset = dsets.CIFAR10(root='./data', train=False, transform=test_transform)
# # test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size * 4, shuffle=False, num_workers=4,
# #                                           sampler=torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False))
# # set test dataloader
#
# partialY = generate_uniform_cv_candidate_labels(labels, partial_rate)

# c = torch.Tensor([2, 5, 6, 9, 1])
# b = torch.zeros(5)
# a = torch.ones(5).bool()
# sorted_idx = torch.argsort(c)
# print(a, b.shape[0], sorted_idx)


import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
def Pico_TSNE(data, target):
    """
    画出特征投影图
    :param data:
    :return:
    """
    t_sne_features = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(data)
    plt.scatter(x=t_sne_features[:, 0], y=t_sne_features[:, 1], c=target, cmap='jet', s=2)
    plt.savefig('tsne.pdf', dpi=800)
    plt.show()


exp_dir = '/home/yangjy/Research/CL+KD/PICO/PiCO-main/experiment/PiCO-CIFAR-10-Noisy-test/dscifar10p0.1n0.2_ps50_lw0.5_pm0.99_he_False_sel0.6_k5s100_uw0.1_sd123'
# exp_dir = 'D:/Research/CL+KD/PiCO/experiment'
with open(exp_dir + '/pseudo_labels.txt', 'r') as f:
    p_lines = f.readlines()
    print(len(p_lines))
with open(exp_dir + '/labels.txt', 'r') as f:
    l_lines = f.readlines()
x_list = []
y_list = []
for line in p_lines:
    line = line.strip('\t')
    line = line.strip('\n')
    line = line.strip('')
    tem_list = []
    for x in line.split('\t'):
        if x is not '':
            tem_list.append(x)
    # tem_list = np.array(tem_list)
    if len(tem_list) is 128:
        x_list.append(np.array(tem_list))

for target in l_lines:
    y_list.append(int(target))

# print(x_list)
# print(y_list)

x_list = np.array(x_list)
y_list = np.array(y_list)
print('target:', y_list, y_list.shape)
print('data:', x_list, x_list.shape)

Pico_TSNE(x_list, y_list)
print('end')
