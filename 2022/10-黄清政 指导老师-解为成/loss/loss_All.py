import torch
import numpy as np


# Definition TripletMarginLoss : hard TripletMarginLoss
def vector_triplet_loss(embeding,labels,nums=7,alpha=0.2):
    # tensor is needed
    a = torch.FloatTensor([alpha]).cuda()
    loss = torch.FloatTensor([0.]).cuda()
    # get batchsize
    batch_size = labels.shape[0]
    for i in range(0, nums):
        anchor_list = []
        negative_list = []
        for L, E in zip(labels, embeding):
            if int(L.item()) == i:
                anchor_list.append(E)
            elif int(L.item()) != i:
                negative_list.append(E)
        if len(anchor_list) != 0 and len(negative_list) != 0:
            for anchor in anchor_list:
                max_list = []
                min_list = []
                for positive in anchor_list:
                    max_list.append((anchor - positive).square().sum())
                for negative in negative_list:
                    min_list.append((anchor - negative).square().sum())

                dap = max_list[0]
                for i in max_list:
                    if float(i.item()) > float(dap.item()):
                        dap = i
                dan = min_list[0]
                for j in min_list:
                    if float(j.item()) < float(dan.item()):
                        dan = j

                if (dap.item() - dan.item() + a.item()) > 0.:
                    loss += (dap - dan + a)

    # tensor can backpropagation
    return loss / batch_size

def matrix_triplet_loss(P,labels,nums=7, alpha=0.2):
    a = torch.FloatTensor([alpha]).cuda()
    loss = torch.FloatTensor([0.]).cuda()
    batch_size = labels.shape[0]
    for i in range(0, nums):
        anchor_list = []
        negative_list = []
        for label,Pic in zip(labels,P):
            if int(label) == i:
                anchor_list.append(Pic)
            else:
                negative_list.append(Pic)

        if len(anchor_list) > 0 and len(negative_list) > 0:
            for anchor in  anchor_list:
                P_max_list = []
                N_min_list = []
                for positive in anchor_list:
                    P_max_list.append((anchor - positive).square().sum())
                for negative in negative_list:
                    N_min_list.append((anchor - negative).square().sum())

                dap = P_max_list[0]
                for i in P_max_list:
                    if float(i.item()) > float(dap.item()):
                        dap = i
                dan = N_min_list[0]
                for j in N_min_list:
                    if float(j.item()) < float(dan.item()):
                        dan = j

                if (dap.item() - dan.item() + a.item()) > 0.:
                    loss += (dap - dan + a)

    return loss / batch_size






