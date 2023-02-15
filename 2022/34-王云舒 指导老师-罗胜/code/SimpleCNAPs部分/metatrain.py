#!/usr/bin/env python3
import pickle
import random
import numpy as np
import torch
from torch import nn, optim
from torch.optim import optimizer
import tqdm
import benchmarks as benchmarks

from src.normalization_layers import TaskNormI
# from src.utils import ValidationAccuracies, loss, aggregate_accuracy
# from src.model import Cnaps
from src.utils import aggregate_accuracy
from simple_cnaps_src.simple_cnaps_l1 import SimpleCnaps
import os
import argparse
import algorithms
import torch.nn.functional as F
import time
import xlwt
from path_index import *

def register_extra_parameters(model):
    for module in model.modules():
        if isinstance(module, TaskNormI):
            module.register_extra_weights()


def init_model(device, use_two_gpus):
    # use_two_gpus = False
    model = SimpleCnaps(device=device, use_two_gpus=use_two_gpus).to(device)
    # model.classifier = distLinear(512,26)
    # model.load_state_dict(torch.load('./trained_model/final/Nostr_simplecnaps_resnet18_3shot_ours_rseed100.pth'))
    # register_extra_parameters(model)
    print(model)
    # set encoder is always in train mode (it only sees context data).
    model.train()
    model.feature_extractor.eval()  #####

    if use_two_gpus:
        model.distribute_model()
    return model


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def lossfn(test_logits_sample, test_labels, device):
    size = test_logits_sample.size()
    sample_count = size[0]  # scalar for the loop counter
    num_samples = torch.tensor([sample_count], dtype=torch.float, device=device, requires_grad=False)

    log_py = torch.empty(size=(size[0], size[1]), dtype=torch.float, device=device)
    for sample in range(sample_count):
        log_py[sample] = -F.cross_entropy(test_logits_sample[sample], test_labels, reduction='none')
    score = torch.logsumexp(log_py, dim=0) - torch.log(num_samples)
    return -torch.sum(score, dim=0)


def aggregate_accuracy(test_logits_sample, test_labels):
    averaged_predictions = torch.logsumexp(test_logits_sample, dim=0)
    # print(averaged_predictions)
    return torch.mean(torch.eq(test_labels, torch.argmax(averaged_predictions, dim=-1)).float())


def fast_adapt(batch, learner, loss, adaptation_steps, shots, ways, device, mode, use_two_gpus, scale):
    data, labels, users, tasks = batch
    data, labels = data.to(device), labels.to(device)   
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots * ways) * 2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    # Adapt the model
    if mode == 'train':

        for step in range(adaptation_steps):
            logits = learner(adaptation_data, adaptation_labels, evaluation_data)
            # dot = make_dot(logits.mean(),params=dict(learner.named_parameters()))
            # dot.save('coscnaps.gv')
            # dot.render('coscnaps.gv',view=True)
            adaptation_error = lossfn(logits, evaluation_labels, device)
            # adaptation_error = loss(logits, evaluation_labels)

            if use_two_gpus:
                regularization_term = (learner.feature_adaptation_network.regularization_term()).cuda(0)
            else:
                regularization_term = (learner.feature_adaptation_network.regularization_term())

            regularizer_scaling = scale
            adaptation_error += regularizer_scaling * regularization_term
            adaptation_accuracy = aggregate_accuracy(logits, evaluation_labels)
            adaptation_error = adaptation_error / 16
            # adaptation_accuracy = accuracy(logits, evaluation_labels)
            adaptation_error.backward(retain_graph=False)
    else:
        with torch.no_grad():
            logits = learner(adaptation_data, adaptation_labels, evaluation_data)

            adaptation_error = lossfn(logits, evaluation_labels, device)
            adaptation_error = adaptation_error / 16
            adaptation_accuracy = aggregate_accuracy(logits, evaluation_labels)

    return adaptation_error, adaptation_accuracy


def cnaps_test(testset, learner, device, users_size = 10):
    preds = []
    labs = []
    good = []
    goodall = 0
    though = 0
    tasks_size = len(testset.dataset.tasks_to_indices.keys())
    for i in range(tasks_size):
        preds.append([])
        labs.append([])
        good.append(0)

    for i in tqdm.tqdm(range(tasks_size)):
        # meta test train
        k = 1
        tpset = testset.sample()
        while k == 1:
            if i not in tpset[3]:
                tpset = testset.sample()
            else:
                k = 0
        data, labels, users, tasks = tpset
        sourcedata, sourcelabels = data.to(device), labels.to(device)
        go = time.time()
        task_good = 0
        task_all = 0
        for j in range(len(testset.dataset)):
            abc = testset.dataset[j]
            a, b, c, d = abc
            if i != d:
                continue
            else:
                a = torch.unsqueeze(a, 0)
            task_all += 1
            label = torch.tensor(b, dtype=torch.long)
            # label1 = torch.tensorb
            # label = testset.dataset.indices_to_labels[i]
            a, label = a.to(device), label.to(device)
            predictions = learner(sourcedata, sourcelabels, a)
            averaged_predictions = torch.logsumexp(predictions, dim=0)
            predictions = torch.argmax(averaged_predictions, dim=-1)
            # predictions = predictions.argmax(dim=1)

            # averaged_predictions = torch.logsumexp(predictions, dim=0)
            # # print(averaged_predictions)
            # predictions = torch.argmax(averaged_predictions, dim=-1)
            # print(predictions)
            tempi = testset.dataset.indices_to_users[j]
            preds[tempi].append(predictions)
            labs[tempi].append(label)
            # preds.append(predictions)
            # labs.append(label)
            if predictions == label:
                task_good += 1
                goodall += 1
            # preds.append(predictions)
            # labs.append(label)
            # if predictions == label:
            #     good += 1
        task_acc = float(task_good)/task_all
        #print('task-'+str(testset.dataset.dataset.task_names[i])+"-acc:"+str(task_acc))
        though += (time.time() - go)


    allacc = float(goodall) / len(testset.dataset)
    print('all acc: ', allacc)
    print('All testing complete in {:.0f}s'.format(though))
    return allacc, though


def main(
        ways=metatrain_trainways,
        shots=metatrain_trainshots,
        meta_lr=5e-4,
        meta_batch_size=16,
        adaptation_steps=1,
        num_iterations=696, # maybe bigger, iter
        seed=42,  # 42 24 100 111
        use_two_gpus=False,
        different='scale',
        dir='ablation',
        scale=0.0001,
):
    nways = str(ways)
    nshots = str(shots)
    nrseed = str(seed)
    nscale = str(scale)
    nscale = nscale.replace('.', '_')
    #destination = "./trained_model/results/model_resnet18_3_" + nways + "ways_" + nshots + "shots_" + nrseed + "seed" + '_2.pth'
    destination = metatrain_destination
    #print("destination:", destination)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # device = torch.device('cuda')
    # os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    os.environ["CUDA_VISIBLE_DEVICES"] = '1,3'
    device = torch.device('cuda')
    model = init_model(device, use_two_gpus)

    tasksets = benchmarks.get_tasksets('digits',
                                       train_samples=2 * shots,
                                       train_ways=ways,
                                       test_samples=2 * shots,
                                       test_ways=26,
                                       root='~/data',
                                       )

    opt = optim.Adam(model.parameters(), meta_lr)
    loss = nn.CrossEntropyLoss(reduction='mean')

    min_loss = 10000
    count = 0
    # testlist = []
    # for k in range(4):
    #     testlist.append([])
    # bestacc = 0
    # besterr = 10000
    allsince = time.time()
    for iteration in range(num_iterations):
        since = time.time()
        opt.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        # meta_test_error = 0.0
        # meta_test_accuracy = 0.0
        for task in range(meta_batch_size):
            torch.set_grad_enabled(True)
            # Compute meta-training loss
            # learner = maml.clone()
            batch = tasksets.train.sample()
            # print(batch)
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               model,
                                                               loss,
                                                               adaptation_steps,
                                                               shots,
                                                               ways,
                                                               device,
                                                               'train',
                                                               use_two_gpus,
                                                               scale)
            # evaluation_error.backward()
            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()
            # Compute meta-validation loss
            # learner = maml.clone()
            batch = tasksets.validation.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               model,
                                                               loss,
                                                               adaptation_steps,  # 3steps
                                                               shots,
                                                               ways,
                                                               device,
                                                               'valid',
                                                               use_two_gpus,
                                                               scale)
            meta_valid_error += evaluation_error.item()
            meta_valid_accuracy += evaluation_accuracy.item()

        print('\n')
        print('Iteration', iteration)
        print('Meta Train Error', meta_train_error / meta_batch_size)
        print('Meta Train Accuracy', meta_train_accuracy / meta_batch_size)
        print('Meta Valid Error', meta_valid_error / meta_batch_size)
        print('Meta Valid Accuracy', meta_valid_accuracy / meta_batch_size)
        opt.step()

        mve = meta_valid_error / meta_batch_size
        if min_loss >= mve:
            min_loss = mve
            count = 0
            # ckpt = model.state_dict()
        else:
            # min_loss = mve
            count = count + 1
        if count > 100:
            break

    allend = time.time() - allsince
    print('All training complete in {:.0f}m {:.0f}s'.format(allend // 60, allend % 60))
    torch.save(model.state_dict(), destination)
    tasksets = benchmarks.get_tasksets('digits',
                                       train_samples=2 * shots,
                                       train_ways=ways,
                                       test_samples=shots,
                                       test_ways=26,
                                       root='~/data',
                                       )

    model.eval()
    with torch.no_grad():
        allacc, though = cnaps_test(tasksets.test, model, device)
    return allacc, though, allend


if __name__ == '__main__':
    shots = [2]
    #seeds = [42,24,100]
    seeds = [42]
    book = xlwt.Workbook(encoding='utf-8',style_compression=0)
    for i in range(len(shots)):
        for j in range(len(seeds)):
            allacc, though, allend = main(shots=shots[i],seed=seeds[j])
            
