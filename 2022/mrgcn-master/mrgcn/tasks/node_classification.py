#!/usr/bin/env python

import logging
from time import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim

from mrgcn.data.batch import FullBatch, MiniBatch
from mrgcn.data.utils import getConfParam
from mrgcn.encodings.graph_features import (construct_features,
                                            isDatatypeIncluded,
                                            getDatatypeConfig)
from mrgcn.models.mrgcn import MRGCN
from mrgcn.models.utils import getPadSymbol
from mrgcn.tasks.utils import EarlyStop, optimizer_params


logger = logging.getLogger()


def run(A, X, Y, X_width, tsv_writer, config,
        modules_config, optimizer_config, featureless,
        test_split, checkpoint):
    tsv_writer.writerow(["epoch", "training_loss", "training_accurary",
                                  "validation_loss", "validation_accuracy",
                                  "test_loss", "test_accuracy"])

    # compile model
    model = build_model(X_width, Y, A, modules_config, config, featureless)
    opt_params = optimizer_params(model, optimizer_config)
    optimizer = optim.Adam(opt_params,
                           lr=config['model']['learning_rate'],
                           weight_decay=config['model']['weight_decay'])
    criterion = nn.MSELoss()

    # train model
    nepoch = config['model']['epoch']
    batchsize = config['task']['batchsize']
    l1_lambda = config['model']['l1_lambda']
    l2_lambda = config['model']['l2_lambda']
    
    # early stopping
    patience = config['task']['early_stopping']['patience']
    tolerance = config['task']['early_stopping']['tolerance']
    early_stop = EarlyStop(patience, tolerance) if patience > 0 else None

    # get pad symbol in case of language model(s)
    pad_symbol_map = dict()
    for datatype in ["xsd.string", "xsd.anyURI"]:
        if isDatatypeIncluded(config, datatype):
            feature_config = getDatatypeConfig(config, datatype)
            if feature_config is None\
               or 'tokenizer' not in feature_config.keys():
                continue

            pad_symbol = getPadSymbol(feature_config['tokenizer'])
            pad_symbol_map[datatype] = pad_symbol

    epoch = 0
    if checkpoint is not None:
        print("[LOAD] Loading model state", end='')
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']

        print(f" - {epoch} epoch")

    # Log wall-clock time
    t0 = time()
    for result in train_model(A, model, optimizer, criterion, X, Y, epoch,
                             nepoch, test_split, batchsize, l1_lambda,
                             l2_lambda, pad_symbol_map, early_stop):
        # log metrics
        tsv_writer.writerow([str(result[0]),
                             str(result[1]),
                             str(result[2]),
                             str(result[3]),
                             str(result[4]),
                             "-1", "-1"])

    logging.info("Training time: {:.2f}s".format(time()-t0))

    # test model
    loss, acc, labels, targets = test_model(A, model, criterion, X, Y, 
                                           test_split, batchsize,
                                           pad_symbol_map)
    # log metrics
    tsv_writer.writerow(["-1", "-1", "-1", "-1", "-1",
                         str(loss), str(acc)])
        
    model = model.to('cpu')

    return (model, optimizer, epoch+nepoch, loss, acc, labels, targets)

def train_model(A, model, optimizer, criterion, X, Y, epoch, nepoch,
                test_split, batchsize, l1_lambda, l2_lambda, pad_symbol_map,
                early_stop):
    Y_train = Y['train']
    Y_valid = Y['valid']
    if test_split == "test":
        # merge training and validation sets
        ri = np.concatenate([Y_train.nonzero()[0], Y_valid.nonzero()[0]])
        ci = np.concatenate([Y_train.nonzero()[1], Y_valid.nonzero()[1]])
        d = np.concatenate([Y_train.data, Y_valid.data])

        Y_train = sp.csr_matrix((d, (ri, ci)))
        Y_valid = None

    # generate batches
    num_layers = model.rgcn.num_layers
    train_batches = mkbatches(A, X, Y_train, batchsize, num_layers)
    for batch in train_batches:
        batch.pad_(pad_symbols=pad_symbol_map)
        batch.to_dense_()
        batch.as_tensors_()
        batch.to(model.devices)
    num_batches_train = len(train_batches)

    valid_batches = list()
    if Y_valid is not None:
        valid_batches = mkbatches(A, X, Y_valid, batchsize, num_layers)
        for batch in valid_batches:
            batch.pad_(pad_symbols=pad_symbol_map)
            batch.to_dense_()
            batch.as_tensors_()
            batch.to(model.devices)

    logging.info("Training for {} epoch".format(nepoch))
    for epoch in range(epoch+1, nepoch+epoch+1):
        if early_stop is not None and early_stop.stop:
            logging.info("Stopping early after %d epoch" % (epoch-1))
            model.load_state_dict(early_stop.best_weights)
            optimizer.load_state_dict(early_stop.best_optim)

            break

        model.train()

        loss_lst = list()
        acc_lst = list()
        for batch_id, batch in enumerate(train_batches, 1):
            batch_str = " [TRAIN] - batch %2.d / %d" % (batch_id,
                                                        num_batches_train)
            print(batch_str, end='\b'*len(batch_str), flush=True)

            batch_node_idx = batch.node_index

            # Training scores
            Y_batch_hat = model(batch).to('cuda')
            Y_batch_train = Y_train[batch_node_idx]

            batch_loss = categorical_crossentropy(Y_batch_hat, Y_batch_train, criterion)
            batch_acc = categorical_accuracy(Y_batch_hat, Y_batch_train)[0]

            if l1_lambda > 0:
                l1_regularization = torch.tensor(0.)
                for name, param in model.named_parameters():
                    if 'weight' not in name:
                        continue
                    l1_regularization += torch.sum(param.abs())

                batch_loss += l1_lambda * l1_regularization

            if l2_lambda > 0:
                l2_regularization = torch.tensor(0.)
                for name, param in model.named_parameters():
                    if 'weight' not in name:
                        continue
                    l2_regularization += torch.sum(param ** 2)

                batch_loss += l2_lambda * l2_regularization

            optimizer.zero_grad()
            batch_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # cast criterion objects to floats to free the memory of the tensors
            # they point to
            batch_loss = float(batch_loss)
            batch_acc = float(batch_acc)

            loss_lst.append(batch_loss)
            acc_lst.append(batch_acc)

        train_loss = np.mean(loss_lst)
        train_acc = np.mean(acc_lst)

        val_loss = -1
        val_acc = -1
        if Y_valid is not None:
            val_loss, val_acc = eval_model(model, valid_batches, Y_valid,
                                           criterion)

            logging.info("{:04d} ".format(epoch) \
                         + "| train loss {:.4f} / acc {:.4f} ".format(train_loss,
                                                                      train_acc)
                         + "| val loss {:.4f} / acc {:.4f}".format(val_loss,
                                                                   val_acc))

            if early_stop is not None:
                early_stop.record(val_loss, model, optimizer)
        else:
            logging.info("{:04d} ".format(epoch) \
                         + "| train loss {:.4f} / acc {:.4f} ".format(train_loss,
                                                                      train_acc))

        yield (epoch,
               train_loss, train_acc,
               val_loss, val_acc)

def eval_model(model, valid_batches, Y_valid, criterion):
    model.eval()

    loss_lst = list()
    acc_lst = list()
    num_batches_valid = len(valid_batches)
    for batch_id, batch in enumerate(valid_batches, 1):
        batch_str = " [VALID] - batch %2.d / %d" % (batch_id, num_batches_valid)
        print(batch_str, end='\b'*len(batch_str), flush=True)

        batch_node_idx = batch.node_index
        with torch.no_grad():
            Y_batch_hat = model(batch).to('cpu')
            Y_batch_valid = Y_valid[batch_node_idx]

            batch_loss = categorical_crossentropy(Y_batch_hat, Y_batch_valid, criterion)
            batch_acc = categorical_accuracy(Y_batch_hat, Y_batch_valid)[0]

        batch_loss = float(batch_loss)
        batch_acc = float(batch_acc)
    
        loss_lst.append(batch_loss)
        acc_lst.append(batch_acc)

    val_loss = np.mean(loss_lst)
    val_acc = np.mean(acc_lst)

    return (val_loss, val_acc)
 
def test_model(A, model, criterion, X, Y, test_split, batchsize,
               pad_symbol_map):
    model.eval()
    Y_test = Y[test_split]

    loss_lst = list()
    acc_lst = list()
    label_lst = list()
    target_lst = list()

    # generate batches
    num_layers = model.rgcn.num_layers
    test_batches = mkbatches(A, X, Y_test, batchsize, num_layers)
    for batch in test_batches:
        batch.pad_(pad_symbols=pad_symbol_map)
        batch.to_dense_()
        batch.as_tensors_()
        batch.to(model.devices)
    num_batches_test = len(test_batches)

    for batch_id, batch in enumerate(test_batches, 1):
        batch_str = " [%s] - batch %2.d / %d" % (test_split.upper(),
                                                 batch_id,
                                                 num_batches_test)
        print(batch_str, end='\b'*len(batch_str), flush=True)

        node_idx = batch.node_index
        with torch.no_grad():
            Y_batch_hat = model(batch).to('cpu')
            Y_batch_test = Y_test[node_idx]
    
            batch_loss = categorical_crossentropy(Y_batch_hat, Y_batch_test, criterion)
            batch_acc, batch_labels, batch_targets = categorical_accuracy(Y_batch_hat, Y_batch_test)

        batch_loss = float(batch_loss)
        batch_acc = float(batch_acc)
    
        loss_lst.append(batch_loss)
        acc_lst.append(batch_acc)
        label_lst.append(batch_labels)
        target_lst.append(batch_targets)

    loss = np.mean(loss_lst)
    acc = np.mean(acc_lst)
    labels = np.concatenate(label_lst)
    targets = np.concatenate(target_lst)

    logging.info("Performance on {} set: loss {:.4f} / accuracy {:.4f}".format(
                  test_split,
                  loss,
                  acc))

    return (loss, acc, labels, targets)

def build_dataset(knowledge_graph, nodes_map, target_triples, config, featureless):
    logger.debug("Starting dataset build")
    # generate target matrix
    Y, sample_map, class_map = mk_target_matrices(target_triples, nodes_map)

    if featureless:
        F = dict()
    else:
        separate_literals = config['graph']['structural']['separate_literals']
        F = construct_features(nodes_map, knowledge_graph,
                               config['graph']['features'],
                               separate_literals)

    logger.debug("Completed dataset build")

    return (F, Y, sample_map, class_map)

def mkbatches(A, X, Y, batchsize, num_layers):
    num_samples = len(Y.data)
    if batchsize <= 0:
        # full batch mode requested by user
        batchsize = num_samples

    batch_slices = [slice(begin, min(begin+batchsize, num_samples))
                   for begin in range(0, num_samples, batchsize)]
    batches = list()
    if len(batch_slices) > 1:
        # mini batch mode
        sample_idx = Y.nonzero()[0]  # global indices of labelled nodes
        for slce in batch_slices:
            batch_node_idx = sample_idx[slce]
            batch = MiniBatch(A, X, batch_node_idx, num_layers)

            batches.append(batch)
    else:
        batch_node_idx = np.arange(Y.shape[0])
        batch = FullBatch(A, X, batch_node_idx)
        batches.append(batch)

    return batches

def mk_target_matrices(target_triples, nodes_map):
    classes = {str(c) for split in target_triples.values() for _,_,c in split} # unique classes
    logger.debug("Target classes ({}): {}".format(len(classes), classes))

    # node/class label to integers; sorted to make consistent over runs
    class_map = sorted(list(classes))
    class_map_inv = {label:i for i,label in enumerate(class_map)}

    # note: by converting targets to strings we lose datatype info, but the use
    # cases where this would matter would be very limited 
    num_nodes = len(nodes_map)
    num_classes = len(class_map)
    sample_map = dict()
    Y = dict()
    for k in target_triples.keys():
        split = sorted(target_triples[k])

        logger.debug("Found {} instances ({})".format(len(split), k))
        target_pair_indices = list()
        sample_map[k] = list()
        for x, _, y in split:
            target_pair_indices.append((nodes_map[x], class_map_inv[str(y)]))
            sample_map[k].append(x)

        rows, cols = map(np.array, zip(*target_pair_indices))
        data = np.ones(len(rows), dtype=np.int8)  # assume <= 128 classes
        Y[k] = sp.csr_matrix((data, (rows, cols)),
                             shape=(num_nodes, num_classes),
                             dtype=np.int8)

    return (Y, sample_map, class_map)

def build_model(X_width, Y, A, modules_config, config, featureless):
    layers = config['model']['layers']
    assert len(layers) >= 2
    logger.debug("Starting model build")

    gcn_gpu_acceleration = getConfParam(config,
                                        "task.gcn_gpu_acceleration",
                                        False)

    # get sizes from dataset
    num_nodes, Y_dim = Y['train'].shape
    num_relations = int(A.shape[1]/num_nodes)

    modules = list()
    # input layer
    modules.append((X_width,
                    layers[0]['hidden_nodes'],
                    layers[0]['type'],
                    nn.ReLU()))

    # intermediate layers (if any)
    i = 1
    for layer in layers[1:-1]:
        modules.append((layers[i-1]['hidden_nodes'],
                        layer['hidden_nodes'],
                        layers[i-1]['type'],
                        nn.ReLU()))

        i += 1

    # output layer
    modules.append((layers[i-1]['hidden_nodes'],
                    Y_dim,
                    layers[i-1]['type'],
                    None))

    model = MRGCN(modules, modules_config, num_relations, num_nodes,
                  num_bases=config['model']['num_bases'],
                  p_dropout=config['model']['p_dropout'],
                  featureless=featureless,
                  bias=config['model']['bias'],
                  gcn_gpu_acceleration=gcn_gpu_acceleration)

    logger.debug("Completed model build")

    return model

def categorical_accuracy(Y_hat, Y):
    idx, targets = Y.nonzero()
    targets = torch.as_tensor(targets, dtype=torch.long)
    _, labels = Y_hat[idx].max(dim=1)

    return (torch.mean(torch.eq(labels, targets).float()), labels, targets)

def categorical_crossentropy(Y_hat, Y, criterion):
    idx, targets = Y.nonzero()
    targets = torch.as_tensor(targets, dtype=torch.long)
    predictions = Y_hat[idx]

    return criterion(predictions, targets)
