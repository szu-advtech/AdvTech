#!/usr/bin/python3

import logging
from shutil import get_terminal_size
from time import time
import warnings

import numpy as np
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


logger = logging.getLogger(__name__)


def run(A, X, X_width, data, tsv_writer, 
        config, modules_config, optimizer_config, featureless, test_split,
        checkpoint):
    header = ["epoch", "loss"]
    for split in ["train", "valid", "test"]:
        header.extend([split+"_mrr_raw", split+"_H@1_raw", split+"_H@3_raw",
                       split+"_H@10_raw", split+"_mrr_flt", split+"_H@1_flt",
                       split+"_H@3_flt", split+"_H@10_flt"])
    tsv_writer.writerow(header)

    # used for clearing a line
    term_width = get_terminal_size().columns

    lp_gpu_acceleration = getConfParam(config,
                                        "task.lprank_gpu_acceleration",
                                        False)
    lp_device = torch.device("cpu")
    if lp_gpu_acceleration:
        if torch.cuda.is_available():
            lp_device = torch.device("cuda")
        else:
            warnings.warn("CUDA Resource not available", ResourceWarning)

    # get sizes from dataset
    # compile model
    model = build_model(X_width, A, modules_config, config, featureless)
    opt_params = optimizer_params(model, optimizer_config)
    optimizer = optim.Adam(opt_params,
                           lr=config['model']['learning_rate'],
                           weight_decay=config['model']['weight_decay'])
    criterion = nn.BCEWithLogitsLoss()

    # mini batching
    test_batchsize = int(config['task']['test_batchsize'])
    mrr_batchsize = int(config['task']['mrr_batchsize'])
    gcn_batchsize = int(config['task']['gcn_batchsize'])

    # train model
    nepoch = config['model']['epoch']
    eval_interval = config['task']['eval_interval']
    filter_ranks = config['task']['filter_ranks']
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

    # prepare splits
    if data is not None:
        if test_split == "test":
            # merge train and valid splits when testing
            data["train"] = np.concatenate([data["train"],
                                            data["valid"]],
                                           axis=0)
            data["valid"] = None  # save memory

    # Log wall-clock time
    t0 = time()
    loss = 0.0
    for result in train_model(A, X, data, model, optimizer,
                              criterion, epoch, nepoch, gcn_batchsize,
                              test_batchsize, mrr_batchsize, eval_interval,
                              filter_ranks, l1_lambda, l2_lambda,
                              pad_symbol_map, early_stop,
                              lp_device, term_width):

        epoch, loss, train_mrr, train_hits_at_k,\
                     valid_mrr, valid_hits_at_k = result

        result_str = [str(epoch), str(loss)]
        for mrr, hits in [(train_mrr, train_hits_at_k),
                          (valid_mrr, valid_hits_at_k)]:
            if mrr is None or hits is None:
                result_str.extend([-1, -1, -1, -1, -1, -1, -1, -1])
                continue

            result_str.extend([str(mrr['raw']), str(hits['raw'][0]),
                               str(hits['raw'][1]), str(hits['raw'][2]),
                               str(mrr['flt']), str(hits['flt'][0]),
                               str(hits['flt'][1]), str(hits['flt'][2])])

        # add test set placeholder
        result_str.extend([-1, -1, -1, -1, -1, -1, -1, -1])

        # log metrics
        tsv_writer.writerow(result_str)

    t1_str = "Training time: {:.2f}s".format(time()-t0)
    width_remain = term_width - len(t1_str)
    logging.info(t1_str + " " * (width_remain//2))          

    # Log wall-clock time
    t0 = time()

    # generate batches
    num_layers = model.rgcn.num_layers
    test_data = data[test_split]
    test_batches = mkbatches(A, X, test_data, gcn_batchsize, test_batchsize, num_layers)
    for i, (batch, batch_data) in enumerate(test_batches):
        batch.pad_(pad_symbols=pad_symbol_map)
        batch.to_dense_()
        batch.as_tensors_()
        batch.to_(model.devices)
        batch_data = torch.from_numpy(batch_data)
        test_batches[i] = (batch, batch_data)

    test_mrr, test_hits_at_k, test_ranks = test_model(test_batches,
                                                      model,
                                                      filter_ranks,
                                                      lp_device,
                                                      mrr_batchsize,
                                                      term_width)

    t1_str = "Testing time: {:.2f}s".format(time()-t0)
    width_remain = term_width - len(t1_str)
    logging.info(t1_str + " " * (width_remain//2))          

    # log metrics
    result_str = [-1 for _ in range(18)]
    result_str.extend([str(test_mrr['raw']),
                       str(test_hits_at_k['raw'][0]),
                       str(test_hits_at_k['raw'][1]),
                       str(test_hits_at_k['raw'][2]),
                       str(test_mrr['flt']),
                       str(test_hits_at_k['flt'][0]),
                       str(test_hits_at_k['flt'][1]),
                       str(test_hits_at_k['flt'][2])])
    tsv_writer.writerow(result_str)

    model = model.to('cpu')

    return (model, optimizer, epoch+nepoch, loss, test_mrr, test_hits_at_k, test_ranks)

def train_model(A, X, data, model, optimizer, criterion,
                epoch, nepoch, gcn_batchsize, test_batchsize,
                mrr_batchsize,
                eval_interval, filter_ranks, l1_lambda, l2_lambda,
                pad_symbol_map, early_stop,
                lp_device, term_width):

    # generate batches
    num_layers = model.rgcn.num_layers
    train_data = data["train"]
    train_batches = mkbatches(A, X, train_data, gcn_batchsize, test_batchsize, num_layers)
    for i, (batch, batch_data) in enumerate(train_batches):
        batch.pad_(pad_symbols=pad_symbol_map)
        batch.to_dense_()
        batch.as_tensors_()
        batch.to(model.devices)
        batch_data = torch.from_numpy(batch_data)
        train_batches[i] = (batch, batch_data)
    num_batches_train = len(train_batches)

    valid_batches = list()
    valid_data = data["valid"]
    if valid_data is not None:
        valid_batches = mkbatches(A, X, valid_data, gcn_batchsize, test_batchsize, num_layers)
        for i, (batch, batch_data) in enumerate(valid_batches):
            batch.pad_(pad_symbols=pad_symbol_map)
            batch.to_dense_()
            batch.as_tensors_()
            batch.to(model.devices)
            batch_data = torch.from_numpy(batch_data)
            valid_batches[i] = (batch, batch_data)

    logging.info("Training for {} epoch".format(nepoch))
    for epoch in range(epoch+1, nepoch+epoch+1):
        if early_stop is not None and early_stop.stop:
            logging.info("Stopping early after %d epoch" % (epoch-1))
            model.load_state_dict(early_stop.best_weights)
            optimizer.load_state_dict(early_stop.best_optim)

            break

        model.train()
        
        loss_lst = list()
        for batch_id, (batch, batch_data) in enumerate(train_batches, 1):
            batch_str = " [TRAIN] - batch %2.d / %d" % (batch_id, num_batches_train)
            print(batch_str, end='\b'*len(batch_str), flush=True)

            # number of triples in batch
            batch_num_samples = batch_data.shape[0]

            # node indices of all nodes in this batch.
            # these have been remapped to local indices in mini-batch mode
            batch_nodes = np.union1d(batch_data[:, 0],
                                     batch_data[:, 2])

            # sample negative triples by copying and corrupting positive triples
            ncorrupt = batch_num_samples//5  # corrupt 20%
            neg_samples_idx = np.random.choice(np.arange(batch_num_samples),
                                               ncorrupt,
                                               replace=False)

            ncorrupt_head = ncorrupt//2  # corrupt head and tail equally
            ncorrupt_tail = ncorrupt - ncorrupt_head
            corrupted_data = np.empty((ncorrupt, 3), dtype=int)

            # within-batch corruption to reduce the need to compute
            # the embeddings of out-of-batch nodes.
            # this should work fine as long as the batch size isn't too small.
            corrupted_data[:] = batch_data[neg_samples_idx]  # deep copy
            corrupted_data[:ncorrupt_head, 0] = np.random.choice(batch_nodes,
                                                                 ncorrupt_head)
            corrupted_data[-ncorrupt_tail:, 2] = np.random.choice(batch_nodes,
                                                                  ncorrupt_tail)

            # create labels; positive samples are 1, negative 0
            Y = torch.ones(batch_num_samples+ncorrupt, dtype=torch.float32)
            Y[-ncorrupt:] = 0

            # compute necessary embeddings
            node_embeddings = model(batch).to(lp_device)
            edge_embeddings = model.rgcn.relations.to(lp_device)
            
            # compute scores
            Y_hat = torch.empty((batch_num_samples+ncorrupt), dtype=torch.float32)

            batch_data_dev = torch.as_tensor(batch_data,
                                             device=lp_device).long()
            Y_hat[:batch_num_samples] = score_distmult_bc((batch_data_dev[:, 0],
                                                           batch_data_dev[:, 1],
                                                           batch_data_dev[:, 2]),
                                                          node_embeddings,
                                                          edge_embeddings).to('cpu')

            corrupted_data_dev = torch.as_tensor(corrupted_data,
                                                 device=lp_device).long()
            Y_hat[-ncorrupt:] = score_distmult_bc((corrupted_data_dev[:, 0],
                                                   corrupted_data_dev[:, 1],
                                                   corrupted_data_dev[:, 2]),
                                                  node_embeddings,
                                                  edge_embeddings).to('cpu')

            # clear gpu cache to save memory
            del node_embeddings
            del edge_embeddings
            del batch_data_dev
            del corrupted_data_dev
            for cuda_dev in range(torch.cuda.device_count()):
                torch.cuda.set_device(cuda_dev)
                torch.cuda.empty_cache()

            # compute loss
            optimizer.zero_grad()
            loss = binary_crossentropy(Y_hat, Y, criterion)

            if l1_lambda > 0:
                l1_regularization = torch.tensor(0.)
                for name, param in model.named_parameters():
                    if 'weight' not in name:
                        continue
                    l1_regularization += torch.sum(param.abs())

                loss += l1_lambda * l1_regularization

            if l2_lambda > 0:
                l2_regularization = torch.tensor(0.)
                for name, param in model.named_parameters():
                    if 'weight' not in name:
                        continue
                    l2_regularization += torch.sum(param ** 2)

                loss += l2_lambda * l2_regularization

            loss.backward()  # training loss
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss = float(loss)
            loss_lst.append(loss)
        
        loss = np.mean(loss_lst)
        results_str = f"{epoch:04d} | loss {loss:.4f}"

        train_mrr, train_hits_at_k = None, None
        valid_mrr, valid_hits_at_k = None, None
        if epoch % eval_interval == 0 or epoch == nepoch:
            train_mrr, train_hits_at_k, _  = test_model(train_batches,
                                                        model,
                                                        filter_ranks,
                                                        lp_device,
                                                        mrr_batchsize,
                                                        term_width)

            results_str += f" | train MRR {train_mrr['raw']:.4f} (raw)"
            if filter_ranks:
                results_str += f" / {train_mrr['flt']:.4f} (filtered)"

            valid_mrr = None
            valid_hits_at_k = None
            if valid_data is not None and epoch < nepoch:
                valid_mrr, valid_hits_at_k, _ = test_model(valid_batches,
                                                           model,
                                                           filter_ranks,
                                                           lp_device,
                                                           mrr_batchsize,
                                                           term_width)

                results_str += f" | valid MRR {valid_mrr['raw']:.4f} (raw) "
                if filter_ranks:
                    results_str += f" / flt {valid_mrr['flt']:.4f} (filtered)"
           
                if early_stop is not None:
                    early_stop.record(1.0 - valid_mrr['raw'],
                                      model, optimizer)
        else:
            width_remain = term_width - len(results_str)
            results_str += " " * (width_remain//2)  # ensure overwrite of batch message

        logging.info(results_str)

        yield (epoch, loss,
               train_mrr, train_hits_at_k,
               valid_mrr, valid_hits_at_k)

def test_model(batches, model, filter_ranks, lp_device,
               mrr_batchsize, term_width):
    model.eval()
    
    K = [1, 3, 10]
    hits_at_k = {"flt":[[] for _ in K], "raw":[[] for _ in K]}
    mrr = {"flt":[], "raw":[]}
    rankings = {"flt":[], "raw":[]}

    num_batches = len(batches)
    with torch.no_grad():
        for batch_id, (batch, batch_data) in enumerate(batches, 1):
            batch_str = " [MRGCN] - batch %2.d / %d" % (batch_id, num_batches)
            width_remain = term_width - len(batch_str)
            batch_str += " " * (width_remain//2)  # ensure overwrite of batch message
            print(batch_str, end='\b'*len(batch_str), flush=True)

            node_embeddings = model(batch).to(lp_device)
            edge_embeddings = model.rgcn.relations.to(lp_device)
            for filtered in [False, True]:
                rank_type = "flt" if filtered else "raw"
                if filtered is True and not filter_ranks:
                    mrr[rank_type].append(-1)
                    for i, _ in enumerate(K):
                        hits_at_k[rank_type][i].append(-1)
                    rankings[rank_type].append(-1)

                    continue

                ranks = compute_ranks_fast(batch_data,
                                           node_embeddings,
                                           edge_embeddings,
                                           mrr_batchsize,
                                           filtered)

                mrr[rank_type].append(torch.mean(1.0 / ranks.float()).item())
                for i, k in enumerate(K):
                    hits_at_k[rank_type][i].append(float(torch.mean((ranks <= k).float())))
                ranks = ranks.tolist()
                rankings[rank_type].append(ranks)

    # average MRR and hits@K over batches; flatten ranks into list
    for rank_type in ("flt", "raw"):
        mrr[rank_type] = np.mean(mrr[rank_type])
        hits_at_k[rank_type] = [np.mean(k) for k in hits_at_k[rank_type]]
        rankings[rank_type] = [r for r_list in rankings[rank_type] for r in r_list]

    return (mrr, hits_at_k, rankings)

def build_dataset(kg, nodes_map, config, featureless):
    logger.debug("Starting dataset build")
    if featureless:
        F = dict()
    else:
        separate_literals = config['graph']['structural']['separate_literals']
        F = construct_features(nodes_map, kg,
                               config['graph']['features'],
                               separate_literals)

    Y = torch.empty((0, 0), dtype=torch.float16)  # dummy

    logger.debug("Completed dataset build")

    return (F, Y)

def build_model(C, A, modules_config, config, featureless):
    layers = config['model']['layers']
    logger.debug("Starting model build")

    # get sizes from dataset
    X_dim = C  # == 0 if featureless
    num_nodes = A.shape[0]
    num_relations = int(A.shape[1]/num_nodes)

    modules = list()
    # input layer
    modules.append((X_dim,
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

    model = MRGCN(modules, modules_config, num_relations, num_nodes,
                  num_bases=config['model']['num_bases'],
                  p_dropout=config['model']['p_dropout'],
                  featureless=featureless,
                  bias=config['model']['bias'],
                  link_prediction=True)

    logger.debug("Completed model build")

    return model

def mkbatches(A, X, data, batchsize_mrgcn, batchsize_mrr, num_layers):
    """ Generate batches from node embeddings

        Prefer batches of nodes over batches of data to avoid exceeding the
        memory use of the MR-GCN in case of too many within-batch nodes. Split
        batches on the number of samples if this exceeds the parameter, to
        avoid mrr memory issues.

        This avoids the need to implement memory-reducing features during
        validation and testing, which often have a much smaller number of
        samples with the same number of nodes.
    """
    sample_nodes = np.union1d(data[:, 0], data[:, 2])  # all nodes in data
    num_nodes = len(sample_nodes)
    if batchsize_mrgcn <= 0:
        # full batch mode requested by user
        batchsize_mrgcn = num_nodes

    if batchsize_mrr <= 0:
        # full batch mode requested by user
        batchsize_mrr = data.shape[0]

    batch_slices = [slice(begin, min(begin+batchsize_mrgcn, num_nodes))
                   for begin in range(0, num_nodes, batchsize_mrgcn)]
    batches = list()
    if len(batch_slices) > 1:
        # mini batch mode
        for slce in batch_slices:
            batch_node_idx = sample_nodes[slce]

            # subset of 'data' that contains a batch node as head or tail.
            # the same triples can occur in at most 2 different batches.
            # prefer this over filtering repeated nodes as a) more data is
            # better and b) avoid (near) empty data in later batches.
            data_mask = ((np.in1d(data[:, 0], batch_node_idx))
                         | (np.in1d(data[:, 2], batch_node_idx)))
            batch_data = data[data_mask]

            # split 'batch_data' in smaller parts to avoid OOM issues when
            # computing their score
            num_samples = batch_data.shape[0]
            for subset in np.array_split(np.arange(num_samples),
                                         max(num_samples//batchsize_mrr, 1)):
                data_subset = np.copy(batch_data[subset])

                # extension of the sliced batch nodes which includes also all
                # connected nodes not part of the slice but which is still needed
                # to compute the batch data embeddings.
                subset_node_idx = np.union1d(data_subset[:, 0],
                                             data_subset[:, 2])

                # remap nodes to match embedding index
                # this is only needed when batching the nodes
                index_map = {v:i for i,v in enumerate(subset_node_idx)}
                data_subset[:, 0] = [index_map[int(i)] for i in data_subset[:, 0]]
                data_subset[:, 2] = [index_map[int(i)] for i in data_subset[:, 2]]

                batch = MiniBatch(A, X, subset_node_idx, num_layers)
                batches.append((batch, data_subset))
    else:
        num_samples = data.shape[0]
        for subset in np.array_split(np.arange(num_samples),
                                     max(num_samples//batchsize_mrr, 1)):
            data_subset = np.copy(data[subset])
            subset_node_idx = np.union1d(data_subset[:, 0],
                                         data_subset[:, 2])


            batch = FullBatch(A, X, subset_node_idx)
            batches.append((batch, data_subset))

    return batches

def binary_crossentropy(Y_hat, Y, criterion):
    # Y_hat := output of score()
    # Y := labels in [0, 1]
    # Y_hat[i] == Y[i] -> i is same triple
    return criterion(Y_hat, Y)


def filter_scores_(scores, data, heads, tails, head=True):
    # set scores of existing facts to -inf
    indices = list()
    for i, (s, p, o) in enumerate(data):
        s, p, o = (s.item(), p.item(), o.item())
        if head:
            indices.extend([(i, si) for si in heads[p, o] if si != s])
        else:
            indices.extend([(i, oi) for oi in tails[s, p] if oi != o])
        # we add the indices of all know triples except the one corresponding
        # to the target triples.

    if len(indices) <= 0:
        return

    indices = torch.tensor(indices)
    scores[indices[:, 0], indices[:, 1]] = float('-inf')


def truedicts(facts):
    heads = dict()
    tails = dict()
    for i in range(facts.shape[0]):
        fact = facts[i]
        s, p, o = fact[0].item(), fact[1].item(), fact[2].item()

        if (p, o) not in heads.keys():
            heads[(p, o)] = list()
        if (s, p) not in tails.keys():
            tails[(s, p)] = list()

        heads[(p, o)].append(s)
        tails[(s, p)].append(o)

    return heads, tails

def compute_ranks_fast(data, node_embeddings, edge_embeddings, mrr_batchsize,
                       filtered=True):
    true_heads, true_tails = truedicts(data) if filtered else (None, None)

    num_facts = data.shape[0]
    num_nodes = node_embeddings.shape[0]

    offset = 0
    out = torch.empty((num_facts*2), dtype=torch.int64)
    for head in [False, True]:  # head or tail prediction
        # compute the full score matrix (filter later)
        bases = data[:, 1:] if head else data[:, :2]
        targets = data[:, 0] if head else data[:, 2]

        # generate all possible triples by expanding and mutating 
        # either head or tail
        bexp = bases.view(num_facts, 1, 2).expand(num_facts,
                                                  num_nodes, 2)
        ar   = torch.arange(num_nodes).view(1, num_nodes, 1).expand(num_facts,
                                                                    num_nodes, 1)
        candidates = torch.cat([ar, bexp] if head else [bexp, ar], dim=2)

        scores = torch.zeros(candidates.shape[:2],
                             device=torch.device("cpu"))

        batches = [slice(begin, min(begin+mrr_batchsize, num_nodes))
                   for begin in range(0, num_nodes, mrr_batchsize)]
        for batch in batches:
            scores[batch] = score_distmult_bc((candidates[batch, :, 0],
                                               candidates[batch, :, 1],
                                               candidates[batch, :, 2]),
                                              node_embeddings,
                                              edge_embeddings).to('cpu')

        # filter out the true triples that aren't the target
        if filtered:
            filter_scores_(scores, data, true_heads, true_tails, head=head)

        # Select the true scores, and count the number of values larger than that
        true_scores = scores[torch.arange(num_facts).long(), targets.long()]
        ranks = torch.sum(scores > true_scores.view(num_facts, 1), dim=1, dtype=torch.int64)
        # -- This is the "optimistic" rank (assuming it's sorted to the front of the ties)
        num_ties = torch.sum(scores == true_scores.view(num_facts, 1), dim=1, dtype=torch.int64)

        # Account for ties (put the true example halfway down the ties)
        ranks = ranks + torch.round((num_ties - 1) / 2).long()

        out[offset:offset+num_facts] = ranks
        offset += num_facts

    return out + 1

def score_distmult_bc(data, node_embeddings, edge_embeddings):
    si, pi, oi = data

    s = node_embeddings[si, :]
    p = edge_embeddings[pi, :]
    o = node_embeddings[oi, :]

    if len(s.size()) == len(p.size()) == len(o.size()): # optimizations for common broadcasting
        if pi.size(-1) == 1 and oi.size(-1) == 1:
            singles = p * o # ignoring batch dimensions, this is a single vector
            return torch.matmul(s, singles.transpose(-1, -2)).squeeze(-1)

        if si.size(-1) == 1 and oi.size(-1) == 1:
            singles = s * o
            return torch.matmul(p, singles.transpose(-1, -2)).squeeze(-1)

        if si.size(-1) == 1 and pi.size(-1) == 1:
            singles = s * p
            return torch.matmul(o, singles.transpose(-1, -2)).squeeze(-1)

    return torch.sum(s * p * o, dim = -1)
