#!/usr/bin/env python

import logging
import argparse
from os import getpid
from time import time

import numpy as np
import toml
import torch

from mrgcn.data.io.tarball import Tarball
from mrgcn.data.io.tsv import TSV
from mrgcn.data.utils import (is_readable,
                              is_writable,
                              set_seed)
from mrgcn.encodings.graph_features import setup_features
import mrgcn.tasks.node_classification as node_classification
import mrgcn.tasks.link_prediction as link_prediction


def run(A, X, Y, X_width, data, acc_writer,
        config, modules_config, optimizer_config,
        featureless, test_split, checkpoint):
    task = config['task']['type']
    logging.info("Starting {} task".format(task))
    print(task)
    if task == "node classification":
        model, optimizer, epoch, loss,\
              acc, labels, targets = node_classification.run(A, X, Y, X_width, acc_writer,
                                                             config,
                                                             modules_config,
                                                             optimizer_config,
                                                             featureless,
                                                             test_split,
                                                             checkpoint)
        return (model, optimizer, epoch, loss, acc, labels, targets)

    elif task == "link prediction":
        model, optimizer, epoch, loss,\
            mrr, hits_at_k, ranks, = link_prediction.run(A, X, X_width, data,
                                                         acc_writer,
                                                         config,
                                                         modules_config,
                                                         optimizer_config,
                                                         featureless, test_split,
                                                         checkpoint)
        return (model, optimizer, epoch, loss, mrr, hits_at_k, ranks)

def main(args, acc_writer, baseFilename, config):
    set_seed(config['task']['seed'])

    test_split = 'test' if args.test else 'valid'  # use test split?

    featureless = True
    if 'features' in config['graph'].keys() and\
       True in [feature['include'] for feature in config['graph']['features']]:
        featureless = False

    assert is_readable(args.input)
    logging.debug("Importing tarball")
    with Tarball(args.input, 'r') as tb:
        A = tb.get('A')
        F = tb.get('F')
        Y = tb.get('Y')  # empty if doing link prediction
        data = tb.get('data')  # empty if doing node classification
        sample_map = tb.get('sample_map')  # empty if doing link prediction
        class_map = tb.get('class_map')  # empty if doing link prediction

    ### prep data ###
    num_nodes = A.shape[0]
    X, X_width, modules_config, optimizer_config = setup_features(F, num_nodes, featureless, config)
    if X_width <= 0:
        featureless = True

    # order to enable state loading, else modules may get different IDs
    modules_config.sort(key=lambda t: t[0]) 

    task = config['task']['type']
    out = run(A, X, Y, X_width, data, acc_writer,
              config, modules_config, optimizer_config,
              featureless, test_split, args.load_checkpoint)

    model, optimizer = None, None
    loss, epoch = 0.0, 0.0
    if task == "node classification":
        model, optimizer, epoch, loss, acc, _, _ = out
        print("loss {:.4f} / accuracy {:.4f}".format(loss, acc))
    elif task == "link prediction":
        model, optimizer, epoch, loss, mrr, hits, _ = out
        results_str = f"Performance on {test_split} set: "\
                      f"MRR (raw) {mrr['raw']:.4f} - H@1 {hits['raw'][0]:.4f}"\
                      f" / H@3 {hits['raw'][1]:.4f} /"\
                      f" H@10 {hits['raw'][2]:.4f}"
        if config['task']['filter_ranks']:
            results_str += f" | MRR (filtered) {mrr['flt']:.4f} - "\
                           f"H@1 {hits['flt'][0]:.4f} / "\
                           f"H@3 {hits['flt'][1]:.4f} / "\
                           f"H@10 {hits['flt'][2]:.4f}"

        print(results_str)

    if not args.save_output:
        return (model, optimizer, loss, epoch)

    if task == "node classification":
        _, _, _, loss, acc, labels, targets = out

        out_writer = TSV(baseFilename+'_out.tsv', 'w')
        out_writer.writerow(['X', 'Y_hat', 'Y'])
        for i in range(len(labels)):
            out_writer.writerow([sample_map[test_split][i],
                                 class_map[labels[i]],
                                 class_map[targets[i]]])

    elif task == "link prediction":
        _, _, _, _, _, _, ranks = out

        rank_writer = TSV(baseFilename+'_ranks.tsv', 'w')
        if config['task']['filter_ranks']:
            rank_writer.writerow(["raw", "filtered"])
            rank_writer.writerows(zip(ranks['raw'],
                                      ranks['flt']))
        else:
            rank_writer.writerow(["raw"])
            for row in ranks['raw']:
                rank_writer.writerow([row])

    return (model, optimizer, loss, epoch)


def init_logger(filename, dry_run, verbose=0):
    if dry_run:
        level = logging.CRITICAL
        if verbose == 1:
            level = logging.INFO
        elif verbose >= 2:
            level = logging.DEBUG

        logging.basicConfig(format='%(message)s',
                            level=level)

        return

    logging.basicConfig(filename=filename,
                        format='[%(asctime)s] %(module)s/%(funcName)s | %(levelname)s: %(message)s',
                        level=logging.DEBUG)

    if verbose > 0:
        stream_handler = logging.StreamHandler()

        level = logging.INFO
        if verbose >= 2:
            level = logging.DEBUG
        stream_handler.setLevel(level)

        logging.getLogger().addHandler(stream_handler)

if __name__ == "__main__":
    timestamp = int(time())

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Configuration file (toml)", required=True, default=None)
    parser.add_argument("-i", "--input", help="Optional prepared input file (tar)", default=None)
    parser.add_argument("-o", "--output", help="Output directory", default="./tmp/")
    parser.add_argument("-v", "--verbose", help="Increase output verbosity", action='count', default=0)
    parser.add_argument("--dry_run", help="Suppress writing output files to disk",
                        action='store_true')
    parser.add_argument("--load_checkpoint", help="Load model state from disk",
                        default=None)
    parser.add_argument("--save_output", help="Write final output to disk.",
                        action='store_true')
    parser.add_argument("--save_checkpoint", help="Save model to disk",
                        action="store_true")
    parser.add_argument("--test", help="Report accuracy on test set rather than on validation set",
                        action='store_true')
    args = parser.parse_args()

    # load configuration
    assert is_readable(args.config)
    config = toml.load(args.config)

    # set output base filename
    baseFilename = "{}{}{}_{}".format(args.output, config['name'], timestamp,\
                                      getpid()) if args.output.endswith("/") \
                    else "{}/{}{}_{}".format(args.output, config['name'],\
                                             timestamp, getpid())
    assert is_writable(baseFilename)

    init_logger(baseFilename+'.log', args.dry_run, args.verbose)

    acc_writer = TSV(baseFilename+'_acc.tsv', 'w', args.dry_run)

    # log parameters
    logging.debug("Arguments:\n{}".format(
        "\n".join(["\t{}: {}".format(arg, getattr(args, arg)) for arg in vars(args)])))
    logging.debug("Configuration:\n{}".format(
        "\n".join(["\t{}: {}".format(k,v) for k,v in config.items()])))

    # run training
    model, optimizer, loss, epoch = main(args, acc_writer, baseFilename, config)

    if args.save_checkpoint:
        f_state = baseFilename + "_model_state_%d.pkl" % epoch
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss}, f_state)
        print("[SAVE] Writing model state to %s" % f_state)

    logging.shutdown()
