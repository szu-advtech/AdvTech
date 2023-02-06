#!/usr/bin/python3

import logging

import numpy as np
from rdflib.term import Literal
from rdflib.namespace import XSD
import scipy.sparse as sp

from models.utils import loadFromHub


_MAX_CHARS = 512  # one-hot encoded

logger = logging.getLogger(__name__)

def generate_features(nodes_map, node_predicate_map, config):
    """ Generate features for XSD anyURI literals

    URIs are treated as strings. Future work should instead separate
    the individual parts.

    Definition
    - anyURI := Char*
    -- Char  := [\t\r\n] + {unicode} + {ISO/IEC 10646}

    :param nodes_map: dictionary of node labels (URIs) : node idx {0, N}
    :param node_predicate_map: dictionary of node labels (URIs): {predicates}
    :param config: configuration dictionary
    :returns: list of length P with lists Q of length 3;
                P :- number of predicates that link to nodes with this feature
                Q :- [seq, node_idx, seq_lengths];
                    seq :- object array with M numpy arrays of length L;
                        M :- number of nodes with this feature, such that M <= N
                        L :- sequence length (number of tokens)
                    node_idx :- numpy vector of length M, mapping seq index to node id
                    seq_lengths :- numpy array of length M, mapping seq index to seq length
              """
    logger.debug("Generating string features")

    return generate_relationwise_features(nodes_map, node_predicate_map,
                                          config)

def generate_relationwise_features(nodes_map, node_predicate_map, config):
    """ Stack vectors row-wise per relation and column stack relations
    """
    m = dict()
    node_idx = dict()
    sequences = dict()
    seq_length_map = dict()
    
    tokenizer_config = config['tokenizer']['config']
    tokenizer = loadFromHub(tokenizer_config)
    if tokenizer.pad_token is None:
        pad_token = config['tokenizer']['pad_token']
        tokenizer.add_special_tokens({'pad_token': pad_token})
    
    features = list(getFeature(nodes_map, XSD.anyURI))
    n = len(features)

    failed = 0
    for node, i in features:
        try:
            sentence = str(node)
            sequence = encode(tokenizer, sentence)
            seq_length = len(sequence)
        except:
            failed += 1

            continue

        if seq_length <= 0:
            failed += 1

            continue

        a = np.array(sequence)[:_MAX_CHARS]

        for p in node_predicate_map[node]:
            if p not in sequences.keys():
                sequences[p] = np.empty(shape=n, dtype=object)
                node_idx[p] = np.empty(shape=(n), dtype=np.int32)
                seq_length_map[p] = np.empty(shape=(n), dtype=np.int32)
                m[p] = 0

            idx = m[p]
            sequences[p][idx] = a
            seq_length_map[p][idx] = seq_length
            node_idx[p][idx] = i
            m[p] = idx + 1

    msum = sum(m.values())
    logger.debug("Generated {} unique anyURI features ({} failed)".format(msum,
                                                                          failed))

    if msum <= 0:
        return None

    return [[sequences[p][:m[p]], node_idx[p][:m[p]], seq_length_map[p][:m[p]]]
            for p in sequences.keys()]

def encode(tokenizer, sentence):
    return tokenizer.encode(sentence, add_special_tokens=True)

def getFeature(nodes_map, datatype):
    for node, i in nodes_map.items():
        if not isinstance(node, Literal):
            continue
        if node.datatype is None or node.datatype.neq(datatype):
            continue

        yield (node, i)
