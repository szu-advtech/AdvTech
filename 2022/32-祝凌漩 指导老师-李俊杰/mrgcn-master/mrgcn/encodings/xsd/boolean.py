#!/usr/bin/python3

import logging
from re import match

import numpy as np
from rdflib.term import Literal
from rdflib.namespace import XSD


_REGEX_BOOLEAN = "true|false|0|1"

logger = logging.getLogger(__name__)

def generate_features(nodes_map, node_predicate_map, config):
    """ Generate encodings for XSD boolean literals

    Definition
    - boolean := 'true' | 'false' | '1' | '0'

    Numerical booleans and their text equivalents are mapped to [-1, 1] for
    maximum separation in the value space. No normalization is needed.

    :param nodes_map: dictionary of node labels (URIs) : node idx {0, N}
    :param node_predicate_map: dictionary of node labels (URIs): {predicates}
    :param config: configuration dictionary
    :returns: list of length P with lists Q of length 3;
                P :- number of predicates that link to nodes with this feature
                Q :- [enc, node_idx, lengths];
                    enc :- numpy array M x C;
                        M :- number of nodes with this feature, such that M <= N
                    node_idx :- numpy vector of length M, mapping seq index to node id
                    lengths :- numpy array of length M with 1
    """
    logger.debug("Generating boolean encodings")
    C = 1  # number of items per feature

    return generate_relationwise_features(nodes_map, node_predicate_map, C,
                                          config)

def generate_relationwise_features(nodes_map, node_predicate_map, C, config):
    m = dict()
    encodings = dict()
    node_idx = dict()

    features = list(getFeature(nodes_map, XSD.boolean))
    n = len(features)

    failed = 0
    for node, i in features:
        try:
            value = str(node)
            if value.isalpha():
                value = 1.0 if value.lower() == 'true' else -1.0
            elif value.isdigit():
                value = 1.0 if int(value) == 1 else -1.0
            else:
                try:
                    value = 1.0 if int(float(value)) == 1 else -1.0
                except:
                    raise Exception()
        except:
            failed += 1
            continue

        for p in node_predicate_map[node]:
            if p not in encodings.keys():
                encodings[p] = np.empty(shape=(n, C), dtype=np.float32)
                node_idx[p] = np.empty(shape=(n), dtype=np.int32)
                m[p] = 0

            idx = m[p]
            # add to matrix structures
            encodings[p][idx] = [value]
            node_idx[p][idx] = i
            m[p] = idx + 1

    msum = sum(m.values())
    logger.debug("Generated {} unique boolean encodings ({} failed)".format(msum,
                                                                            failed))

    if msum <= 0:
        return None

    return [[encodings[p][:m[p]], node_idx[p][:m[p]], np.ones(m[p])]
            for p in encodings.keys()]

def validate(value):
    return match(_REGEX_BOOLEAN, value)

def getFeature(nodes_map, datatype):
    for node, i in nodes_map.items():
        if not isinstance(node, Literal):
            continue
        if node.datatype is None or node.datatype.neq(datatype):
            continue

        yield (node, i)
