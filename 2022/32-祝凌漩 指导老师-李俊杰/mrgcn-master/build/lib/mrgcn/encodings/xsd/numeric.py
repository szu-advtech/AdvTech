#!/usr/bin/python3

import logging
from re import match

import numpy as np
from rdflib.term import Literal
from rdflib.namespace import XSD


_REGEX_NUMERIC = "\d+"

_XSD_NUMERICAL = {
    XSD.double,
    XSD.decimal,
    XSD.float,
    XSD.integer,
    XSD.long,
    XSD.int,
    XSD.short,
    XSD.byte,
    XSD.nonNegativeInteger,
    XSD.nonPositiveInteger,
    XSD.unsignedLong,
    XSD.unsignedInt,
    XSD.unsignedShort,
    XSD.unsignedByte,
    XSD.negativeInteger,
    XSD.positiveInteger}

logger = logging.getLogger(__name__)

def generate_features(node_map, node_predicate_map, config):
    """ Generate encodings for XSD numeric literals

    Definition
    - numeric := \d+

    Returns an 2D array A and an vector b, such that A[i] holds the vector
    representation of the feature belonging to node b[i].

    Encoding
    - a vector v of length C = 1
    -- v[0] : \d+ : numerical value(s)

    :param nodes_map: dictionary of node labels (URIs) : node idx {0, N}
    :param node_predicate_map: dictionary of node labels (URIs): {predicates}
    :param config: configuration dictionary
    :returns: list of length P with lists Q of length 3;
                P :- number of predicates that link to nodes with this feature
                Q :- [enc, node_idx, None];
                    enc :- numpy array M x C;
                        M :- number of nodes with this feature, such that M <= N
                        C :- desired output dimension of encoder
                    node_idx :- numpy vector of length M, mapping seq index to node id
                    None :- not used here

    """
    logger.debug("Generating numerical encodings")
    C = 1  # number of items per feature per relation

    if config['datatype'] == "xsd.numeric":
        datatype = _XSD_NUMERICAL
    elif config['datatype'] == "xsd.double":
        datatype = [XSD.double]
    elif config['datatype'] == "xsd.decimal":
        datatype = [XSD.decimal]
    elif config['datatype'] == "xsd.float":
        datatype = [XSD.float]
    elif config['datatype'] == "xsd.integer":
        datatype = [XSD.integer]
    elif config['datatype'] == "xsd.long":
        datatype = [XSD.long]
    elif config['datatype'] == "xsd.int":
        datatype = [XSD.int]
    elif config['datatype'] == "xsd.short":
        datatype = [XSD.short]
    elif config['datatype'] == "xsd.byte":
        datatype = [XSD.byte]
    elif config['datatype'] == "xsd.nonNegativeInteger":
        datatype = [XSD.nonNegativeInteger]
    elif config['datatype'] == "xsd.nonPositiveInteger":
        datatype = [XSD.nonPositiveInteger]
    elif config['datatype'] == "xsd.unsignedLong":
        datatype = [XSD.unsignedLong]
    elif config['datatype'] == "xsd.unsignedInt":
        datatype = [XSD.unsignedInt]
    elif config['datatype'] == "xsd.unsignedShort":
        datatype = [XSD.unsignedShort]
    elif config['datatype'] == "xsd.unsignedByte":
        datatype = [XSD.unsignedByte]
    elif config['datatype'] == "xsd.negativeInteger":
        datatype = [XSD.negativeInteger]
    elif config['datatype'] == "xsd.positiveInteger":
        datatype = [XSD.positiveInteger]

    return generate_relationwise_features(node_map, node_predicate_map, C,
                                          config, datatype)

def generate_relationwise_features(node_map, node_predicate_map, C, config,
                                   datatype):
    """ Stack vectors row-wise per relation and column stack relations
    """
    m = dict()
    encodings = dict()
    node_idx = dict()
    values_min = dict()
    values_max = dict()
    
    features = list(getFeature(node_map, datatype))
    n = len(features)
    
    failed = 0
    for node, i in features:
        try:
            value = float(str(node))
        except:
            failed += 1
            continue

        for p in node_predicate_map[node]:
            if p not in encodings.keys():
                encodings[p] = np.empty(shape=(n, C), dtype=np.float32)
                node_idx[p] = np.empty(shape=(n), dtype=np.int32)
                m[p] = 0
                values_min[p] = None
                values_max[p] = None

            if values_max[p] is None or value > values_max[p]:
                values_max[p] = value
            if values_min[p] is None or value < values_min[p]:
                values_min[p] = value

            idx = m[p]
            # add to matrix structures
            encodings[p][idx] = [value]
            node_idx[p][idx] = i
            m[p] = idx + 1

    msum = sum(m.values())
    logger.debug("Generated {} unique {} encodings ({} failed)".format(
        msum,
        config['datatype'],
        failed))

    if msum <= 0:
        return None

    # normalization over encodings
    for p in encodings.keys():
        idc = np.arange(m[p])
        if values_max[p] == values_min[p]:
            encodings[p][idc] = 0.0
            continue

        encodings[p][idc] = (2*(encodings[p][idc] - values_min[p]) /
                            (values_max[p] - values_min[p])) -1.0

    return [[encodings[p][:m[p]], node_idx[p][:m[p]], np.ones(m[p])]
            for p in encodings.keys()]

def validate(value):
    return match(_REGEX_NUMERIC, value)

def getFeature(nodes_map, datatypes):
    for node, i in nodes_map.items():
        if not isinstance(node, Literal):
            continue
        if node.datatype is None or node.datatype not in datatypes:
            continue

        yield (node, i)

