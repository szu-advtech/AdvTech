#!/usr/bin/python3

import logging
from re import fullmatch

from deep_geometry import vectorizer as gv
import numpy as np
from rdflib.term import Literal, URIRef
from rdflib import Namespace
import scipy.sparse as sp


_REGEX_COORDINATE = "\d\.?\d* \d\.?\d*"
_REGEX_WKTPOINT = "POINT\s?\(" + _REGEX_COORDINATE + "\)"
_REGEX_WKTPOLYGON = "POLYGON\s?\((\(" + _REGEX_COORDINATE + "[,\s" + _REGEX_COORDINATE + "]*\)(,\s)?)*\)"
_REGEX_WKTLITERAL = _REGEX_WKTPOINT + "|" + _REGEX_WKTPOLYGON

_OGC_NAMESPACE = Namespace(URIRef("http://www.opengis.net/ont/geosparql#"))

_MAX_POINTS = 64
_GEOVECTORIZER_VEC_LENGTH = 7

logger = logging.getLogger(__name__)

def generate_features(nodes_map, node_predicate_map, config, time_dim=1):
    """ Generate features for OGC WKT literals

    :param nodes_map: dictionary of node labels (URIs) : node idx {0, N}
    :param node_predicate_map: dictionary of node labels (URIs): {predicates}
    :param config: configuration dictionary
    :param time_dim: dimension of time (0 for RNN, 1 for CNN)
    :returns: list of length P with lists Q of length 3;
                P :- number of predicates that link to nodes with this feature
                Q :- [seq, node_idx, seq_lengths];
                    seq :- list with M CSR sparse arrays E x L if time_dim == 0 else L x E;
                        M :- number of nodes with this feature, such that M <= N
                        E :- Geometry embedding size (fixed)
                        L :- sequence length
                    node_idx :- numpy vector of length M, mapping seq index to node id
                    seq_lengths :- numpy array length M, mapping seq index to seq length
    """
    logger.debug("Generating wktLiteral features")

    return generate_relationwise_features(nodes_map, node_predicate_map,
                                          config, time_dim)

def generate_relationwise_features(nodes_map, node_predicate_map, config,
                                   time_dim):
    """ Stack vectors row-wise per relation and column stack relations
    """
    m = dict()
    node_idx = dict()
    data = dict()
    vec_length_map = dict()
    
    features = list(getFeature(nodes_map, _OGC_NAMESPACE.wktLiteral))
    n = len(features)

    failed = 0
    for node, i in features:
        try:
            value = str(node)
            vec = gv.vectorize_wkt(value)[:_MAX_POINTS,:]
        except:
            failed += 1
            continue

        vec_length = vec.shape[0]
        if vec_length <= 0:
            failed += 1
            continue

        # add means of X,Y to vector 
        mean_x = np.mean(vec[:,0])
        mean_y = np.mean(vec[:,1])
        vec = np.hstack([np.vstack([[mean_x, mean_y]]*vec_length), vec])

        if time_dim == 0:
            a = sp.csr_matrix(vec)
        else:  # time_dim == 1
            a = sp.csr_matrix(vec.T)

        for p in node_predicate_map[node]:
            if p not in data.keys():
                data[p] = np.empty(shape=n, dtype=object)
                node_idx[p] = np.empty(shape=(n), dtype=np.int32)
                vec_length_map[p] = np.empty(shape=(n), dtype=np.int32)
                m[p] = 0

            idx = m[p]
            data[p][idx] = a
            vec_length_map[p][idx] = vec_length
            node_idx[p][idx] = i
            m[p] = idx + 1


    msum = sum(m.values())
    logger.debug("Generated {} unique wktLiteral features ({} failed)".format(msum,
                                                                              failed))

    if msum <= 0:
        return None

    # normalization
    for p, pdata in data.items():
        pdata = pdata[:m[p]]

        sc = GeomScalerSparse(time_dim)
        means = sc.fit(pdata)
        data[p] = sc.transform(pdata, means)

    return [[data[p], node_idx[p][:m[p]], vec_length_map[p][:m[p]]]
            for p in data.keys()]

def validate(value):
    return fullmatch(_REGEX_WKTLITERAL, value)

def getFeature(nodes_map, datatype):
    for node, i in nodes_map.items():
        if not isinstance(node, Literal):
            continue
        if node.datatype is None or node.datatype.neq(datatype):
            continue

        yield (node, i)

### add sparse matrix support for geoscaler ###

class GeomScalerSparse:
    FULL_STOP_INDEX = -1
    def __init__(self, time_dim=0):
        self.scale_factor = 1.
        self.time_dim = time_dim

    def fit(self, geometry_vectors):
        means = [self.localized_mean(v) for v in geometry_vectors]
        min_maxs = list()

        for index, geometry in enumerate(geometry_vectors):
            full_stop_point_index = self.get_full_stop_index(geometry)

            x_and_y_coords = geometry[:full_stop_point_index, 2:4]\
                    if self.time_dim == 0 else geometry[2:4, :full_stop_point_index]
            min_maxs.append([
                np.min(x_and_y_coords - means[index]),
                np.max(x_and_y_coords - means[index])
            ])

        self.scale_factor = np.std(min_maxs)

        return means

    def transform(self, geometry_vectors, means):
        n = len(geometry_vectors)
        localized = np.empty(shape=n, dtype=object)
        for index, geometry in enumerate(geometry_vectors):
            stop_index = self.get_full_stop_index(geometry) + 1
            geometry_copy = geometry.copy()
            if self.time_dim == 0:
                geometry_copy[:stop_index, 2:4] -= means[index]
                geometry_copy[:stop_index, 2:4] /= self.scale_factor
            else:
                geometry_copy[2:4, :stop_index] -= means[index]
                geometry_copy[2:4, :stop_index] /= self.scale_factor

            localized[index] = geometry_copy

        return localized

    def get_full_stop_index(self, geometry_vector):
        full_stop_slice = geometry_vector[:, self.FULL_STOP_INDEX]\
                if self.time_dim == 0 else geometry_vector[self.FULL_STOP_INDEX, :]
        full_stop_point_index = sp.find(full_stop_slice == 1.0)[self.time_dim]

        if len(full_stop_point_index) <= 0:
            # we lack an end point (trimmed?)
            full_stop_point_index = geometry_vector.shape[self.time_dim]
        else:
            full_stop_point_index = full_stop_point_index[0]

        if full_stop_point_index == 0:
            # we're a point
            full_stop_point_index = 1

        return full_stop_point_index

    def localized_mean(self, geometry_vector):
        full_stop_point_index = self.get_full_stop_index(geometry_vector)
        geom_mean = [0, 0]
        if self.time_dim == 0:
            geom_mean = geometry_vector[:full_stop_point_index, 2:4].mean(axis=0)
        elif self.time_dim == 1:
            geom_mean = geometry_vector[2:4, :full_stop_point_index].mean(axis=1)
        else:
            raise ValueError("Invallid time dimension")

        return geom_mean
