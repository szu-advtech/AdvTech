#!/usr/bin/python3

from functools import reduce
from importlib import import_module
import logging
from typing import Sequence

import numpy as np
import scipy.sparse as sp

from mrgcn.data.utils import getConfParam
from mrgcn.encodings.xsd.xsd_hierarchy import XSDHierarchy
from mrgcn.models.temporal_cnn import TCNN

logger = logging.getLogger(__name__)

ENCODINGS_PKG = "mrgcn.encodings"
AVAILABLE_FEATURES = {"xsd.boolean", "xsd.numeric","xsd.string", "xsd.anyURI",
                      "blob.image", "ogc.wktLiteral", "xsd.date",
                      "xsd.dateTime", "xsd.gYear"}

def construct_features(nodes_map, knowledge_graph, feature_configs,
                      separate_literals):
    """ Construct specified features for given nodes
    """
    hierarchy = XSDHierarchy()

    # keep track which predicates are used to link to which objects
    node_predicate_map = dict()
    for _,p,o in knowledge_graph.triples(separate_literals=separate_literals):
        if o not in node_predicate_map.keys():
            node_predicate_map[o] = set()
        node_predicate_map[o].add(p)

    features = dict()
    for feature_config in feature_configs:
        if not feature_config['include']:
            continue

        feature_name = feature_config['datatype']
        feature = feature_module(hierarchy, feature_name)
        if feature is None:
            logger.debug("Specified feature not available: {}".format(feature_name))
            continue

        # dynamically load module
        module = import_module("{}.{}".format(ENCODINGS_PKG, feature))
        feature_encoding = module.generate_features(nodes_map,
                                                    node_predicate_map,
                                                    feature_config)
        # feature_encoding is a list for every predicate with this feature as
        # range, with each member being also a list L of length 3 with L[0] the 
        # vectorized encodings (array if fixed length, list of arrays
        # otherwise), L[1] an array with node indices that map the encodings to the
        # correct graph' nodes, and L[2] an array with the lengths of the
        # encodings (uniform if fixed length).

        if feature_encoding is not None:
            features[feature_name] = feature_encoding

    return features

def setup_features(F, num_nodes, featureless, config):
    X_width = 0  # number of columns in X
    X = [np.empty((num_nodes, X_width), dtype=float)]  # dummy

    modules_config = list()
    optimizer_config = list()
    if not featureless:
        features_enabled = features_included(config)
        logging.debug("Features included: {}".format(", ".join(features_enabled)))
        for datatype in features_enabled:
            if datatype in F.keys():
                logger.debug("Found {} encoding set(s) for datatype {}".format(
                    len(F[datatype]),
                    datatype))

        # create batched  representations for neural encodings
        feature_configs = config['graph']['features']
        features, modules_config, optimizer_config, feat_width = construct_feature_matrix(F,
                                                                                          features_enabled,
                                                                                          feature_configs)
        X_width += feat_width
        X.extend(features)

    return (X, X_width, modules_config, optimizer_config)

def feature_module(hierarchy, feature_name):
    for feature in AVAILABLE_FEATURES:
        # prefer more tailored module
        if feature == feature_name:
            return feature

    if not feature_name.startswith("xsd"):
        return None

    feature_name = feature_name[4:]
    for feature in AVAILABLE_FEATURES:
        if not feature.startswith("xsd"):
            continue
        if hierarchy.subtypeof(feature[4:], feature_name):
            return feature

    return None

def construct_feature_matrix(F, features_enabled, feature_configs):
    embeddings_width = 0
    modules_config = list()
    embeddings = list()
    optimizer_config = list()

    datatypes = list(set.intersection(set(features_enabled),
                                      set(F.keys()),
                                      AVAILABLE_FEATURES))
    for datatype in sorted(datatypes):
        feature_config = next((conf for conf in feature_configs
                               if conf['datatype'] == datatype),
                              dict())
        print(feature_config)
        embedding_dim = feature_config['embedding_dim']
        print('embedding dim is')
        print(embedding_dim)
        dropout = feature_config['p_dropout']

        # parameters to pass to optimizer
        optim_params = dict()
        for param_name, param_value in feature_config.items():
            if not param_name.startswith('optim_'):
                # not an optimizer parameter
                continue

            param_name = param_name.lstrip('optim_')
            optim_params[param_name] = param_value

        optimizer_config.append((datatype, optim_params))
        encoding_sets = F.pop(datatype, list())

        # by default, the encodings for each modality are stored
        # as separate arrays for each predicate they are connected with.
        # Here we merge these arrays such that the result is a single array
        # per modality that is agnistic to the predicates.
        weight_sharing = getConfParam(feature_config, 'share_weights', False)
        if weight_sharing:
            logger.debug("weight sharing enabled for {}".format(datatype))
            if datatype in ["blob.image"]:
                encoding_sets = merge_img_encoding_sets(encoding_sets)
            elif datatype in ["ogc.wktLiteral"]:
                encoding_sets = merge_sparse_encoding_sets(encoding_sets)
            elif datatype in ["xsd.string", "xsd.anyURI"]:
                encoding_sets = merge_discrete_encoding_sets(encoding_sets)
            elif datatype in ["xsd.date", "xsd.dateTime", "xsd.gYear",
                              "xsd.boolean", "xsd.numeric"]:
                encoding_sets = merge_continuous_encoding_sets(encoding_sets)
            else:
                logger.warning("Unsupported datatype %s" % datatype)

        # add a bit of noise to reduce the chance of overfitting, eg
        # when one neural encoder converges faster than others.
        noise_mp = getConfParam(feature_config, 'noise_multiplier', -1)
        p_noise = getConfParam(feature_config, 'p_noise', -1)
        if p_noise > 0:
            logger.debug("adding noise to {}".format(datatype))
            if datatype in ["ogc.wktLiteral"]:
                add_noise_(encoding_sets, p_noise, noise_mp, sparse=True)
            elif datatype in ["blob.image", "xsd.date", "xsd.dateTime",
                              "xsd.gYear", "xsd.boolean", "xsd.numeric"]:
                add_noise_(encoding_sets, p_noise, noise_mp, sparse=False)
            else:
                logger.warning("Unsupported datatype %s" % datatype)

        gpu_acceleration = getConfParam(feature_config, 'gpu_acceleration', False)

        # generate modality-specific configurations to pass to
        # the MR-GCN to tell is what encoders to prepare.
        num_encoding_sets = len(encoding_sets)
        for encodings, _, seq_lengths in encoding_sets:
            if datatype in ["xsd.boolean", "xsd.numeric"]:
                feature_dim = 1
                feature_size = encodings.shape[feature_dim]
                modules_config.append((datatype, (feature_size,
                                                  embedding_dim,
                                                  dropout),
                                       gpu_acceleration))
            elif datatype in ["xsd.date", "xsd.dateTime", "xsd.gYear"]:
                feature_dim = 1
                feature_size = encodings.shape[feature_dim]
                modules_config.append((datatype, (feature_size,
                                                  embedding_dim,
                                                  dropout),
                                       gpu_acceleration))
            elif datatype in ["xsd.string", "xsd.anyURI"]:
                # stored as vstacked arrays of tokens of variable length
                model_config = feature_config['model']
                modules_config.append((datatype, (model_config,
                                                  embedding_dim,
                                                  dropout),
                                       gpu_acceleration))
            elif datatype in ["ogc.wktLiteral"]:
                # stored as array of arrays (point_repr x num_points)
                # determine average vector length
                feature_dim = 0  # set to 1 for RNN
                feature_size = reduce(lambda total, enc: total + enc.shape[feature_dim],
                                      encodings, 0) // len(encodings)

                # adjust model size to 'best' fit sequence lengths of CNN
                model_size = "M"
                if not weight_sharing or num_encoding_sets <= 1:
                    seq_length_q25 = np.quantile(seq_lengths, 0.25)
                    if seq_length_q25 < TCNN.LENGTH_M:
                        model_size = "S"
                    elif seq_length_q25 < TCNN.LENGTH_L:
                        model_size = "M"
                    else:
                        model_size = "L"

                modules_config.append((datatype, (feature_size,
                                                  embedding_dim,
                                                  model_size,
                                                  dropout),
                                       gpu_acceleration))
            elif datatype in ["blob.image"]:
                # stored as tensor (num_images x num_channels x width x height)
                model_config = feature_config['model']
                transform_config = feature_config['transform']
                modules_config.append((datatype, (model_config,
                                                  transform_config,
                                                  embedding_dim,
                                                  dropout),
                                       gpu_acceleration))
            else:
                logger.warning("Unsupported datatype %s" % datatype)

            embeddings_width += embedding_dim

        # deal with outliers
        rm_outliers = getConfParam(feature_config, 'remove_outliers', False)
        if rm_outliers:
            if datatype in ["ogc.wktLiteral", "xsd.string", "xsd.anyURI"]:
                encoding_sets = [remove_outliers(*f) for f in encoding_sets]
            else:
                raise Warning("Outlier removal not supported for datatype %s" %
                              datatype)

        tr_outliers = getConfParam(feature_config, 'trim_outliers', False)
        if tr_outliers:
            if datatype in ["ogc.wktLiteral"]:
                feature_dim = 0  # set to 1 for RNN
                encoding_sets = [trim_outliers_sparse(*f, feature_dim) for f in encoding_sets]
            elif datatype in ["xsd.string", "xsd.anyURI"]:
                feature_dim = 0  # set to 1 for RNN
                encoding_sets = [trim_outliers_dense(*f, feature_dim) for f in encoding_sets]

            else:
                raise Warning("Outlier trimming not supported for datatype %s" %
                              datatype)

        embeddings.append([datatype, encoding_sets, gpu_acceleration])

    return (embeddings, modules_config, optimizer_config, embeddings_width)


def _mkdense(encodings, node_idx, encodings_length_map, n):
    """ Return N x M matrix with N := NUM_NODES and M := NUM_COLS
        Use node index to map encodings to correct nodes
    """
    F = np.zeros(shape=(n, *encodings.shape[1:]), dtype=np.float32)
    F[node_idx] = encodings

    return F

def features_included(config):
    features = set()

    if 'features' not in config['graph']:
        return features
    feature_configs = config['graph']['features']

    for feature_config in feature_configs:
        if not feature_config['include']:
            continue

        features.add(feature_config['datatype'])

    return features

def merge_sparse_encoding_sets(encoding_sets):
    """ Merge sparse encoding sets into a single set; aka share weights

        Expects 2D sparse CSR matrices as encodings.

        Encodings sets contain the encoded feature matrices of the same
        datatype but connected by different predicates. If literals with the
        same value are represented by a single node, then the same node can
        have multiple values. This is handled by averaging the values for these
        nodes.
    """
    if len(encoding_sets) <= 1:
        return encoding_sets

    node_idx = np.concatenate([node_idx for _, node_idx, _ in encoding_sets])
    node_idx_unique, node_idx_counts = np.unique(node_idx, return_counts=True)
    node_idx_mult = node_idx_unique[node_idx_counts > 1]  # nodes with multiple features

    N = node_idx_unique.shape[0]  # N <= NUM_NODES

    encodings_merged = np.empty(shape=N, dtype=object)
    node_idx_merged = node_idx_unique
    seq_length_merged = np.zeros(shape=N, dtype=np.int32)

    merged_idx_map = {v:i for i,v in enumerate(node_idx_merged)}
    to_merge = dict()
    for encodings, node_index, seq_length in encoding_sets:
        # assume that non-filled values are zero
        for i in range(len(node_index)):
            idx = node_index[i]
            merged_idx = merged_idx_map[idx]

            enc = encodings[i]
            enc_length = seq_length[i]
            if idx in node_idx_mult:
                if idx not in to_merge:
                    to_merge[idx] = list()
                to_merge[idx].append(enc)

                continue

            encodings_merged[merged_idx] = enc
            seq_length_merged[merged_idx] = enc_length

    for idx, encodings in to_merge.items():
        shapes = [enc.shape for enc in encodings]
        enc_shape = tuple(map(max, zip(*shapes)))  # maximum for all dims
        enc_length = enc_shape[-1]

        a = np.zeros(shape=enc_shape)
        for enc in encodings:
            a[:enc.shape[0], :enc.shape[1]] += enc.todense()
        a /= len(encodings)

        merged_idx = merged_idx_map[idx]
        encodings_merged[merged_idx] = sp.csr_matrix(a)
        seq_length_merged[merged_idx] = enc_length

    return [[encodings_merged, node_idx_merged, seq_length_merged]]

def merge_discrete_encoding_sets(encoding_sets):
    """ Merge encoding sets into a single set; aka share weights

        Encodings sets contain the encoded feature matrices of the same
        datatype but connected by different predicates. If literals with the
        same value are represented by a single node, then the same node can
        have multiple values. This is handled by averaging the values for these
        nodes.
    """
    if len(encoding_sets) <= 1:
        return encoding_sets

    # all nodes with this modality
    node_idx = np.concatenate([node_idx for _, node_idx, _ in encoding_sets])
    node_idx_unique, node_idx_counts = np.unique(node_idx, return_counts=True)
    node_idx_mult = node_idx_unique[node_idx_counts > 1]  # nodes with multiple features

    N = node_idx_unique.shape[0]  # N <= NUM_NODES

    encodings_merged = np.empty(shape=N, dtype=object)
    node_idx_merged = node_idx_unique
    seq_length_merged = np.zeros(shape=N, dtype=np.int32)

    merged_idx_map = {v:i for i,v in enumerate(node_idx_merged)}
    merged_idx_multvalue_map = {merged_idx_map[v]:list() for v in node_idx_mult}
    merged_idx_multlength_map = {merged_idx_map[v]:list() for v in node_idx_mult}
    for encodings, node_index, seq_length in encoding_sets:
        # assume that non-filled values are zero
        for i in range(len(node_index)):
            idx = node_index[i]  # global node index
            merged_idx = merged_idx_map[idx]  # node index in merged matrix

            enc = encodings[i]
            if idx in node_idx_mult:
                merged_idx_multvalue_map[merged_idx].append(enc)
                merged_idx_multlength_map[merged_idx].append(seq_length[i])
            else:
                encodings_merged[merged_idx] = enc

            if seq_length is not None:
                # use max length for nodes that occur more than once
                seq_length_merged[merged_idx] = max(seq_length_merged[merged_idx],
                                                    seq_length[i])

    for i in range(node_idx_mult.shape[0]):
        # majority vote per column, else pick first
        idx = node_idx_mult[i]
        merged_idx = merged_idx_map[idx]
        encodings = merged_idx_multvalue_map[merged_idx]
        lengths = merged_idx_multlength_map[merged_idx]

        longest_enc_idx = np.argmax([len(enc) for enc in encodings])
        longest_enc_length = len(encodings[longest_enc_idx])
        a = -np.ones(shape=(len(encodings), longest_enc_length), dtype=int)
        for j in range(len(encodings)):
            a[j,:lengths[j]] = encodings[j]
        
        merged_enc = list()
        for j in range(longest_enc_length):
            values, counts = np.unique(a[a[:,j]>=0, j], return_counts=True)
            merged_enc.append(values[np.argmax(counts)])

        encodings_merged[merged_idx] = np.array(merged_enc)
        seq_length_merged[merged_idx] = longest_enc_length

    return [[encodings_merged, node_idx_merged, seq_length_merged]]

def merge_continuous_encoding_sets(encoding_sets):
    """ Merge encoding sets into a single set; aka share weights

        Encodings sets contain the encoded feature matrices of the same
        datatype but connected by different predicates. If literals with the
        same value are represented by a single node, then the same node can
        have multiple values. This is handled by averaging the values for these
        nodes.
    """
    if len(encoding_sets) <= 1:
        return encoding_sets

    # all nodes with this modality
    node_idx = np.concatenate([node_idx for _, node_idx, _ in encoding_sets])
    node_idx_unique, node_idx_counts = np.unique(node_idx, return_counts=True)
    node_idx_mult = node_idx_unique[node_idx_counts > 1]  # nodes with multiple features

    N = node_idx_unique.shape[0]  # N <= NUM_NODES
    M = max([enc[0].shape[1] for enc in encoding_sets])  # highest width

    encodings_merged = np.zeros(shape=(N, M), dtype=np.float32)
    node_idx_merged = node_idx_unique
    seq_length_merged = np.zeros(shape=N, dtype=np.int32)

    merged_idx_map = {v:i for i,v in enumerate(node_idx_merged)}
    merged_idx_multvalue_map = {merged_idx_map[v]:None for v in node_idx_mult}
    for encodings, node_index, seq_length in encoding_sets:
        # assume that non-filled values are zero
        for i in range(len(node_index)):
            idx = node_index[i]  # global node index
            merged_idx = merged_idx_map[idx]  # node index in merged matrix

            enc = encodings[i]
            if idx in node_idx_mult:
                mult_enc = merged_idx_multvalue_map[merged_idx]
                if mult_enc is None:
                    mult_enc = np.zeros(enc.shape, dtype=np.float32)
                merged_idx_multvalue_map[merged_idx] = mult_enc + enc
            else:
                encodings_merged[merged_idx,:int(seq_length[i])] = enc

            if seq_length is not None:
                # use max length for nodes that occur more than once
                seq_length_merged[merged_idx] = max(seq_length_merged[merged_idx],
                                                    seq_length[i])

    for i in range(node_idx_mult.shape[0]):
        idx = node_idx_mult[i]
        merged_idx = merged_idx_map[idx]
        count = node_idx_counts[node_idx_counts > 1][i]

        # average vectors for nodes that occur more than once
        encodings_merged[merged_idx] = merged_idx_multvalue_map[merged_idx] / count

    return [[encodings_merged, node_idx_merged, seq_length_merged]]

def merge_img_encoding_sets(encoding_sets):
    """ Merge 3D encoding sets into a single set; aka share weights

        Encodings sets contain the encoded feature matrices of the same
        datatype but connected by different predicates. If literals with the
        same value are represented by a single node, then the same node can
        have multiple values. This is handled by averaging the values for these
        nodes.
    """
    if len(encoding_sets) <= 1:
        return encoding_sets

    node_idx = np.concatenate([node_idx for _, node_idx, _ in encoding_sets])
    node_idx_unique, node_idx_counts = np.unique(node_idx, return_counts=True)
    node_idx_mult = node_idx_unique[node_idx_counts > 1]  # nodes with multiple features

    N = node_idx_unique.shape[0]  # N <= NUM_NODES
    c, W, H = encoding_sets[0][0].shape[1:]  # assume same image size on all

    encodings_merged = np.zeros(shape=(N, c, W, H), dtype=np.float32)
    node_idx_merged = node_idx

    merged_idx_map = {v:i for i,v in enumerate(node_idx_merged)}
    merged_idx_multvalue_map = {merged_idx_map[v]:None for v in node_idx_mult}
    for encodings, node_index, _ in encoding_sets:
        # assume that non-filled values are zero
        for i in range(len(node_index)):
            idx = node_index[i]
            merged_idx = merged_idx_map[idx]

            enc = encodings[i]
            if idx in node_idx_mult:
                mult_enc = merged_idx_multvalue_map[merged_idx]
                if mult_enc is None:
                    mult_enc = np.zeros(enc.shape, dtype=np.float32)
                merged_idx_multvalue_map[merged_idx] = mult_enc + enc

                continue

            encodings_merged[merged_idx] = enc
    
    for i in range(node_idx_mult.shape[0]):
        idx = node_idx_mult[i]
        merged_idx = merged_idx_map[idx]
        count = node_idx_counts[node_idx_counts > 1][i]

        # average vectors for nodes that occur more than once
        encodings_merged[merged_idx] = merged_idx_multvalue_map[merged_idx] / count

    return [[encodings_merged, node_idx_merged, -np.ones(N)]]

def stack_encoding_sets(encoding_sets):
    """ Stack encoding sets horizontally. Entries for the same node are
    placed on the same row.
    """
    if len(encoding_sets) <= 1:
        return encoding_sets

    node_idx = np.concatenate([node_idx for _, node_idx, _ in encoding_sets])
    node_idx_unique = np.unique(node_idx)

    N = node_idx_unique.shape[0]  # N <= NUM_NODES
    M = sum(map(lambda enc: enc.shape[1], encoding_sets))  # sum of widths

    encodings_merged = np.zeros(shape=(N, M), dtype=np.float32)
    node_idx_merged = sorted(node_idx_unique)
    seq_length_merged = np.repeat([M], repeats=N)

    merged_idx_map = {v:i for i,v in enumerate(node_idx_merged)}
    j = 0
    for encodings, node_index, seq_length in encoding_sets:
        m = encodings.shape[1]
        # assume that non-filled values are zero
        for k in range(len(node_index)):
            idx = node_index[k]
            merged_idx = merged_idx_map[idx]

            enc = encodings[k]
            encodings_merged[merged_idx,j:j+m] = enc

        j += m

    return [[encodings_merged, node_idx_merged, seq_length_merged]]

def add_noise_(encoding_sets, p_noise, multiplier=0.01, sparse=False):
    for mset in encoding_sets:
        encodings = mset[0]  # numpy array
        if sparse:  # variable length encodings
            for i in range(len(encodings)):
                size = encodings[i].size
                shape = encodings[i].shape

                b = np.random.binomial(1, p_noise, size=size)
                noise = b.reshape(shape) * (2 * np.random.random(shape) - 1)
                encodings[i].data += multiplier * noise
        else:
            size = encodings.size
            shape = encodings.shape

            b = np.random.binomial(1, p_noise, size=size)
            noise = b.reshape(shape) * (2 * np.random.random(shape) - 1)
            encodings += multiplier * noise

def trim_outliers_dense(sequences, node_idx, seq_length_map, feature_dim=0):
    # split outliers
    q25 = np.quantile(seq_length_map, 0.25)
    q75 = np.quantile(seq_length_map, 0.75)
    IQR = q75 - q25
    cut_off = IQR * 1.5

    if IQR <= 0.0:  # no length difference
        return [sequences, node_idx, seq_length_map]

    n = len(sequences)
    sequences_trimmed = np.empty(n, dtype=object)
    seq_length_map_trimmed = np.zeros(n, dtype=int)
    for i, seq_length in enumerate(seq_length_map):
        sequence = sequences[i]
        threshold = int(q75 + cut_off)
        if seq_length > threshold:
            sequence = np.concatenate([sequence[:threshold-1],
                                       sequence[-1]])

        sequences_trimmed[i] = sequence
        seq_length_map_trimmed[i] = sequence.shape[0]

    m = len(sequences_trimmed)
    d = len(sequences) - m
    if d > 0:
        logger.debug("Trimmed {} outliers)".format(d))

    return [sequences_trimmed, node_idx, seq_length_map_trimmed]

def trim_outliers_sparse(sequences, node_idx, seq_length_map, feature_dim=0):
    # split outliers
    q25 = np.quantile(seq_length_map, 0.25)
    q75 = np.quantile(seq_length_map, 0.75)
    IQR = q75 - q25
    cut_off = IQR * 1.5

    if IQR <= 0.0:  # no length difference
        return [sequences, node_idx, seq_length_map]

    n = len(sequences)
    sequences_trimmed = np.empty(n, dtype=object)
    seq_length_map_trimmed = np.zeros(n, dtype=int)
    for i, seq_length in enumerate(seq_length_map):
        sequence = sequences[i]
        threshold = int(q75 + cut_off)
        if seq_length > threshold:
            sequence = sequence[:, :threshold] if feature_dim == 0\
                else sequence[:threshold, :]

        sequences_trimmed[i] = sequence
        seq_length_map_trimmed[i] = sequence.shape[1-feature_dim]

    m = len(sequences_trimmed)
    d = len(sequences) - m
    if d > 0:
        logger.debug("Trimmed {} outliers)".format(d))

    return [sequences_trimmed, node_idx, seq_length_map_trimmed]

def remove_outliers(sequences, node_idx, seq_length_map):
    # split outliers
    q25 = np.quantile(seq_length_map, 0.25)
    q75 = np.quantile(seq_length_map, 0.75)
    IQR = q75 - q25
    cut_off = IQR * 1.5

    if IQR <= 0.0:  # no length difference
        return [sequences, node_idx, seq_length_map]

    n = len(sequences)
    sequences_filtered = np.empty(n, dtype=object)
    node_idx_filtered = np.zeros(n, dtype=int)
    seq_length_map_filtered = np.zeros(n, dtype=int)
    j = 0
    for i, seq_length in enumerate(seq_length_map):
        if seq_length < q25 - cut_off or seq_length > q75 + cut_off:
            # skip outlier
            continue

        sequences_filtered[j] = sequences[i]
        node_idx_filtered[j] = node_idx[i]
        seq_length_map_filtered[j] = seq_length

        j += 1

    d = len(sequences) - j
    if d > 0:
        logger.debug("Filtered {} outliers ({} remain)".format(d, j))

    return [sequences_filtered[:j], node_idx_filtered[:j], seq_length_map_filtered[:j]]

def isDatatypeIncluded(config, datatype):
    if 'graph' in config.keys()\
        and 'features' in config['graph'].keys():
            features = config['graph']['features']
            for feature in features:
                if feature['datatype'] == datatype:
                   return feature['include']

    return False

def getDatatypeConfig(config, datatype):
    if 'graph' in config.keys()\
        and 'features' in config['graph'].keys():
            features = config['graph']['features']
            for feature in features:
                if feature['datatype'] == datatype:
                    return feature

    return None
