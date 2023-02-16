#!/usr/bin/env python

from copy import deepcopy
import logging

logger = logging.getLogger(__name__)

def optimizer_params(model, optim_config):
    opt_params = list()
    for module_name_numbered, module in model.module_dict.items():
        module_name = module_name_numbered.split('_')[0]

        # filter parameters of frozen layers
        params = [param for param in module.parameters() if param.requires_grad]
        module_params = {'params': params}

        if module_name == 'RGCN':
            # use default lr
            opt_params.append(module_params)
            continue

        for datatype, params in optim_config:
            if datatype in ['xsd.boolean', 'xsd.numeric']:
                mod_name = 'FC'
            elif datatype in ['xsd.date', 'xsd.dateTime', 'xsd.gYear']:
                mod_name = 'FC'
            elif datatype in ['xsd.string', 'xsd.anyURI']:
                mod_name = 'Transformer'
            elif datatype in ['ogc.wktLiteral']:
                mod_name = 'GeomCNN'
            elif datatype in ['blob.image']:
                mod_name = 'ImageCNN'
            else:
                raise Exception('Unsupported datatype: %s' % datatype)

            if module_name == mod_name:
                module_params.update(params)
                break

        opt_params.append(module_params)

    return opt_params

class EarlyStop:
    stop = None
    tolerance = -1
    patience = -1
    _patience_default = -1
    best_score = -1
    best_weights = None
    best_optim = None
    delay = -1

    def __init__(self, patience=7, tolerance=0.01, delay=10):
        self.tolerance = tolerance
        self.delay = delay
        self._patience_default = patience

        self.reset_counter()

    def record(self, score, weights, optim):
        if self.delay > 0:
            self.delay -= 1

            return

        if self.best_score < 0:
            self._update(score, weights, optim)

            return

        self.patience -= 1
        if (score + self.tolerance) < self.best_score:
            self._update(score, weights, optim)
            self.reset_counter()

        if self.patience <= 0:
            self.stop = True

    def _update(self, score, weights, optim):
        self.best_score = score
        self.best_weights = deepcopy(weights.state_dict())
        self.best_optim = deepcopy(optim.state_dict())

    def reset_counter(self):
        self.patience = self._patience_default
        self.stop = False

#def mkbatches(mat, node_idx, num_batches=1):
#    """ split N x * array in batches
#    """
#    n = mat.shape[0]  # number of samples
#    num_batches = min(n, num_batches)
#    idc = np.arange(n, dtype=np.int32)
#
#    if num_batches <= 1:
#        logger.debug("Full batch mode")
#
#    idc_assignments = np.array_split(idc, num_batches)
#    node_assignments = [np.array(node_idx, dtype=np.int32)[slce]
#                        for slce in idc_assignments]
#
#    return list(zip(idc_assignments, node_assignments))
#
#def mkbatches_varlength(sequences, node_idx, seq_length_map,
#                        num_batches=1):
#    n = len(sequences)
#    num_batches = min(n, num_batches)
#    if num_batches <= 1:
#        logger.debug("Full batch mode")
#
#    # sort on length
#    idc = np.arange(n, dtype=np.int32)
#    _, sequences_sorted_idc = zip(*sorted(zip(seq_length_map, idc)))
#
#    seq_assignments = np.array_split(sequences_sorted_idc, num_batches)
#    node_assignments = [np.array(node_idx, dtype=np.int32)[slce]
#                        for slce in seq_assignments]
#
#    return list(zip(seq_assignments, node_assignments))

