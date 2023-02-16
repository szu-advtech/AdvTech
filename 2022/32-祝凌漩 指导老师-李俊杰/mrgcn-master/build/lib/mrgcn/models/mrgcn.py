#!/usr/bin/env python

import logging
from operator import itemgetter
import warnings

import numpy as np
import torch
import torch.nn as nn

from mrgcn.encodings.blob.image import Normalizer as IMNORM
from mrgcn.models.perceptron import MLP
from mrgcn.models.temporal_cnn import TCNN
from mrgcn.models.imagecnn import ImageCNN
from mrgcn.models.transformer import Transformer
from mrgcn.models.rgcn import RGCN
from mrgcn.models.utils import loadFromHub, torch_intersect1d
from mrgcn.data.batch import MiniBatch


CUDA = torch.device("cuda")

logger = logging.getLogger(__name__)

class MRGCN(nn.Module):
    def __init__(self, modules, embedding_modules, num_relations,
                 num_nodes, num_bases=-1, p_dropout=0.0, featureless=False,
                 bias=False, link_prediction=False, gcn_gpu_acceleration=False):
        """
        Multimodal Relational Graph Convolutional Network
        Mini Batch support
        """
        super().__init__()

        assert len(modules) > 0

        self.num_nodes = num_nodes  # != labelled nodes
        self.p_dropout = p_dropout
        self.module_dict = nn.ModuleDict()

        language_model = None
        image_model = None

        self.im_norm = None

        self.devices = dict()

        # add embedding layers
        self.modality_modules = dict()
        self.modality_out_dim = 0
        self.compute_modality_embeddings = False
        h, i, j, k = 0, 0, 0, 0
        for datatype, args, gpu_acceleration in embedding_modules:
            module = None
            dim_out = -1
            seq_length = -1

            if datatype in ["xsd.boolean", "xsd.numeric"]:
                ncols, dim_out, dropout = args
                module = MLP(input_dim=ncols,
                            output_dim=dim_out,
                            num_layers=1,
                            p_dropout=dropout)

                self.module_dict["FC_num_"+str(i)] = module
                h += 1
            if datatype in ["xsd.date", "xsd.dateTime", "xsd.gYear"]:
                ncols, dim_out, dropout = args
                module = MLP(input_dim=ncols,
                            output_dim=dim_out,
                            num_layers=2,
                            p_dropout=dropout)

                self.module_dict["FC_temp_"+str(i)] = module
                h += 1
            if datatype in ["xsd.string", "xsd.anyURI"]:
                model_config, dim_out, dropout = args
                if language_model is None:
                    language_model = loadFromHub(model_config)

                module = Transformer(language_model,
                                     output_dim=dim_out,
                                     p_dropout=dropout)

                self.module_dict["Transformer_"+str(i)] = module
                i += 1
            if datatype == "blob.image":
                model_config, transform_config, dim_out, dropout = args
                if image_model is None:
                    image_model = loadFromHub(model_config)

                module = ImageCNN(image_model,
                                  output_dim=dim_out,
                                  p_dropout=dropout)

                self.module_dict["ImageCNN_"+str(j)] = module
                j += 1

                if 'mean' in transform_config.keys()\
                    and 'std' in transform_config.keys():
                        self.im_norm = IMNORM(transform_config['mean'],
                                              transform_config['std'])
            if datatype == "ogc.wktLiteral":
                nrows, dim_out, model_size, dropout = args
                module = TCNN(features_in=nrows,
                              features_out=dim_out,
                              p_dropout=dropout,
                              size=model_size)
                seq_length = module.minimal_length
                
                self.module_dict["GeomCNN_"+str(k)] = module
                k += 1

            if datatype not in self.modality_modules.keys():
                self.modality_modules[datatype] = list()
            self.modality_modules[datatype].append((module, seq_length, dim_out))
            self.modality_out_dim += dim_out
            self.compute_modality_embeddings = True

            device = torch.device("cpu")
            if gpu_acceleration:
                if torch.cuda.is_available():
                    device = torch.device("cuda")
                else:
                    warnings.warn("CUDA Resource not available", ResourceWarning)

            # record which device the module is on
            self.devices[datatype] = device
            module.to(device)

            if seq_length > 0:
                logger.debug(f"Setting sequence length to {seq_length} for datatype {datatype}")

        # add graph convolution layers
        self.rgcn = RGCN(modules, num_relations, num_nodes,
                          num_bases, p_dropout, featureless, bias,
                          link_prediction)
        self.module_dict["RGCN"] = self.rgcn

        # record all devices used by the modules
        device = torch.device("cpu")
        if gcn_gpu_acceleration:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                warnings.warn("CUDA Resource not available", ResourceWarning)

        self.devices["relational"] = device
        self.rgcn.to(device)
        
        # place X on the GPU if every module is on the GPU
        # else default to cpu
        self.X_device = torch.device("cuda")
        for dev in self.devices.values():
            if dev is not device:
                self.X_device = torch.device("cpu")
                break

    def forward(self, batch):
        if isinstance(batch, MiniBatch):
            return self._forward_mini_batch(batch)

        # full batch
        return self._forward_full_batch(batch)

    def _forward_full_batch(self, batch):
        X, F = batch.X[0], batch.X[1:]

        X_dev = None
        if self.compute_modality_embeddings:
            # expects a natural ordering of nodes; same as X
            batch_idx = torch.arange(self.num_nodes)  # full batch

            XF = self._compute_modality_embeddings(F, batch_idx)

            X = X.to(self.X_device)
            X = torch.cat([X,XF], dim=1)

            # move to same device as rgcn
            rgcn_device = self.devices["relational"]
            X_dev = X.to(rgcn_device)

        if X_dev is not None:
            X_dev = X_dev.float()

        A_dev = batch.A  # A := sparse tensor

        # Forward pass through graph convolution layers
        X_dev = self.rgcn(X_dev, A_dev)

        return X_dev

    def _forward_mini_batch(self, batch):
        """ Mini batch MR-GCN.
        """

        X_dev = None
        if self.compute_modality_embeddings:
            # compute the initial embeddings to use as input to the R-GCN.
            # this is only necessary for the most-distant nodes, since 
            # nodes closer to the batch nodes will use the embeddings computed
            # by the previous graph convvolution layers.
            X, F = batch.X[0], batch.X[1:]

            # most distant nodes
            batch_idx = batch.A.neighbours[-1]

            XF = self._compute_modality_embeddings(F, batch_idx)
            
            X = X.to(self.X_device)
            X = torch.cat([X,XF], dim=1)

            # move to same device as rgcn
            rgcn_device = self.devices["relational"]
            X_dev = X.to(rgcn_device)

        if X_dev is not None:
            X_dev = X_dev.float()

        A_dev = batch.A  # A := object of MiniBatch class

        # Forward pass through graph convolution layers
        X_dev = self.rgcn(X_dev, A_dev)

        return X_dev

    def _compute_modality_embeddings(self, F, batch_idx):
        batch_num_nodes = len(batch_idx)
        X = torch.zeros((batch_num_nodes, self.modality_out_dim),
                        dtype=torch.float32, device=self.X_device)
        offset = 0
        for datatype, encoding_sets, _ in F:
            if datatype not in self.modality_modules.keys():
                continue

            num_sets = len(encoding_sets)
            for i, encoding_set in enumerate(encoding_sets):
                module, _, out_dim = self.modality_modules[datatype][i]

                encodings, node_idx, _ = encoding_set
                common_nodes = torch_intersect1d(node_idx, batch_idx)
                if len(common_nodes) <= 0:
                    # no nodes in this batch have this datatype
                    offset += out_dim
                    continue

                # find indices for common nodes
                F_batch_mask = torch.isin(node_idx, common_nodes)
                X_batch_mask = torch.isin(batch_idx, common_nodes)

                logger.debug(" {} (set {} / {})".format(datatype,
                                                        i+1,
                                                        num_sets))

                # forward pass and store on correct position in output tensor
                if datatype in ["xsd.string", "xsd.anyURI"]:
                    data = encodings[F_batch_mask].int()
                elif datatype == "blob.image":
                    data_raw = encodings[F_batch_mask]
                    data = self.im_norm.normalize_(data_raw)
                else:
                    data = encodings[F_batch_mask].float()

                out = module(data)
                if self.X_device is not torch.device("cuda"):
                    # X is on cpu
                    out = out.to(torch.device("cpu"))

                X[X_batch_mask, offset:offset+out_dim] = out

                offset += out_dim

        return X
