import torch
import torch.nn as nn
from collections import OrderedDict

from .utils import split_first_dim_linear
from .config_networks_new import ConfigureNetworks
from .set_encoder_new import mean_pooling
import torch.nn.functional as F
from path_index import *

torch.autograd.set_detect_anomaly(True)

NUM_SAMPLES = 1


# NOW RESNET 18

class SimpleCnaps(nn.Module):

    def __init__(self, device, use_two_gpus, mt=False):
        super(SimpleCnaps, self).__init__()
        self.device = device
        self.use_two_gpus = use_two_gpus
        networks = ConfigureNetworks(
            #pretrained_resnet_path='./trained_model/pretrain/pretrainmodel_rseed42_digits_zbk_resnet18_1.pkl',
            pretrained_resnet_path=metatrain_pretrainmodel,
            feature_adaptation='film', mt=mt)
        # networks = ConfigureNetworks(pretrained_resnet_path='./trained_model/pretrain/pretrainmodel_rseed42_resnet18_ours_senet.pkl',
        #                                feature_adaptation='film', mt=mt)                             
        # networks = ConfigureNetworks(pretrained_resnet_path='./trained_model/pretrain/pretrainmodel_rseed42_resnet18_act_ichar_new_sr100train.pkl',
        #                              feature_adaptation='film', mt=mt)

        self.set_encoder = networks.get_encoder()

        # self.attn_fn = MultiHeadAttention(1, 512, 512, 512, dropout=0.5)
        # self.classifier_adaptation_network = networks.get_classifier_adaptation()
        # self.classifier = networks.get_classifier()

        self.feature_extractor = networks.get_feature_extractor()
        self.feature_adaptation_network = networks.get_feature_adaptation()
        self.task_representation = None
        self.class_representations = OrderedDict()

        # self.fc = nn.Linear(512,10,bias=False)
        # self.fc.weight.requires_grad = False

        # self.class_precision_matrices = OrderedDict() # Dictionary mapping class label (integer) to regularized precision matrices estimated

    def forward(self, context_images, context_labels, target_images):

        self.task_representation = self.set_encoder(context_images)
        context_features, target_features = self._get_features(context_images, target_images)
        self._build_class_reps_and_covariance_estimates(context_features, context_labels)
        class_means = torch.stack(list(self.class_representations.values())).squeeze(1)
        sample_logits = self._L1dist(class_means, target_features)
        self.class_representations.clear()
        return split_first_dim_linear(sample_logits, [NUM_SAMPLES, target_images.shape[0]])

    def _get_features(self, context_images, target_images):
        # Parallelize forward pass across multiple GPUs (model parallelism)
        if self.use_two_gpus:
            context_images_1 = context_images.cuda(1)
            target_images_1 = target_images.cuda(1)
            task_representation_1 = self.task_representation.cuda(1)
            self.feature_extractor_params = self.feature_adaptation_network(task_representation_1)
            context_features_1 = self.feature_extractor(context_images_1, self.feature_extractor_params)
            context_features = context_features_1.cuda(0)
            target_features_1 = self.feature_extractor(target_images_1, self.feature_extractor_params)
            target_features = target_features_1.cuda(0)
        else:
            self.feature_extractor_params = self.feature_adaptation_network(self.task_representation)
            context_features = self.feature_extractor(context_images, self.feature_extractor_params)
            target_features = self.feature_extractor(target_images, self.feature_extractor_params)

        return context_features, target_features

    def _L1dist(self, support, query):
        caler = nn.PairwiseDistance(p=1)

        for i in range(query.shape[0]):

            distance = -caler(query[i], support).view(1, -1)
            if i == 0:
                attention_distribution = distance
            else:
                attention_distribution = torch.cat((attention_distribution, distance), 0)

        return attention_distribution

    def _cosdist(self, support, query):

        for i in range(query.shape[0]):

            feature1 = support.view(support.shape[0], -1)
            feature2 = query[i].view(1, -1)
            feature2 = feature2.view(feature2.shape[0], -1)
            feature1 = F.normalize(feature1)
            feature2 = F.normalize(feature2)
            distance = -(1 - feature2.mm(feature1.t()))
            if i == 0:
                attention_distribution = distance
            else:
                attention_distribution = torch.cat((attention_distribution, distance), 0)

        return attention_distribution

    def _build_class_reps_and_covariance_estimates(self, context_features, context_labels):
        for c in torch.unique(context_labels):
            class_features = torch.index_select(context_features, 0, self._extract_class_indices(context_labels, c))
            class_rep = mean_pooling(class_features)
            self.class_representations[c.item()] = class_rep
            
    def estimate_cov(self, examples, rowvar=False, inplace=False):
        if examples.dim() > 2:
            raise ValueError('m has more than 2 dimensions')
        if examples.dim() < 2:
            examples = examples.view(1, -1)
        if not rowvar and examples.size(0) != 1:
            examples = examples.t()
        factor = 1.0 / (examples.size(1) - 1)
        if inplace:
            examples -= torch.mean(examples, dim=1, keepdim=True)
        else:
            examples = examples - torch.mean(examples, dim=1, keepdim=True)
        examples_t = examples.t()

        return factor * examples.matmul(examples_t).squeeze()

    def estimate_cov1(self, examples, rowvar=False, inplace=False):
        if examples.dim() > 2:
            raise ValueError('m has more than 2 dimensions')
        if examples.dim() < 2:
            examples = examples.view(1, -1)
        if not rowvar and examples.size(0) != 1:
            examples = examples.t()
        factor = 1.0 / (examples.size(1) - 1)
        examples = examples[0] - torch.mean(examples[0])
        tempt = examples.t()
        out = factor * examples.matmul(tempt).squeeze()

        return out

    @staticmethod
    def _extract_class_indices(labels, which_class):
        class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
        class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
        return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector

    def distribute_model(self):
        self.feature_extractor.cuda(1)
        self.feature_adaptation_network.cuda(1)
