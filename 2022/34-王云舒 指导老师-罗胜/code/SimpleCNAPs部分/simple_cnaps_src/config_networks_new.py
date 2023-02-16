from .resnet_new import film_resnet18, resnet18, film_resnet10, film_resnet34
from .adaptation_networks import NullFeatureAdaptationNetwork, FilmAdaptationNetwork, \
    LinearClassifierAdaptationNetwork, FilmLayerNetwork, FilmArAdaptationNetwork
from .set_encoder_new import SetEncoder
from .utils import cosine_classifier, linear_classifier

"""
Creates the set encoder, feature extractor and feature adaptation networks.
创建集编码器、特征提取器和特征自适应网络。
"""


class ConfigureNetworks:
    def __init__(self, pretrained_resnet_path, feature_adaptation, mt=False):

        self.classifier = linear_classifier
        # self.classifier = cosine_classifier

        self.encoder = SetEncoder() #集编码器，相当于一个神经网络？
        z_g_dim = self.encoder.pre_pooling_fn.output_size

        # parameters for ResNet18
        num_maps_per_layer = [64, 128, 256, 512]
        # num_blocks_per_layer = [1, 1, 1, 1]
        # num_blocks_per_layer = [3, 4, 6, 3]
        num_blocks_per_layer = [2,2,2,2]
        num_initial_conv_maps = 64

        if feature_adaptation == "no_adaptation":
            self.feature_extractor = resnet18(
                pretrained=True,
                pretrained_model_path=pretrained_resnet_path,
                mt=mt
            )
            self.feature_adaptation_network = NullFeatureAdaptationNetwork()

        elif feature_adaptation == "film":
            self.feature_extractor = film_resnet18(
                pretrained=True,
                pretrained_model_path=pretrained_resnet_path,
                mt=mt
            )
            self.feature_adaptation_network = FilmAdaptationNetwork(
                layer=FilmLayerNetwork,
                num_maps_per_layer=num_maps_per_layer,
                num_blocks_per_layer=num_blocks_per_layer,
                z_g_dim=z_g_dim
            )

        elif feature_adaptation == 'film+ar':
            self.feature_extractor = film_resnet10(
                pretrained=True,
                pretrained_model_path=pretrained_resnet_path,
                mt=mt
            )
            self.feature_adaptation_network = FilmArAdaptationNetwork(
                feature_extractor=self.feature_extractor,
                num_maps_per_layer=num_maps_per_layer,
                num_blocks_per_layer=num_blocks_per_layer,
                num_initial_conv_maps=num_initial_conv_maps,
                z_g_dim=z_g_dim
            )

        # Freeze the parameters of the feature extractor 冻结特征提取器的参数

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # for name, param in self.feature_extractor.named_parameters():
        #     k1 = name.split('.')
        #     if k1[0].startswith('layer'):
        #         if k1[2] == 'se':
        #             print(k1[2])
        #             continue 
        #     param.requires_grad = False

    def get_encoder(self):
        return self.encoder

    def get_classifier(self):
        return self.classifier

    def get_feature_adaptation(self):
        return self.feature_adaptation_network

    def get_feature_extractor(self):
        return self.feature_extractor
