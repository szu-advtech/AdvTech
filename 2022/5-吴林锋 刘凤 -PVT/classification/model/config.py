import ml_collections


def get_PVT_classification_small_config():
    config = ml_collections.ConfigDict()
    config.C = [3, 64, 128, 320, 512]
    config.P = [4, 2, 2, 2]
    config.R = [8, 4, 2, 1]
    config.H = 224
    config.W = 224
    config.num_stages = 4
    config.num_heads = [1, 2, 4, 8]
    config.mlp_rate = [8, 8, 4, 4]
    config.num_encoder_layers = [3, 4, 6, 3]  # TransformerEncoder层数
    # encoder层数
    config.qk_scale = None
    config.qkv_bias = True
    config.drop_rate = 0.
    config.train_dir = 'K:/Projects/PC Projects/pvt/classification/data/flower_photos/tar/train'
    config.val_dir = 'K:/Projects/PC Projects/pvt/classification/data/flower_photos/tar/val'
    config.batch_size = 8
    config.num_classes = 5
    config.decoder_channels = (256, 128, 64, 16)
    config.activation = 'softmax'
    config.epoch = 50
    config.lr = 0.001
    config.lrf = 0.01
    config.F4 = False
    return config

def get_PVT_classification_tiny_config():
    config = ml_collections.ConfigDict()
    config.C = [3, 64, 128, 320, 512]
    config.P = [4, 2, 2, 2]
    config.R = [8, 4, 2, 1]
    config.H = 32
    config.W = 32
    config.qkv_bias = True
    config.num_stages = 4
    config.num_heads = [1, 2, 5, 8]
    config.mlp_rate = [8, 8, 4, 4]
    config.num_encoder_layers = [2, 2, 2, 2]  # TransformerEncoder层数
    # encoder层数
    config.qk_scale = None
    config.drop_rate = 0.
    config.train_dir = '/data/flower_photos/tar/train'
    config.val_dir = '/data/flower_photos/tar/val'
    config.batch_size = 8
    config.num_classes = 10
    config.epoch = 100
    config.lr = 0.001
    config.lrf = 0.01
    return config




CONFIGS = {
    'PVT_classification_small': get_PVT_classification_small_config(),
    'PVT_classification_tiny': get_PVT_classification_tiny_config()
}
