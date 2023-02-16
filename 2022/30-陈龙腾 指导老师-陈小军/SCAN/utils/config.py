"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import yaml
from easydict import EasyDict
from utils.utils import mkdir_if_missing


def create_config(config_file_env, config_file_exp, topk, checkpoint,best_model):
    # Config for environment path
    with open(config_file_env, 'r') as stream:
        root_dir = yaml.safe_load(stream)['root_dir']

    with open(config_file_exp, 'r') as stream:
        config = yaml.safe_load(stream)

    cfg = EasyDict()

    # Copy
    for k, v in config.items():
        cfg[k] = v

    cfg['num_neighbors'] = topk
    cfg['DIR'] = '/public16_data/clt/SCAN/'

    # Set paths for pretext task (These directories are needed in every stage)
    base_dir = os.path.join(root_dir, cfg['train_db_name'])
    pretext_dir = os.path.join(base_dir, 'pretext')
    mkdir_if_missing(base_dir)
    mkdir_if_missing(pretext_dir)
    cfg['pretext_dir'] = pretext_dir
    cfg['pretext_checkpoint'] = os.path.join(pretext_dir, 'checkpoint.pth.tar')
    # cfg['pretext_model'] = os.path.join(pretext_dir, 'model.pth.tar')

    if cfg['train_db_name'] == 'cifar-10':
        cfg['pretext_model'] = os.path.join(pretext_dir, 'simclr_cifar-10.pth.tar')
    elif cfg['train_db_name'] == 'stl-10':
        cfg['pretext_model'] = os.path.join(pretext_dir, 'simclr_stl-10.pth.tar')
    elif cfg['train_db_name'] == 'cifar-20':
        cfg['pretext_model'] = os.path.join(pretext_dir, 'simclr_cifar-20.pth.tar')
    else:
        cfg['pretext_model'] = os.path.join('output/baseline/imagenet_50/pretext', 'moco_v2_800ep_pretrain.pth.tar')

    cfg['top{}_neighbors_train_path'.format(cfg['num_neighbors'])] = os.path.join(pretext_dir,
                                                    'top{}-train-neighbors.npy'.format(cfg['num_neighbors']))

    cfg['topk_neighbors_val_path'] = os.path.join(pretext_dir, 'topk-val-neighbors.npy')

    # If we perform clustering or self-labeling step we need additional paths.
    # We also include a run identifier to support multiple runs w/ same hyperparams.
    if cfg['setup'] in ['scan', 'selflabel']:
        base_dir = os.path.join(root_dir, cfg['train_db_name'])
        scan_dir = os.path.join(base_dir, 'scan')
        selflabel_dir = os.path.join(base_dir, 'selflabel')
        mkdir_if_missing(base_dir)
        mkdir_if_missing(scan_dir)
        mkdir_if_missing(selflabel_dir)
        cfg['scan_dir'] = scan_dir
        cfg['scan_checkpoint'] = os.path.join(scan_dir, checkpoint)
        cfg['scan_model'] = scan_dir
        # cfg['scan_model'] = os.path.join(scan_dir, 'model.pth.tar')    # using in scan
        cfg['best_scan_model'] = os.path.join(scan_dir, best_model)  # using in selflabel
        cfg['scan_best_clustering_results'] = os.path.join(scan_dir, 'best_clustering_results.pth.tar')
        cfg['selflabel_dir'] = selflabel_dir
        cfg['selflabel_checkpoint'] = os.path.join(selflabel_dir, checkpoint)
        cfg['selflabel_model'] = selflabel_dir
        cfg['best_selflabel_results'] = os.path.join(selflabel_dir, 'best_selflabel_results.pth.tar')

    return cfg 
