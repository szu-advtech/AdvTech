"""
Codes for running the real-world experiments
in the paper "VAE-IPS: A Deep Generative Recommendation Method for Unbiased Learning from Implicit Feedback".
"""

import argparse
import yaml
import warnings

import tensorflow as tf

from data.preprocessor import preprocess_dataset
from trainer import Trainer

parser = argparse.ArgumentParser()

parser.add_argument('--model_names', type=str, default=['vae', 'vae-ips', 'vae-ips-imp'])
parser.add_argument('--preprocess_data', action='store_true', default=False)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    tf.get_logger().setLevel("ERROR")
    args = parser.parse_args()

    model_names = args.model_names

    for model_name in model_names:
        # hyper-parameters
        config = yaml.safe_load(open(f'../conf/params_{model_name}.yaml', 'rb'))
        threshold = config['threshold']
        eta = config['eta']
        lam = config['lam']
        dim = config['dim']
        batch_size = config['batch_size']
        max_iters = config['max_iters']

        # run simulations
        # 处理原始数据集
        if args.preprocess_data:
            preprocess_dataset(threshold=threshold)
            print('Finished preprocess_dataset()!')

        # 搭建模型
        trainer = Trainer(eta=eta, lam=lam, dim=dim, batch_size=batch_size, max_iters=max_iters, model_name=model_name)
        # 开始训练模型
        trainer.run()

        print('\n', '=' * 25, '\n')
        print(f'Finished Running {model_name}!')
        print('\n', '=' * 25, '\n')
