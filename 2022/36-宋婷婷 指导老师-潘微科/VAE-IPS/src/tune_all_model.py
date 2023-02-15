"""
Codes for training recommenders on semi-synthetic datasets used in the semi-synthetic experiments
in the paper "Unbiased Recommender Learning from Missing-Not-At-Random Implicit Feedback".
"""
from pathlib import Path

import numpy as np
from numpy import ndarray
from tqdm import tqdm
import tensorflow as tf
from tensorflow.python.framework import ops

from evaluate.evaluator import unbiased_evaluator
from models.recommenders import vaeRecommender
from propensity import create_propensity


def vae_trainer(sess: tf.Session, model: vaeRecommender,
                      train_matrix: np.ndarray, train_score: np.ndarray,
                      max_iters: int = 1000, batch_size: int = 2**10,) -> ndarray:
    """Train and Evaluate Implicit Recommender."""
    # initialise all the TF variables
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    num_users = train_matrix.shape[0]   # 15400
    num_items = train_matrix.shape[1]   # 1000

    idxList = list(range(num_users))

    # train the given implicit recommender
    np.random.seed(12345)

    # prediction X_hat
    X_hat = np.zeros((num_users, num_items))
    for iter in tqdm(np.arange(max_iters)):
        np.random.shuffle(idxList)    # 将训练数据中的user_id打乱顺序
        # train for one epoch
        for bnum, st_idx in enumerate(range(0, num_users, batch_size)):
            end_idx = min(st_idx + batch_size, num_users)
            X_batch = train_matrix[idxList[st_idx:end_idx]]    # (batch_size, num_items)
            train_score_batch = train_score[idxList[st_idx:end_idx]]    # (batch_size, num_items)

            # calculate training loss
            _, loss, X_batch_hat = sess.run([model.apply_grads, model.loss, model.logits],
                                   feed_dict={
                                              model.input_ph: X_batch,
                                              model.scores: train_score_batch,
                                              model.keep_prob_ph: 0.5,
                                              model.anneal_ph: 0.2,
                                              model.is_training_ph: 0
                                              })
            X_hat[idxList[st_idx:end_idx], :] = X_batch_hat

    sess.close()

    return X_hat


class Trainer:
    """Trainer Class for ImplicitRecommender."""
    at_k = [5]

    def __init__(self, etas: np.array, lams: np.array, dims: np.array, batch_size: int = 10, max_iters: int = 1000, model_name: str = 'vae-ips') -> None:
        """Initialize class."""
        self.etas = etas
        self.lams = lams
        self.dims = dims
        self.batch_size = batch_size
        self.max_iters = max_iters
        self.model_name = model_name

    def run(self) -> None:
        """Train implicit recommenders."""
        train = np.load(f'../data/point/train.npy')
        val = np.load(f'../data/point/val.npy')
        train_pos = np.load(f'../data/point/train_pos.npy')    # 仅含正反馈数据

        num_users = train_pos[:, 0].max() + 1   # 15400
        num_items = train_pos[:, 1].max() + 1   # 1000

        # train data
        train_matrix = np.zeros((num_users, num_items))
        # pscore for vae-ips and vae-ips-imp
        train_score = np.zeros((num_users, num_items))

        if self.model_name == 'vae':
            print('vae  ')
            for row in tqdm(train_pos):
                train_matrix[row[0], row[1]] = 1.0

        elif self.model_name == 'vae-ips':
            print('vae-ips  pscore')
            pscore = np.load(f'../data/point/pscore.npy')
            for row in tqdm(train_pos):
                train_score[:, row[1]] = pscore[row[1]]
                train_matrix[row[0], row[1]] = 1.0

        elif self.model_name == 'vae-ips-imp':
            print('vae-ips-imp  pscore')
            train_score = create_propensity()
            for row in tqdm(train_pos):
                train_matrix[row[0], row[1]] = 1.0

        df = pd.DataFrame(columns=['eta', 'lam', 'dim', 'dcg5'], dtype=float)
        i = 0
        for eta in self.etas:
            for lam in self.lams:
                for dim in self.dims:
                    print('=' * 25, i+1)
                    tf.set_random_seed(12345)
                    ops.reset_default_graph()
                    sess = tf.Session()
                    # 搭建模型网络框架
                    model = vaeRecommender(num_items=num_items, dim=dim, eta=eta, lam=lam, model_name=self.model_name)
                    # 训练模型
                    X_hat = vae_trainer(sess, model=model, train_matrix=train_matrix, train_score=train_score, max_iters=self.max_iters,
                                        batch_size=2**self.batch_size)    # (num_users, num_items)
                    # print('X_hat:\n', X_hat)
                    dcg5 = unbiased_evaluator(preds=X_hat, train=train, val=val, pscore=train_score, k=5)
                    print('dcg5:', dcg5)
                    data = [eta, lam, dim, dcg5]
                    df.loc[i] = data
                    i += 1
        ret_path = Path(f'../logs/{self.model_name}/tune/')
        ret_path.mkdir(parents=True, exist_ok=True)
        df.to_csv(ret_path / f'aoa_all_dcg5.csv')

        all_dcg5 = df.loc[:, 'dcg5']
        max_dcg5_idx = all_dcg5[all_dcg5 == all_dcg5.max()].index
        best_params = df.loc[max_dcg5_idx, :]
        best_params.to_csv(ret_path / f'best_params.csv')


import argparse
import yaml
import warnings
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--model_names', type=str, default=['vae', 'vae-ips', 'vae-ips-imp'])
parser.add_argument('--preprocess_data', action='store_true', default=False)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    tf.get_logger().setLevel("ERROR")
    args = parser.parse_args()

    # hyper-parameters
    config = yaml.safe_load(open('../conf/params_val.yaml', 'rb'))
    threshold = config['threshold']
    etas = config['etas']
    lams = config['lams']
    dims = config['dims']
    batch_size = config['batch_size']
    max_iters = config['max_iters']
    model_names = args.model_names

    for model_name in model_names:

        # 搭建模型
        trainer = Trainer(etas=etas, lams=lams, dims=dims, batch_size=batch_size, max_iters=max_iters, model_name=model_name)
        # 开始训练模型
        trainer.run()

        print('\n', '=' * 25, '\n')
        print(f'Finished Running {model_name}!')
        print('\n', '=' * 25, '\n')
