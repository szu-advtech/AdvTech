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

from evaluate.evaluator import aoa_evaluator
from models.recommenders import vaeRecommender
from propensity import create_propensity


def vae_trainer(sess: tf.Session, model: vaeRecommender,
                      train_matrix: np.ndarray, pscore: np.ndarray,
                      max_iters: int = 1000, batch_size: int = 2**10,
                      model_name: str = 'vae-ips') -> ndarray:
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
            train_score_batch = pscore[idxList[st_idx:end_idx]]    # (batch_size, num_items)

            # calculate training loss
            _, loss, X_batch_hat = sess.run([model.apply_grads, model.loss, model.logits],
                                   feed_dict={
                                              model.input_ph: X_batch,
                                              model.scores: train_score_batch,
                                              model.keep_prob_ph: 0.5,
                                              model.anneal_ph: 0.2,
                                              model.is_training_ph: 1
                                              })
            X_hat[idxList[st_idx:end_idx], :] = X_batch_hat

    sess.close()

    return X_hat


class Trainer:
    """Trainer Class for ImplicitRecommender."""
    at_k = [1, 3, 5]

    def __init__(self, eta: float = 0.01, lam=1e-6, dim: int = 5, batch_size: int = 10, max_iters: int = 1000, model_name: str = 'vae-ips') -> None:
        """Initialize class."""
        self.eta = eta
        self.lam = lam
        self.dim = dim
        self.batch_size = batch_size
        self.max_iters = max_iters
        self.model_name = model_name

    def run(self) -> None:
        """Train implicit recommenders."""
        train_pos = np.load(f'../data/point/train_pos.npy')    # 仅含正反馈数据
        test = np.load(f'../data/point/test.npy')

        num_users = np.int(train_pos[:, 0].max() + 1)   # 15400
        num_items = np.int(train_pos[:, 1].max() + 1)   # 1000

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


        tf.set_random_seed(12345)
        ops.reset_default_graph()
        sess = tf.Session()

        # 搭建模型网络框架
        model = vaeRecommender(num_items=num_items, dim=self.dim, eta=self.eta, lam=self.lam, model_name=self.model_name)
        # 训练模型
        X_hat = vae_trainer(sess, model=model, train_matrix=train_matrix, pscore=train_score, max_iters=self.max_iters,
                            batch_size=2**self.batch_size, model_name=self.model_name)    # (num_users, num_items)
        # print('X_hat:\n', X_hat)

        # evaluate a given recommender
        ret_path = Path(f'../logs/{self.model_name}/results/')
        ret_path.mkdir(parents=True, exist_ok=True)
        results = aoa_evaluator(preds=X_hat, test=test, model_name=self.model_name, at_k=self.at_k)
        results.to_csv(ret_path / f'aoa_all.csv')
