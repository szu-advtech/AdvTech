"""Evaluate Implicit Recommendation models."""
from typing import  List

import numpy as np
import pandas as pd
from tqdm import tqdm

from .metrics import average_precision_at_k, dcg_at_k, recall_at_k


metrics = {'DCG': dcg_at_k,
           'Recall': recall_at_k,
           'MAP': average_precision_at_k}


def aoa_evaluator(preds: np.ndarray,
                  test: np.ndarray,
                  model_name: str,
                  at_k: List[int] = [1, 3, 5]) -> pd.DataFrame:
    """Calculate ranking metrics with average-over-all evaluator."""
    # test data
    users = test[:, 0]    # (54000,)
    items = test[:, 1]    # (54000,)
    relevances = 0.001 + 0.999 * test[:, 2]    # (54000,)

    # prepare ranking metrics
    results = {}
    for k in at_k:
        for metric in metrics:
            results[f'{metric}@{k}'] = []

    # calculate ranking metrics
    np.random.seed(12345)
    for user in tqdm(set(users)):
        indices = users == user     # (54000,)
        pos_items = items[indices]  # (10,)
        rel = relevances[indices]   # (10,)

        # predict ranking score for each user
        scores = preds[user, pos_items]    # (10,)
        for k in at_k:
            for metric, metric_func in metrics.items():
                results[f'{metric}@{k}'].append(metric_func(rel, scores, k))

        # aggregate results
        results_df = pd.DataFrame(index=results.keys())
        results_df[model_name] = list(map(np.mean, list(results.values())))

    return results_df


def unbiased_evaluator(preds: np.ndarray,
                       train: np.ndarray,
                       val: np.ndarray,
                       pscore: np.ndarray,
                       metric: str = 'DCG',
                       num_negatives: int = 100,
                       k: int = 5) -> float:
    """Calculate ranking metrics by unbiased evaluator."""
    # validation data
    users = val[:, 0]    # 12671
    items = val[:, 1]    # 12671
    positive_pairs = np.r_[train[train[:, 2] == 1, :2], val]    # (125077, 2)

    # calculate an unbiased DCG score
    dcg_values = list()
    unique_items = np.unique(items)    # 950
    np.random.seed(12345)
    for user in tqdm(set(users)):
        indices = users == user
        pos_items = items[indices]
        all_pos_items = positive_pairs[positive_pairs[:, 0] == user]
        neg_items = np.random.permutation(np.setdiff1d(unique_items, all_pos_items))[:num_negatives]
        used_items = np.r_[pos_items, neg_items]
        pscore_ = pscore[user][used_items]
        relevances = np.r_[np.ones_like(pos_items), np.zeros(num_negatives)]

        # calculate an unbiased DCG score for a user
        scores = preds[user][used_items]
        dcg_values.append(metrics[metric](relevances, scores, k, pscore_))

    return np.mean(dcg_values)
