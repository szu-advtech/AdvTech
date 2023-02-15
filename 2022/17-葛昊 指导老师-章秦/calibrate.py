import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score


parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0, help='CUDA device')
parser.add_argument('--buckets', type=int, default=10, help='number of calibration buckets')
parser.add_argument('--temperature', type=float, default=1., help='softmax temperature')
parser.add_argument('--train_path', type=str, help='training output file')
parser.add_argument('--test_path', type=str, help='testing output file')
parser.add_argument('--label_smoothing', type=float, default=0., help='label smoothing \\alpha')
parser.add_argument('--do_train', action='store_true', default=False, help='enable training')
parser.add_argument('--do_evaluate', action='store_true', default=False, help='enable evaluation')
args = parser.parse_args()
print(args)


def load_output(path):

    with open(path) as f:
        elems = [json.loads(l.rstrip()) for l in f]
        for elem in elems:
            elem['true'] = torch.tensor(elem['true']).long()
            elem['logits'] = torch.tensor(elem['logits']).float()
        return elems


def get_bucket_scores(y_score):

    bucket_values = [[] for _ in range(args.buckets)]
    bucket_indices = [[] for _ in range(args.buckets)]
    for i, score in enumerate(y_score):
        for j in range(args.buckets):
            if score < float((j + 1) / args.buckets):
                break
        bucket_values[j].append(score)
        bucket_indices[j].append(i)
    return (bucket_values, bucket_indices)


def get_bucket_confidence(bucket_values):

    return [
        np.mean(bucket)
        if len(bucket) > 0 else -1.
        for bucket in bucket_values
    ]


def get_bucket_accuracy(bucket_values, y_true, y_pred):

    per_bucket_correct = [
        [int(y_true[i] == y_pred[i]) for i in bucket]
        for bucket in bucket_values
    ]
    return [
        np.mean(bucket)
        if len(bucket) > 0 else -1.
        for bucket in per_bucket_correct
    ]


def calculate_error(n_samples, bucket_values, bucket_confidence, bucket_accuracy):

    assert len(bucket_values) == len(bucket_confidence) == len(bucket_accuracy)
    assert sum(map(len, bucket_values)) == n_samples

    expected_error, max_error, total_error = 0., 0., 0.
    for (bucket, accuracy, confidence) in zip(
        bucket_values, bucket_accuracy, bucket_confidence
    ):
        if len(bucket) > 0:
            delta = abs(accuracy - confidence)
            expected_error += (len(bucket) / n_samples) * delta
            max_error = max(max_error, delta)
            total_error += delta
    return (expected_error * 100., max_error * 100., total_error * 100.)


def create_one_hot(n_classes):

    smoothing_value = args.label_smoothing / (n_classes - 1)
    one_hot = torch.full((n_classes,), smoothing_value).float()
    return one_hot


def cross_entropy(output, target, n_classes):

    model_prob = create_one_hot(n_classes)
    model_prob[target] = 1. - args.label_smoothing
    return F.kl_div(output, model_prob, reduction='sum').item()


if args.do_train:
    elems = load_output(args.train_path)
    n_classes = len(elems[0]['logits'])

    best_nll = float('inf')
    best_temperature = -1

    temp_values = map(lambda x: round(x / 100 + 0.01, 2), range(1000))

    for temp in tqdm(temp_values, leave=False, desc='training'):
        nll = np.mean(
            [
                cross_entropy(
                    F.log_softmax(elem['logits'] / temp, 0), elem['true'], n_classes
                )
                for elem in elems
            ]
        )
        if nll < best_nll:
            best_nll = nll
            best_temp = temp

    args.temperature = best_temp

    output_dict = {'temperature': best_temp}

    print()
    print('*** training ***')
    for k, v in output_dict.items():
        print(f'{k} = {v}')


if args.do_evaluate:
    elems = load_output(args.test_path)
    n_classes = len(elems[0]['logits'])

    labels = [elem['true'] for elem in elems]
    preds = [elem['pred'] for elem in elems]

    log_probs = [F.log_softmax(elem['logits'] / args.temperature, 0) for elem in elems]
    confs = [prob.exp().max().item() for prob in log_probs]
    nll = [
        cross_entropy(log_prob, label, n_classes)
        for log_prob, label in zip(log_probs, labels)
    ]

    bucket_values, bucket_indices = get_bucket_scores(confs)
    bucket_confidence = get_bucket_confidence(bucket_values)
    bucket_accuracy = get_bucket_accuracy(bucket_indices, labels, preds)

    accuracy = accuracy_score(labels, preds) * 100.
    avg_conf = np.mean(confs) * 100.
    avg_nll = np.mean(nll)
    expected_error, max_error, total_error = calculate_error(
        len(elems), bucket_values, bucket_confidence, bucket_accuracy
    )

    output_dict = {
        'accuracy': accuracy,
        'confidence': avg_conf,
        'temperature': args.temperature,
        'neg log likelihood': avg_nll,
        'expected error': expected_error,
        'max error': max_error,
        'total error': total_error,
    }

    print()
    print('*** evaluating ***')
    for k, v in output_dict.items():
        print(f'{k} = {v}')
