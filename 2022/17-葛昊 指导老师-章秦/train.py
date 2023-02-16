import argparse
import csv
import sys
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from transformers import AdamW, AutoModel, AutoTokenizer
from tqdm import tqdm

import os
import ast

csv.field_size_limit(sys.maxsize)

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0, help='CUDA device')
parser.add_argument('--model', type=str, help='pre-trained model (bert-base-uncased, roberta-base)')
parser.add_argument('--task', type=str, help='task name (SNLI, MNLI, QQP, TwitterPPDB, SWAG, HellaSWAG)')
parser.add_argument('--max_seq_length', type=int, default=256, help='max sequence length')
parser.add_argument('--ckpt_path', type=str, help='model checkpoint path')
parser.add_argument('--output_path', type=str, help='model output path')
parser.add_argument('--train_path', type=str, help='train dataset path')
parser.add_argument('--dev_path', type=str, help='dev dataset path')
parser.add_argument('--test_path', type=str, help='test dataset path')
parser.add_argument('--epochs', type=int, default=3, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--learning_rate', type=float, default=2e-5, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0., help='weight decay')
parser.add_argument('--max_grad_norm', type=float, default=1., help='gradient clip')
parser.add_argument('--do_train', action='store_true', default=False, help='enable training')
parser.add_argument('--do_evaluate', action='store_true', default=False, help='enable evaluation')
parser.add_argument('--train_from', type=str, help='load checkpoint')
parser.add_argument('--multigpu', action='store_true', help='Multiple GPU')
parser.add_argument('--ls', action='store_true', help='enable label smoothing')
parser.add_argument('--wo_similar', action='store_true',
                    help='without using the most similar instance in the other category')
parser.add_argument('--wo_dissimilar', action='store_true',
                    help='without using the most dissimilar instance in the other category')
args = parser.parse_args()
print(args)

if args.multigpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
    args.device = torch.device("cuda")

assert args.task in ('SNLI', 'MNLI', 'QQP', 'TwitterPPDB', 'SWAG', 'HellaSWAG')
assert args.model in ('bert-base-uncased', 'roberta-base')

if args.task in ('SNLI', 'MNLI'):
    n_classes = 3
elif args.task in ('QQP', 'TwitterPPDB'):
    n_classes = 2
elif args.task in ('SWAG', 'HellaSWAG'):
    n_classes = 1


def cuda(tensor):
    return tensor.to(args.device)


def load(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size, shuffle=shuffle)


def adamw_params(model):

    no_decay = ['bias', 'LayerNorm.weight']
    params = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        },
    ]
    return params


def encode_pair_inputs(sentence1, sentence2, passed_tokenizer):

    inputs = passed_tokenizer.encode_plus(
        sentence1, sentence2, add_special_tokens=True, max_length=args.max_seq_length
    )
    input_ids = inputs['input_ids']
    if args.model == 'bert-base-uncased':
        segment_ids = inputs['token_type_ids']
    attention_mask = inputs['attention_mask']
    padding_length = args.max_seq_length - len(input_ids)
    input_ids += [passed_tokenizer.pad_token_id] * padding_length
    if args.model == 'bert-base-uncased':
        segment_ids += [0] * padding_length
    attention_mask += [0] * padding_length

    if args.model == 'bert-base-uncased':
        for input_elem in (input_ids, segment_ids, attention_mask):
            assert len(input_elem) == args.max_seq_length
        return (
            cuda(torch.tensor(input_ids)).long(),
            cuda(torch.tensor(segment_ids)).long(),
            cuda(torch.tensor(attention_mask)).long(),
        )
    else:
        for input_elem in (input_ids, attention_mask):
            assert len(input_elem) == args.max_seq_length
        return (
            cuda(torch.tensor(input_ids)).long(),
            cuda(torch.tensor(attention_mask)).long(),
        )


def encode_mc_inputs(context, start_ending, endings, tokenizer):
    context_tokens = tokenizer.tokenize(context)
    start_ending_tokens = tokenizer.tokenize(start_ending)
    all_input_ids = []
    all_segment_ids = []
    all_attention_masks = []
    for ending in endings:
        ending_tokens = start_ending_tokens + tokenizer.tokenize(ending)
        inputs = tokenizer.encode_plus(
            " ".join(context_tokens), " ".join(ending_tokens), add_special_tokens=True, max_length=args.max_seq_length
        )
        input_ids = inputs['input_ids']
        if args.model == 'bert-base-uncased':
            segment_ids = inputs['token_type_ids']
        attention_mask = inputs['attention_mask']
        padding_length = args.max_seq_length - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * padding_length
        if args.model == 'bert-base-uncased':
            segment_ids += [0] * padding_length
        attention_mask += [0] * padding_length

        if args.model == 'bert-base-uncased':
            for input_elem in (input_ids, segment_ids, attention_mask):
                assert len(input_elem) == args.max_seq_length
            all_input_ids.append(input_ids)
            all_segment_ids.append(segment_ids)
            all_attention_masks.append(attention_mask)
        else:
            for input_elem in (input_ids, attention_mask):
                assert len(input_elem) == args.max_seq_length
            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
    if args.model == 'bert-base-uncased':
        return (
            cuda(torch.tensor(all_input_ids)).long(),
            cuda(torch.tensor(all_segment_ids)).long(),
            cuda(torch.tensor(all_attention_masks)).long(),
        )
    else:
        return (
            cuda(torch.tensor(all_input_ids)).long(),
            cuda(torch.tensor(all_attention_masks)).long(),
        )


def encode_label(label):
    """Wraps label in tensor."""

    return cuda(torch.tensor(label)).long()


class SNLIProcessor:
    """Data loader for SNLI."""

    def __init__(self):
        self.label_map = {'entailment': 0, 'contradiction': 1, 'neutral': 2}

    def valid_inputs(self, sentence1, sentence2, label):
        return len(sentence1) > 0 and len(sentence2) > 0 and label in self.label_map

    def load_samples(self, path, guided=False):
        samples = []
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)  # skip header
            desc = f'loading \'{path}\''
            for row in tqdm(reader, desc=desc):
                try:
                    guid = row[1]
                    sentence1 = row[4]
                    sentence2 = row[7]
                    label = row[2]
                    if guided:
                        grads = ast.literal_eval(row[-1])
                        grads = [float(x) for x in grads]
                    else:
                        grads = []
                    if self.valid_inputs(sentence1, sentence2, label):
                        label = self.label_map[label]
                        samples.append((sentence1, sentence2, label, guid, grads))
                except:
                    pass
        return samples


class MNLIProcessor(SNLIProcessor):
    """Data loader for MNLI."""

    def load_samples(self, path, guided=False):
        samples = []
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)  # skip header
            desc = f'loading \'{path}\''
            for row in tqdm(reader, desc=desc):
                try:
                    sentence1 = row[8]
                    sentence2 = row[9]
                    label = row[-1]
                    if self.valid_inputs(sentence1, sentence2, label):
                        label = self.label_map[label]
                        samples.append((sentence1, sentence2, label, [], []))
                except:
                    pass
        return samples


class QQPProcessor:
    """Data loader for QQP."""

    def valid_inputs(self, sentence1, sentence2, label):
        return len(sentence1) > 0 and len(sentence2) > 0 and label in ('0', '1')

    def load_samples(self, path, guided=False):
        samples = []
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)  # skip header
            desc = f'loading \'{path}\''
            for row in tqdm(reader, desc=desc):
                try:
                    guid = row[0]
                    sentence1 = row[4]
                    sentence2 = row[5]
                    label = row[1]
                    if guided:
                        grads = ast.literal_eval(row[-1])
                        grads = [float(x) for x in grads]
                    else:
                        grads = []
                    if self.valid_inputs(sentence1, sentence2, label):
                        label = int(label)
                        samples.append((sentence1, sentence2, label, guid, grads))
                except:
                    pass
        return samples


class TwitterPPDBProcessor:
    """Data loader for TwittrPPDB."""

    def valid_inputs(self, sentence1, sentence2, label):
        return len(sentence1) > 0 and len(sentence2) > 0 and label != 3

    def load_samples(self, path, guided=False):
        samples = []
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            desc = f'loading \'{path}\''
            for row in tqdm(reader, desc=desc):
                try:
                    sentence1 = row[0]
                    sentence2 = row[1]
                    label = eval(row[2])[0]
                    if self.valid_inputs(sentence1, sentence2, label):
                        label = 0 if label < 3 else 1
                        samples.append((sentence1, sentence2, label, [], []))
                except:
                    pass
        return samples


class SWAGProcessor:
    """Data loader for SWAG."""

    def load_samples(self, path, guided=False):
        samples = []
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            desc = f'loading \'{path}\''
            for row in tqdm(reader, desc=desc):
                try:
                    guid = row[5]
                    context = str(row[0])
                    start_ending = str(row[-1])
                    endings = row[1:5]
                    label = int(row[6])
                    if guided:
                        grads = ast.literal_eval(row[-1])
                        grads = [float(x) for x in grads]
                    else:
                        grads = []
                    samples.append((context, start_ending, endings, label, guid, grads))
                except:
                    pass
        return samples


class HellaSWAGProcessor:
    """Data loader for HellaSWAG."""

    def load_samples(self, path, guided=False):
        samples = []
        with open(path) as f:
            desc = f'loading \'{path}\''
            for line in f:
                try:
                    line = line.rstrip()
                    input_dict = json.loads(line)
                    context = input_dict['ctx_a']
                    start_ending = input_dict['ctx_b']
                    endings = input_dict['endings']
                    label = input_dict['label']
                    samples.append((context, start_ending, endings, label, [], []))
                except:
                    pass
        return samples


def select_processor():
    """Selects data processor using task name."""

    return globals()[f'{args.task}Processor']()


class TextDataset(Dataset):

    def __init__(self, path, processor, passed_tokenizer, guided=False):
        self.samples = processor.load_samples(path, guided=guided)
        self.cache = {}
        self.tokenizer = passed_tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        res = self.cache.get(i, None)
        if res is None:
            sample = self.samples[i]
            if args.task in ('SNLI', 'MNLI', 'QQP', 'MRPC', 'TwitterPPDB'):
                sentence1, sentence2, label, guid, grads = sample
                if args.model == 'bert-base-uncased':
                    input_ids, segment_ids, attention_mask = encode_pair_inputs(
                        sentence1, sentence2, self.tokenizer
                    )
                else:
                    input_ids, attention_mask = encode_pair_inputs(
                        sentence1, sentence2, self.tokenizer
                    )
                packed_inputs = (sentence1, sentence2)
            elif args.task in ('SWAG', 'HellaSWAG'):
                if args.model == 'bert-base-uncased':
                    context, ending_start, endings, label, guid, grads = sample
                    input_ids, segment_ids, attention_mask = encode_mc_inputs(
                        context, ending_start, endings, self.tokenizer
                    )
                else:
                    context, ending_start, endings, label, guid, grads = sample
                    input_ids, attention_mask = encode_mc_inputs(
                        context, ending_start, endings, self.tokenizer
                    )
            label_id = encode_label(label)
            if grads is not None:
                grads = cuda(torch.tensor(grads))
            if args.model == 'bert-base-uncased':
                res = ((input_ids, segment_ids, attention_mask), label_id, guid, grads)
            else:
                res = ((input_ids, attention_mask), label_id, guid, grads)
            self.cache[i] = res
        return res


class Model(nn.Module):
    """Pre-trained model for classification."""

    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained(args.model)
        self.classifier = nn.Linear(768, n_classes)
        if args.task in ("SWAG"):
            self.n_choices = -1

    def forward(self, input_ids, segment_ids, attention_mask):
        # On SWAG and HellaSWAG, collapse the batch size and
        # choice size dimension to process everything at once
        if args.task in ('SWAG', 'HellaSWAG'):
            self.n_choices = input_ids.size(1)
            input_ids = input_ids.view(-1, input_ids.size(-1))
            if args.model == 'bert-base-uncased':
                segment_ids = segment_ids.view(-1, segment_ids.size(-1))
            attention_mask = attention_mask.view(-1, attention_mask.size(-1))

        transformer_params = {
            'input_ids': input_ids,
            'token_type_ids': (
                segment_ids if args.model == 'bert-base-uncased' else None
            ),
            'attention_mask': attention_mask,
        }

        transformer_outputs = self.model(**transformer_params)
        if args.task in ('SWAG', 'HellaSWAG'):
            pooled_output = transformer_outputs[1]
            logits = self.classifier(pooled_output)
            logits = logits.view(-1, self.n_choices)
        else:
            cls_output = transformer_outputs[0][:, 0]
            logits = self.classifier(cls_output)
        return logits


def smoothing_label(target, smoothing):
    """Label smoothing"""
    _n_classes = n_classes if args.task not in ('SWAG', 'HellaSWAG') else 4
    confidence = 1. - smoothing
    smoothing_value = smoothing / (_n_classes - 1)
    one_hot = cuda(torch.full((_n_classes,), smoothing_value))
    model_prob = one_hot.repeat(target.size(0), 1)
    model_prob.scatter_(1, target.unsqueeze(1), confidence)
    return model_prob


def train(d1, d2=None):
    """Fine-tunes pre-trained model on training set."""

    model.train()
    train_loss = 0.
    train_loader1 = tqdm(load(d1, args.batch_size, True))
    optimizer = AdamW(adamw_params(model), lr=args.learning_rate, eps=1e-8)
    for i, dataset in enumerate(train_loader1):
        inputs, label, guid, grads = dataset
        optimizer.zero_grad()
        if args.model == 'bert-base-uncased':
            logits = model(*inputs)
        else:
            logits = model(inputs[0], None, inputs[1])
        if args.ls:
            if args.model == 'bert-base-uncased':
                if args.task == 'SNLI':
                    smoothing_value = 0.001
                elif args.task == 'QQP':
                    smoothing_value = 0.03
                elif args.task == 'SWAG':
                    smoothing_value = 0.3
            elif args.model == 'roberta-base':
                if args.task == 'SNLI':
                    smoothing_value = 0.003
                elif args.task == 'QQP':
                    smoothing_value = 0.03
                elif args.task == 'SWAG':
                    smoothing_value = 0.3
            label = smoothing_label(label, smoothing_value)
            # loss = torch.mean(torch.sum(-label * torch.log_softmax(logits, dim=-1), dim=0))
            loss = F.kl_div(F.log_softmax(logits, 1), label, reduction='sum')
        else:
            loss = criterion(logits, label)

        loss.backward()
        train_loss += loss.item()
        if i == 0:
            pass
        else:
            train_loader1.set_description(f'train loss = {(train_loss / i):.6f}')
        if args.max_grad_norm > 0.:
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
    return train_loss / len(train_loader1)


def evaluate(dataset):
    """Evaluates pre-trained model on development set."""

    model.eval()
    eval_loss = 0.
    eval_loader = tqdm(load(dataset, args.batch_size, False))
    # for i, (inputs, label, guid) in enumerate(eval_loader, 1):
    for i, d in enumerate(eval_loader):
        inputs, label, _, _ = d
        with torch.no_grad():
            if args.model == 'bert-base-uncased':
                logits = model(*inputs)
            else:
                logits = model(inputs[0], None, inputs[1])
            loss = criterion(logits, label)

        eval_loss += loss.item()
        if i == 0:
            pass
        else:
            eval_loader.set_description(f'eval loss = {(eval_loss / i):.6f}')
    return eval_loss / len(eval_loader)


if args.multigpu:
    model = Model()
    model = nn.DataParallel(model).to(args.device)
else:
    model = cuda(Model())
if args.train_from:
    model.load_state_dict(torch.load(args.train_from))
processor = select_processor()
tokenizer = AutoTokenizer.from_pretrained(args.model)

criterion = nn.CrossEntropyLoss()

if args.train_path:
    train_dataset = TextDataset(args.train_path, processor, tokenizer)
    print(f'train samples = {len(train_dataset)}')
if args.dev_path:
    dev_dataset = TextDataset(args.dev_path, processor, tokenizer)
    print(f'dev samples = {len(dev_dataset)}')
if args.test_path:
    test_dataset = TextDataset(args.test_path, processor, tokenizer)
    print(f'test samples = {len(test_dataset)}')

if args.do_train:
    print('*** training ***')
    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        train_loss = train(train_dataset)
        eval_loss = evaluate(dev_dataset)
        if eval_loss < best_loss:
            best_loss = eval_loss
            torch.save(model.state_dict(), args.ckpt_path)
        print(
            f'epoch = {epoch} | '
            f'train loss = {train_loss:.6f} | '
            f'eval loss = {eval_loss:.6f}'
        )

if args.do_evaluate:
    if not os.path.exists(args.ckpt_path):
        raise RuntimeError(f'\'{args.ckpt_path}\' does not exist')

    print()
    print('*** evaluating ***')

    output_dicts = []
    model.load_state_dict(torch.load(args.ckpt_path))
    model.eval()
    test_loader = tqdm(load(test_dataset, args.batch_size, False))

    for i, d in enumerate(test_loader):
        inputs, label, _, _ = d
        with torch.no_grad():
            if args.model == 'bert-base-uncased':
                logits = model(*inputs)
            else:
                logits = model(inputs[0], None, inputs[1])
            loss = criterion(logits, label)

            for j in range(logits.size(0)):
                probs = F.softmax(logits[j], -1)
                output_dict = {
                    'index': args.batch_size * i + j,
                    'true': label[j].item(),
                    'pred': logits[j].argmax().item(),
                    'conf': probs.max().item(),
                    'logits': logits[j].cpu().numpy().tolist(),
                    'probs': probs.cpu().numpy().tolist(),
                }
                output_dicts.append(output_dict)

    print(f'writing outputs to \'{args.output_path}\'')
    with open(args.output_path, 'w+') as f:
        for i, output_dict in enumerate(output_dicts):
            output_dict_str = json.dumps(output_dict)
            f.write(f'{output_dict_str}\n')

    y_true = [output_dict['true'] for output_dict in output_dicts]
    y_pred = [output_dict['pred'] for output_dict in output_dicts]
    y_conf = [output_dict['conf'] for output_dict in output_dicts]

    accuracy = accuracy_score(y_true, y_pred) * 100.
    f1 = f1_score(y_true, y_pred, average='macro') * 100.
    confidence = np.mean(y_conf) * 100.

    results_dict = {
        'accuracy': accuracy_score(y_true, y_pred) * 100.,
        'macro-F1': f1_score(y_true, y_pred, average='macro') * 100.,
        'confidence': np.mean(y_conf) * 100.,
    }
    for k, v in results_dict.items():
        print(f'{k} = {v}')
