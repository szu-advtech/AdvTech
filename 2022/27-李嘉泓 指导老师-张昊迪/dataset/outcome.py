from typing import List, Dict

import torch
from pandas import DataFrame
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from transformers import PreTrainedTokenizer
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer


def collate_batch(featurized_samples: List[Dict]):
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(x['input_ids']) for x in featurized_samples],
        batch_first=True)

    attention_masks = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(x['attention_mask']) for x in featurized_samples],
        batch_first=True)

    batch = {"input_ids": input_ids,
             "attention_masks": attention_masks,
             "tokens": [x['tokens'] for x in featurized_samples],
             "targets": np.array([x['target'] for x in featurized_samples]),
             "sample_ids": [x['sample_id'] for x in featurized_samples]}

    if 'token_type_ids' in featurized_samples[0]:
        token_type_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x['token_type_ids']) for x in featurized_samples],
            batch_first=True)
        batch["token_type_ids"] = token_type_ids

    return batch


def sample_to_features_multilabel(sample: pd.Series, tokenizer: PreTrainedTokenizer, labels: List[str], max_length=512,
                                  text_column="text") -> Dict:
    tokenized = tokenizer.encode_plus(sample[text_column],
                                      padding=True,
                                      truncation=True,
                                      pad_to_multiple_of=512 if max_length > 512 else None,
                                      max_length=max_length)

    featurized_sample = {"input_ids": tokenized["input_ids"],
                         "attention_mask": tokenized["attention_mask"],
                         "tokens": tokenized.encodings[0].tokens,
                         "target": sample[labels].to_numpy().astype(int),
                         "sample_id": sample["id"]}

    if "token_type_ids" in tokenized:
        featurized_sample["token_type_ids"] = tokenized["token_type_ids"]

    return featurized_sample


class OutcomeDiagnosesDataset(Dataset):

    def __init__(self,
                 file_path,
                 tokenizer: PreTrainedTokenizer,
                 all_codes_path,
                 max_length=512,
                 text_column='text',
                 label_column='short_codes'):
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column

        self.data: DataFrame = pd.read_csv(file_path, dtype={"id": str})

        # binarize labels
        mlb = MultiLabelBinarizer()

        with open(all_codes_path) as all_codes_file:
            all_codes = all_codes_file.read().split(" ")

        mlb.fit([all_codes])
        binary_labels_set = mlb.transform(self.data[label_column].str.split(","))
        self.labels = mlb.classes_
        self.data[self.labels] = binary_labels_set

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> T_co:
        featurized_sample = sample_to_features_multilabel(sample=self.data.iloc[index],
                                                          tokenizer=self.tokenizer,
                                                          labels=self.labels,
                                                          max_length=self.max_length,
                                                          text_column=self.text_column)
        return featurized_sample

    def get_num_classes(self):
        return len(self.labels)
