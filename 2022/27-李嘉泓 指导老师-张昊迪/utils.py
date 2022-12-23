import json
import os
from pathlib import Path
from typing import Optional, Union, Iterable, List

import matplotlib
import numpy as np
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
import shutil


def freeze_model_weights(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def attention_mask_from_tokens(masks, token_list):
    mask_patterns = [["chief", "complaint", ":"],
                     ["present", "illness", ":"],
                     ["medical", "history", ":"],
                     ["medication", "on", "admission", ":"],
                     ["allergies", ":"],
                     ["physical", "exam", ":"],
                     ["family", "history", ":"],
                     ["social", "history", ":"],
                     ["[CLS]"],
                     ["[SEP]"],
                     ]

    for i, tokens in enumerate(token_list):
        for j, token in enumerate(tokens):
            for pattern in mask_patterns:
                if pattern == tokens[j:j + len(pattern)]:
                    masks[i, j:j + len(pattern)] = 0

    return masks


def get_bert_vectors_per_sample(batch, bert, use_cuda, linear=None):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_masks"]
    token_type_ids = batch["token_type_ids"]

    if use_cuda:
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()
        token_type_ids = token_type_ids.cuda()

    output = bert(input_ids=input_ids,
                  attention_mask=attention_mask,
                  token_type_ids=token_type_ids)

    if linear is not None:
        if use_cuda:
            linear = linear.cuda()
        token_vectors = linear(output.last_hidden_state)
    else:
        token_vectors = output.last_hidden_state

    mean_over_tokens = token_vectors.mean(dim=1)

    return mean_over_tokens, token_vectors


def get_attended_vector_per_sample(batch, bert, use_cuda, linear=None):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_masks"]
    token_type_ids = batch["token_type_ids"]

    if use_cuda:
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()
        token_type_ids = token_type_ids.cuda()

    output = bert(input_ids=input_ids,
                  attention_mask=attention_mask,
                  token_type_ids=token_type_ids)

    if linear is not None:
        if use_cuda:
            linear = linear.cuda()
        token_vectors = linear(output.last_hidden_state)
    else:
        token_vectors = output.last_hidden_state

    mean_over_tokens = token_vectors.mean(dim=1)

    return mean_over_tokens, token_vectors


def pad_batch_samples(batch_samples: Iterable, num_tokens: int) -> List:
    padded_samples = []
    for sample in batch_samples:
        missing_tokens = num_tokens - len(sample)
        tokens_to_append = ["[PAD]"] * missing_tokens
        padded_samples += sample + tokens_to_append
    return padded_samples


class ProjectorCallback(ModelCheckpoint):
    def __init__(
            self,
            train_dataloader,
            project_n_batches=-1,  # -1 means project all batches
            dirpath: Optional[Union[str, Path]] = None,
            filename: Optional[str] = None,
            monitor: Optional[str] = None,
            verbose: bool = False,
            save_last: Optional[bool] = None,
            save_top_k: Optional[int] = None,
            save_weights_only: bool = False,
            mode: str = "auto",
            period: int = 1,
            prefix: str = ""
    ):
        super().__init__(dirpath=dirpath, filename=filename, monitor=monitor, verbose=verbose, save_last=save_last,
                         save_top_k=save_top_k, save_weights_only=save_weights_only, mode=mode, period=period,
                         prefix=prefix)
        self.train_dataloader = train_dataloader
        self.project_n_batches = project_n_batches

    def on_validation_end(self, trainer, pl_module):
        """
        After each validation step, save the learned token and prototype embeddings for analysis in the Projector.
        """
        super().on_validation_end(trainer, pl_module)

        with torch.no_grad():

            all_vectors = []
            metadata = []
            for i, batch in enumerate(self.train_dataloader):
                _, _, batch_features = pl_module(batch, return_metadata=True)

                targets = batch["targets"]

                features = batch_features[0]
                tokens = batch_features[1]
                prototype_vectors = batch_features[2]

                batch_size = features.shape[0]

                window_len = features.shape[1]

                for sample_i in range(batch_size):
                    for window_i in range(window_len):
                        window_vector = features[sample_i][window_i]
                        window_tokens = tokens[sample_i * window_len + window_i]

                        if window_tokens == "[PAD]" or window_tokens == "[SEP]":
                            continue

                        all_vectors.append(window_vector)
                        metadata.append([window_tokens, targets[sample_i]])

                if ["PROTO_0", 0] not in metadata:
                    for j, vector in enumerate(prototype_vectors):
                        prototype_class = int(j // pl_module.prototypes_per_class)
                        all_vectors.append(vector.squeeze())
                        metadata.append([f"PROTO_{prototype_class}", prototype_class])

                if self.project_n_batches != -1 and i >= self.project_n_batches - 1:
                    break

            trainer.logger.experiment.add_embedding(torch.stack(all_vectors), metadata, global_step=trainer.global_step,
                                                    metadata_header=["tokens", "target"])

            delete_intermediate_embeddings(trainer.logger.experiment.log_dir, trainer.global_step)


def delete_intermediate_embeddings(log_dir, current_step):
    dir_content = os.listdir(log_dir)
    for file_or_dir in dir_content:
        try:
            file_as_integer = int(file_or_dir)
            abs_path = os.path.join(log_dir, file_or_dir)

            if os.path.isdir(abs_path) and file_as_integer != current_step and file_as_integer != 0:
                remove_dir(abs_path)

        except:
            continue

        embedding_config = """embeddings {{
            tensor_name: "default:{embedding_id}"
            metadata_path: "{embedding_id}/default/metadata.tsv"
            tensor_path: "{embedding_id}/default/tensors.tsv"\n}}"""

        config_text = embedding_config.format(embedding_id="00000") + "\n" + \
                      embedding_config.format(embedding_id=f"{current_step:05}")

        with open(os.path.join(log_dir, "projector_config.pbtxt"), "w") as config_file_write:
            config_file_write.write(config_text)


def remove_dir(path):
    try:
        shutil.rmtree(path)
        print(f"delete dir {path}")
    except OSError as e:
        print("Error: %s : %s" % (path, e.strerror))


def load_eval_buckets(eval_bucket_path):
    buckets = None
    if eval_bucket_path is not None:
        with open(eval_bucket_path) as bucket_file:
            buckets = json.load(bucket_file)
    return buckets


def build_heatmaps(case_tokens, token_scores, tint="red", amplifier=8):
    heatmap_per_prototype = []
    for prototype_scores in token_scores:

        template = '<span style="color: black; background-color: {}">{}</span>'
        heatmap_string = ''
        for word, color in zip(case_tokens, prototype_scores):
            color = min(1, color * amplifier)
            if tint == "red":
                hex_color = matplotlib.colors.rgb2hex([1, 1 - color, 1 - color])
            elif tint == "blue":
                hex_color = matplotlib.colors.rgb2hex([1 - color, 1 - color, 1])
            else:
                hex_color = matplotlib.colors.rgb2hex([1 - color, 1, 1 - color])

            if "##" not in word:
                heatmap_string += '&nbsp'
                word_string = word
            else:
                word_string = word.replace("##", "")

            heatmap_string += template.format(hex_color, word_string)

        heatmap_per_prototype.append(heatmap_string)

    return heatmap_per_prototype
