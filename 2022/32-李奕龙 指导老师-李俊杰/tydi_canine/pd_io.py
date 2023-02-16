# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import collections
import glob
import gzip
import json
from typing import Iterator, List, Sequence, Text
from tydi_canine import preproc
from tydi_canine import data
import logging
import numpy as np
from tydi_canine import tydi_tokenization_interface as tok_interface


def read_entries(input_jsonl_pattern: Text,
                 tokenizer: tok_interface.TokenizerWithOffsets,
                 max_passages: int, max_position: int, fail_on_invalid: bool):
    """Reads TyDi QA examples from JSONL files.

      Args:
        input_jsonl_path: path of the gzipped JSONL files to read.
        tokenizer: Used to create special marker symbols to insert into the text.
        max_passages: see FLAGS.max_passages.
        max_position: see FLAGS.max_position.
        fail_on_invalid: Immediately stop if an error is found?

      Yields:
        tuple:
          input_file: str
          line_no: int
          tydi_entry: "TyDiEntry"s, dicts as returned by `create_entry_from_json`,
            one per line of the input JSONL files.
          debug_info: Dict containing debugging data.
  """
    matches = glob.glob(input_jsonl_pattern, recursive=True)
    if not matches:
        raise ValueError(f"No files matched: {input_jsonl_pattern}")
    elif len(matches) > 1:
        logging.warning(f">>> more than one {input_jsonl_pattern} exists, {matches[0]} will be parsed.")
    input_jsonl_path = matches[0]
    logging.info(f">>> loading file from {input_jsonl_path}")
    with gzip.GzipFile(input_jsonl_path, "rb") as input_file:  # pytype: disable=wrong-arg-types
        for line_no, line in enumerate(input_file, 1):
            json_elem = json.loads(line, object_pairs_hook=collections.OrderedDict)
            entry = preproc.create_entry_from_json(
                json_elem,
                tokenizer,
                max_passages=max_passages,
                max_position=max_position,
                fail_on_invalid=fail_on_invalid)

            if not entry:
                logging.info("Invalid Example %d", json_elem["example_id"])
                if fail_on_invalid:
                    raise ValueError("Invalid example at {}:{}".format(
                        input_jsonl_path, line_no))

            # Return a `debug_info` dict that methods throughout the codebase
            # append to with debugging information.
            debug_info = {"json": json_elem}
            yield input_jsonl_path, line_no, entry, debug_info


class CreatePDExampleFn(object):
    """Functor for creating TyDi `tf.Example`s to be written to a TFRecord file."""

    def __init__(self, is_training, max_question_length, max_seq_length,
                 doc_stride, include_unknowns,
                 tokenizer: tok_interface.TokenizerWithOffsets):
        self.is_training = is_training
        self.tokenizer = tokenizer
        self.max_question_length = max_question_length
        self.max_seq_length = max_seq_length
        self.doc_stride = doc_stride
        self.include_unknowns = include_unknowns

    def process(self,
                entry,
                errors,
                debug_info=None):
        """Converts TyDi entries into serialized tf examples.
    Args:
      entry: "TyDi entries", dicts as returned by `create_entry_from_json`.
      errors: A list that this function appends to if errors are created. A
        non-empty list indicates problems.
      debug_info: A dict of information that may be useful during debugging.
        These elements should be used for logging and debugging only. For
        example, we log how the text was tokenized into WordPieces.
    Yields:
      `tf.train.Example` with the features needed for training or inference
      (depending on how `is_training` was set in the constructor).
    """
        if not debug_info:
            debug_info = {}
        tydi_example = data.to_tydi_example(entry, self.is_training)
        debug_info["tydi_example"] = tydi_example
        input_features = preproc.convert_single_example(
            tydi_example,
            tokenizer=self.tokenizer,
            is_training=self.is_training,
            max_question_length=self.max_question_length,
            max_seq_length=self.max_seq_length,
            doc_stride=self.doc_stride,
            include_unknowns=self.include_unknowns,
            errors=errors,
            debug_info=debug_info,
        )
        for input_feature in input_features:
            input_feature.example_index = int(entry["id"])
            input_feature.unique_id = (
                    input_feature.example_index + input_feature.doc_span_index)

            features = [input_feature.input_ids,
                        input_feature.input_mask,
                        input_feature.segment_ids]

            metas = [input_feature.unique_id,
                     input_feature.example_index,
                     input_feature.language_id]
            labels = None
            offsets = None

            if self.is_training:
                # there is no labels for testing dataset
                labels = [input_feature.start_position,
                          input_feature.end_position,
                          input_feature.answer_type
                          ]
            else:
                # this should significantly reduce the size of processed training dataset
                offsets = [input_feature.wp_start_offset,
                           input_feature.wp_end_offset]

            yield features, metas, labels, offsets
