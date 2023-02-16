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
from tydi_canine import char_splitter, pd_io
import json
from absl import app, logging
from absl import flags
import gzip
from tydi_canine import debug
import collections
import time
import h5py
from tydi_canine.h5df_config import *

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "input_jsonl", None,
    "Gzipped files containing NQ examples in Json format, one per line.")

flags.DEFINE_string("output_dir", None,
                    "Output directory for input files with all features extracted.")

flags.DEFINE_string("record_count_file", None,
                    "Output file that will contain a single integer "
                    "(the number of records in `output_tfrecord`). "
                    "This should always be used when generating training data "
                    "as it is required by the training program.")

flags.DEFINE_bool(
    "is_training", True,
    "Whether to prepare features for training or for evaluation. Eval features "
    "don't include gold labels")

flags.DEFINE_bool(
    "oversample", False,
    "Whether to sample languages with fewer data samples more heavily.")

flags.DEFINE_bool(
    "fail_on_invalid", True,
    "Stop immediately on encountering an invalid example? "
    "If false, just print a warning and skip it.")

flags.DEFINE_integer("max_oversample_ratio", 10,
                     "Maximum number a single example can be oversampled.")

flags.DEFINE_integer(
    "max_examples", 0,
    "If positive, stop once these many examples have been converted.")

flags.DEFINE_integer(
    "max_passages", 45, "Maximum number of passages to consider for a "
                        "single article. If an article contains more than"
                        "this, they will be discarded during training. "
                        "BERT's WordPiece vocabulary must be modified to include "
                        "these within the [unused*] vocab IDs.")

flags.DEFINE_integer(
    "max_position", 45,
    "Maximum passage position for which to generate special tokens.")

flags.DEFINE_integer(
    "max_seq_length", 512,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "doc_stride", 128,
    "When splitting up a long document into chunks, how much stride to "
    "take between chunks.")

flags.DEFINE_integer(
    "max_question_length", 64,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_float(
    "include_unknowns", -1.0,
    "If positive, probability of including answers of type `UNKNOWN`.")


def get_lang_counts(input_jsonl_path):
    """Gets the number of examples for each language."""
    lang_dict = collections.Counter()
    with gzip.GzipFile(input_jsonl_path, "rb") as input_file:  # pytype: disable=wrong-arg-types
        for line in input_file:
            json_elem = json.loads(line)
            lang_dict[json_elem["language"]] += 1
    return lang_dict


def main(args):
    time1 = time.time()
    examples_processed = 0
    num_examples_with_correct_context = 0
    num_errors = 0
    sample_ratio = {}

    # Print the first n examples to let user know what's going on.
    num_examples_to_print = 0
    if FLAGS.oversample and FLAGS.is_training:
        lang_count = get_lang_counts(input_jsonl_path=FLAGS.input_jsonl)
        max_count = max(count for lang, count in lang_count.items())
        for lang, curr_count in lang_count.items():
            sample_ratio[lang] = int(
                min(FLAGS.max_oversample_ratio, max_count / curr_count))

    splitter = char_splitter.CharacterSplitter()
    creator_fn = pd_io.CreatePDExampleFn(
        is_training=FLAGS.is_training,
        max_question_length=FLAGS.max_question_length,
        max_seq_length=FLAGS.max_seq_length,
        doc_stride=FLAGS.doc_stride,
        include_unknowns=FLAGS.include_unknowns,
        tokenizer=splitter)
    num_features = 0

    logging.info(f">>> input features will be store at {FLAGS.output_dir}")

    df = h5py.File(FLAGS.output_dir, 'w')

    # caches for h5df
    features, metas, labels, offsets = [], [], [], []

    def save_cache_to_h5df(h5df):
        """
        save all datas in features, metas, labels, offsets into target h5df
        """
        logging.info(f"save {len(features)} samples into h5df")
        save_h5_data(h5df, data=features, dset_name=feature_group_name)
        save_h5_data(h5df, data=metas, dset_name=meta_group_name)
        features.clear()
        metas.clear()
        if FLAGS.is_training:
            save_h5_data(h5df, data=labels, dset_name=label_group_name)
            labels.clear()
        else:
            save_h5_data(h5df, data=offsets, dset_name=offset_group_name)
            offsets.clear()

    for filename, line_no, entry, debug_info in pd_io.read_entries(
            input_jsonl_pattern=FLAGS.input_jsonl,
            tokenizer=splitter,
            max_passages=FLAGS.max_passages,
            max_position=FLAGS.max_position,
            fail_on_invalid=FLAGS.fail_on_invalid):
        errors = []
        for feature, meta, label, offset in creator_fn.process(entry, errors, debug_info):
            if FLAGS.oversample:  # code for oversample will not use for canine
                for _ in range(sample_ratio[entry["language"]]):
                    features.append(feature)
                    metas.append(meta)
                    if FLAGS.is_training:
                        labels.append(label)
                    else:
                        offsets.append(offset)
                    num_features += 1
            else:
                features.append(feature)
                metas.append(meta)
                if FLAGS.is_training:
                    labels.append(label)
                else:
                    offsets.append(offset)
                num_features += 1

        if errors or examples_processed < num_examples_to_print:
            debug.log_debug_info(filename, line_no, entry, debug_info,
                                 splitter.id_to_string)

        if examples_processed % 100 == 0:
            logging.info("Examples processed: %d", examples_processed)
        examples_processed += 1

        if errors:
            logging.info(
                "Encountered errors while creating {} example ({}:{}): {}".format(
                    entry["language"], filename, line_no, "; ".join(errors)))
            if FLAGS.fail_on_invalid:
                raise ValueError(
                    "Encountered errors while creating example ({}:{}): {}".format(
                        filename, line_no, "; ".join(errors)))
            num_errors += 1
            if num_errors % 10 == 0:
                logging.info("Errors so far: %d", num_errors)
        if entry["has_correct_context"]:
            num_examples_with_correct_context += 1
        if 0 < FLAGS.max_examples <= examples_processed:
            break
        if len(features) >= 25600:
            save_cache_to_h5df(df)

    save_cache_to_h5df(df)

    logging.info("Examples with correct context retained: %d of %d",
                 num_examples_with_correct_context, examples_processed)

    # Even though the input is shuffled, we need to do this in case we're
    # oversampling.
    # If not oversampling, shuffle is not need. For canine, shuffle is no need
    logging.info("Number of total features %d", num_features)

    df.close()
    if FLAGS.record_count_file:
        with open(FLAGS.record_count_file, "w") as writer:
            writer.write(str(num_features))

    print(f"time cose: {(time.time() - time1) / 60:.2f} min")


if __name__ == "__main__":
    app.run(main)
