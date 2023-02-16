import abc
import collections
import glob
import gzip
import json
import os
import pickle
from typing import Optional, Text

import h5py
import paddle
import time
from paddle.amp import GradScaler, auto_cast
import paddlenlp

from tydi_canine import postproc
from tydi_canine.tydi_modeling import CanineForTydiQA, CrossEntropyLossForTydi
from tydi_canine.data_utils import get_dataloader
import logging
import random
import numpy as np
from paddle import distributed as dist
from tydi_canine.h5df_config import *


logger = logging.getLogger(__name__)


class TyDiRunner(metaclass=abc.ABCMeta):
    """See run_tydi.py. All attributes are copied directly from the args."""

    def __init__(
            self,
            # IO args
            output_dir: Optional[Text] = None,
            train_input_dir: Optional[Text] = None,
            state_dict_path: Optional[Text] = None,
            predict_file: Optional[Text] = None,
            precomputed_predict_file: Optional[Text] = None,
            output_prediction_file: Optional[Text] = None,

            # running mode
            do_train: bool = False,
            do_predict: bool = False,
            do_file_construct: bool = False,

            # fine-tuning args
            dev_split_ratio: float = 0.01,
            fp16: bool = False,
            scale_loss=2 ** 15,
            gradient_accumulation_steps=1,
            seed: int = 7,
            checkout_steps: int = 200,
            max_seq_length: int = 512,
            train_batch_size: int = 16,
            predict_batch_size: int = 8,
            learning_rate: float = 5e-5,
            num_train_epochs: float = 3.0,
            warmup_proportion: float = 0.1,
            logging_steps: int = 1000,

            # evaluation args
            candidate_beam: int = 30,
            max_answer_length: int = 100,
            max_position: int = 45,
            max_to_predict: Optional[int] = None,
    ):
        self.output_dir = output_dir
        self.train_input_dir = train_input_dir
        self.state_dict_path = state_dict_path
        self.predict_file = predict_file
        self.precomputed_predict_file = precomputed_predict_file
        self.output_prediction_file = output_prediction_file

        self.checkout_steps = checkout_steps
        self.max_seq_length = max_seq_length
        self.do_train = do_train
        self.do_predict = do_predict
        self.do_file_construct = do_file_construct
        self.train_batch_size = train_batch_size
        self.predict_batch_size = predict_batch_size
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.warmup_proportion = warmup_proportion
        self.max_answer_length = max_answer_length
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.logging_steps = logging_steps
        self.scale_loss = scale_loss
        self.seed = seed
        self.dev_split_ratio = dev_split_ratio
        self.max_position = max_position
        self.max_to_predict = max_to_predict
        self.fp16 = fp16
        self.candidate_beam = candidate_beam

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.logger_init()
        self.save_weight_file_name = f"tydi_seed_{self.seed}.pdparams"
        if self.logging_steps is None:
            self.logging_steps = 100

    def logger_init(self):
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[
                logging.FileHandler(
                    os.path.join(self.output_dir, f"run_seed_{self.seed}.log"),
                    mode="w",
                    encoding="utf-8", ),
                logging.StreamHandler()
            ], )
        logger.info("**********  Configuration Arguments **********")
        for arg, value in sorted(vars(self).items()):
            logger.info(f"{arg}: {value}")
        logger.info("**************************************************")

    def load_model(self):
        model = CanineForTydiQA.from_pretrained('canine-s')
        if self.max_seq_length > model.canine.max_position_embeddings:
            raise ValueError(
                f"Cannot use sequence length {self.max_seq_length} "
                "because the CANINE model was only trained up to sequence length "
                f"{model.canine.max_position_embeddings}")
        loaded_state_dict = False
        if self.state_dict_path is not None:
            if os.path.isdir(self.state_dict_path):
                state_dict_file = os.path.join(self.state_dict_path, self.save_weight_file_name)
                if os.path.exists(state_dict_file):
                    logger.info(f"\n>>>>>> loading weight from {state_dict_file}")
                    load_state_dict = paddle.load(state_dict_file)
                    model.set_state_dict(load_state_dict)
                    loaded_state_dict = True
            else:
                logger.info(f"\n>>>>>> loading weight from {self.state_dict_path}")
                load_state_dict = paddle.load(self.state_dict_path)
                model.set_state_dict(load_state_dict)
                loaded_state_dict = True

        if dist.get_world_size() > 1:
            model = paddle.DataParallel(model)
        if self.do_predict and not loaded_state_dict:
            raise ValueError(f"no state dict found in path {self.state_dict_path}")
        return model

    def run(self):
        if self.do_train:
            self.train()
        if self.do_predict:
            self.predict()
            self.gen_tydi_result()
        if self.do_file_construct:
            self.gen_tydi_result()

    def save_ckpt(self, model, name="tydi.pdparams"):
        if dist.get_rank() == 0:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            paddle.save(model.state_dict(), os.path.join(self.output_dir, name))

    def split_training_samples(self, shuffle=False):
        """
        Indeed, paddle dataloader and sampler provided shuffle and index, but
        this is just in order to split the train and dev set from h5df database.

        Returns:
             List[int],List[int]: sample ids for train samples and dev samples.
        """
        with h5py.File(self.train_input_dir, 'r') as fp:
            num_samples = fp[feature_group_name].len()
        sample_ids = list(range(num_samples))
        if shuffle:
            random.shuffle(sample_ids)
        split = int(num_samples * self.dev_split_ratio)

        return sample_ids[split:], sample_ids[:split]

    def train(self):
        """
        Please refer to `note.md` at root directory for more information about hyperparameter and training
        settings.
        """
        if dist.get_world_size() > 1:
            dist.init_parallel_env()
        random.seed(self.seed)
        np.random.seed(self.seed)
        paddle.seed(self.seed)

        model = self.load_model()

        train_samples_ids, dev_samples_ids = self.split_training_samples()

        train_data_loader = get_dataloader(sample_ids=train_samples_ids,
                                           h5df_path=self.train_input_dir,
                                           batch_size=self.train_batch_size,
                                           is_train=True)

        if self.predict_batch_size is None:
            self.predict_batch_size = self.train_batch_size

        dev_data_loader = get_dataloader(sample_ids=dev_samples_ids,
                                         h5df_path=self.train_input_dir,
                                         batch_size=self.predict_batch_size,
                                         is_train=True)

        num_train_steps = int(len(train_data_loader) // self.gradient_accumulation_steps *
                              self.num_train_epochs)

        logger.info(">>> num_training_samples: %d", len(train_samples_ids))
        logger.info(">>> num_dev_samples: %d", len(dev_samples_ids))
        logger.info(">>> theory batch size (batch * gradient step * n_gpus): %d",
                    self.train_batch_size * self.gradient_accumulation_steps * dist.get_world_size())
        logger.info(">>> num_train_steps for each GPU: %d", num_train_steps)

        lr_scheduler = paddlenlp.transformers.LinearDecayWithWarmup(
            learning_rate=self.learning_rate,
            total_steps=num_train_steps,
            warmup=int(num_train_steps * self.warmup_proportion))

        decay_params = [
            p.name for n, p in model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm"])
        ]

        optimizer = paddle.optimizer.AdamW(
            learning_rate=lr_scheduler,
            parameters=model.parameters(),
            epsilon=1e-6,
            apply_decay_param_fun=lambda x: x in decay_params,
        )
        criterion = CrossEntropyLossForTydi()

        scaler = None
        if self.fp16:
            scaler = GradScaler(init_loss_scaling=self.scale_loss)

        model.train()
        global_step = 0
        time1 = time.time()
        optimizer.clear_grad()
        losses = paddle.to_tensor([0.0])
        loss_list, dev_loss_list, acc_list, diff_list = [], [], [], []  # for multi-gpu
        logger.info(">>> start training...")
        for epoch in range(1, int(self.num_train_epochs) + 1):
            for step, batch in enumerate(train_data_loader, start=1):
                with auto_cast(enable=self.fp16,
                               custom_white_list=["softmax", "gelu"]):
                    logits = model(input_ids=batch['input_ids'],
                                   token_type_ids=batch['segment_ids'],
                                   attention_mask=batch['input_mask']
                                   )

                    loss = criterion(logits=logits,
                                     start_positions=batch['start_positions'],
                                     end_positions=batch['end_positions'],
                                     answer_types=batch['answer_types'])

                    loss = loss / self.gradient_accumulation_steps
                    losses += loss.detach()  # losses for logging only

                if self.fp16:
                    scaled = scaler.scale(loss)
                    scaled.backward()
                else:
                    loss.backward()

                if step % self.gradient_accumulation_steps == 0:
                    global_step += 1

                    if self.fp16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.clear_grad()
                    lr_scheduler.step()

                    if global_step % self.logging_steps == 0 or \
                            global_step == num_train_steps:

                        if dist.get_world_size() > 1:
                            dist.barrier()
                        local_loss = losses / self.logging_steps
                        dev_loss_tensor, acc, diff = self.evaluate(model=model,
                                                                   dev_data_loader=dev_data_loader,
                                                                   criterion=criterion)
                        if dist.get_world_size() > 1:
                            dist.all_gather(loss_list, local_loss)
                            dist.all_gather(dev_loss_list, dev_loss_tensor)
                            dist.all_gather(acc_list, acc)
                            dist.all_gather(diff_list, diff)

                            if dist.get_rank() == 0:
                                logging_loss = (paddle.stack(loss_list).sum() / len(
                                    loss_list)).item()
                                dev_loss = (paddle.stack(dev_loss_list).sum() / len(
                                    dev_loss_list)).item()
                                logging_acc = (paddle.stack(acc_list).sum() / len(
                                    acc_list)).item()
                                logging_diff = (paddle.stack(diff_list).sum() / len(
                                    diff_list)).item()

                                logger.info(f"Step {global_step}/{num_train_steps} train loss {logging_loss:.4f}"
                                            f" dev loss {dev_loss:.4f} acc {logging_acc:.2f}% diff {logging_diff:.2f}"
                                            f" time {(time.time() - time1) / 60:.2f}min"
                                            )
                            dist.barrier()
                        else:
                            logging_loss = local_loss.item()
                            logger.info(f"Step {global_step}/{num_train_steps} train loss {logging_loss:.4f}"
                                        f" dev loss {dev_loss_tensor.item():.4f} acc {acc.item():.2f}% "
                                        f"diff {diff.item():.2f}"
                                        f" time {(time.time() - time1) / 60:.2f}min"
                                        )
                        losses = paddle.to_tensor([0.0])
                        time1 = time.time()
                        model.train()
                    if global_step % self.checkout_steps == 0:
                        self.save_ckpt(model, name=f"{global_step}_{self.seed}_tydi.pdparams")

        logger.info(f"training done, total steps trained: {global_step}")
        self.save_ckpt(model, name=self.save_weight_file_name)

    @paddle.no_grad()
    def evaluate(self, model, dev_data_loader, criterion):
        """
        This Evaluating step is just for observing the trend of fine-tuning and debugging. It does not
        provide any information about the final Tydi task testing score.
        Returns:
            losses (Tensor): losses on development set.
            total_acc (Tensor): Mean Accuracy on Answer type prediction task.
            total_diff (Tensor): difference between target answer span and predicted answer span,
                           w.r.t. the start and end index.
        """
        model.eval()
        total_acc, total_diff = paddle.to_tensor([0.0]), paddle.to_tensor([0.0])
        losses = paddle.to_tensor([0.0])
        acc_count, diff_count = 0, 0
        for step, batch in enumerate(dev_data_loader, start=1):
            start_logits, end_logits, type_logits = model(input_ids=batch['input_ids'],
                                                          token_type_ids=batch['segment_ids'],
                                                          attention_mask=batch['input_mask']
                                                          )
            start_positions = batch['start_positions']
            end_positions = batch['end_positions']
            answer_types = batch['answer_types']
            loss = criterion(logits=(start_logits, end_logits, type_logits),
                             start_positions=start_positions,
                             end_positions=end_positions,
                             answer_types=answer_types)
            start_pred = paddle.argmax(start_logits, axis=-1)
            end_pred = paddle.argmax(end_logits, axis=-1)
            type_pred = paddle.argmax(type_logits, axis=-1)

            total_acc += paddle.sum((type_pred == answer_types)[answer_types != 0])
            acc_count += paddle.sum(answer_types != 0).item()

            start_mask = start_positions != 0
            total_diff += paddle.sum(paddle.abs(start_pred - start_positions)[start_mask])
            total_diff += paddle.sum(paddle.abs(end_pred - end_positions)[start_mask])
            diff_count += paddle.sum(start_mask).item()

            losses += loss.detach()

        losses /= len(dev_data_loader) + 1
        total_acc /= acc_count + 1
        total_diff /= diff_count * 2 + 1
        return losses, total_acc * 100, total_diff

    @paddle.no_grad()
    def predict(self):
        """Run prediction."""
        assert self.precomputed_predict_file is not None, "precomputed predict file is missed"
        # Accumulates all of the prediction results to be written to the output.
        results_path = glob.glob(os.path.join(self.output_dir, "results_gpu_*.pickle"))
        if len(results_path) > 0:
            logger.warning(f"please remove results_gpu_*.pickle file in {self.output_dir} first, "
                           f"if you want to start a new prediction new prediction start")
            return

        if dist.get_world_size() > 1:
            dist.init_parallel_env()

        with h5py.File(self.precomputed_predict_file, 'r') as fp:
            num_samples = fp[feature_group_name].len()
        sample_ids = list(range(num_samples))

        test_data_loader = get_dataloader(sample_ids=sample_ids,
                                          h5df_path=self.precomputed_predict_file,
                                          batch_size=self.predict_batch_size,
                                          is_train=False
                                          )
        logger.info(f">>> Number of prediction samples: {num_samples}")

        model = self.load_model()
        model.eval()

        all_logits = []
        time1 = time.time()
        for step, batch in enumerate(test_data_loader):
            start_logits, end_logits, answer_type_logits = model(input_ids=batch['input_ids'],
                                                                 token_type_ids=batch['segment_ids'],
                                                                 attention_mask=batch['input_mask'])
            if step % self.logging_steps == 0 and dist.get_rank() == 0:
                logger.info(f"step {step}/{len(test_data_loader)} "
                            f"time {(time.time() - time1) / 60:.2f}min")
                time1 = time.time()
            all_logits.append(
                [batch['unique_ids'].detach().numpy(),
                 start_logits.detach().numpy(),
                 end_logits.detach().numpy(),
                 answer_type_logits.detach().numpy()]
            )

        if dist.get_world_size() > 1:
            with open(os.path.join(self.output_dir, f"results_gpu_{dist.get_rank()}.pickle"), "wb") as fp:
                pickle.dump(all_logits, fp)
            dist.barrier()
        else:
            with open(os.path.join(self.output_dir, "results_gpu_0.pickle"), "wb") as fp:
                pickle.dump(all_logits, fp)

    def gen_tydi_result(self):
        """
        Almost copied from tydi baseline `google-research/language/language/canine/tydiqa/run_tydi_lib.py`.
        Generate the prediction result that matches Tydi task requirement.
        """
        if dist.get_world_size() > 1:
            if dist.get_rank() != 0:
                # keep only one progress for loading testing data
                return
        logger.info(">>> start generating TyqiQA task evaluation file")

        results_path = glob.glob(os.path.join(self.output_dir, "results_gpu_*.pickle"))
        all_logits = []
        for result_path in results_path:
            with open(result_path, "rb") as fp:
                all_logits.extend(pickle.load(fp))

        all_results = []
        logger.info(f">>> Loaded predicted logits from {results_path}")
        logger.info(">>> start processing predicted logits")

        num_logits = len(all_logits)
        for _ in range(num_logits):
            unique_ids, start_logits, end_logits, answer_type_logits = all_logits.pop()
            for batch_id in range(unique_ids.shape[0]):
                unique_id = int(unique_ids[batch_id].astype('int32'))
                start_logit = start_logits[batch_id]
                end_logit = end_logits[batch_id]
                answer_type_logit = answer_type_logits[batch_id]
                all_results.append(
                    RawResult(
                        unique_id=unique_id,
                        start_logits=start_logit,
                        end_logits=end_logit,
                        answer_type_logits=answer_type_logit))

            # Allow `None` or `0` to disable this behavior.
            if self.max_to_predict and len(all_results) == self.max_to_predict:
                logger.info(
                    "WARNING: Stopping predictions early since "
                    "max_to_predict == %d", self.max_to_predict)
                break

        candidates_dict = self.read_candidates(self.predict_file)

        predict_features = []
        logger.info("loading precomputed evaluation meta data....")
        with h5py.File(self.precomputed_predict_file, "r") as fp:
            meta_dataset = fp[meta_group_name]
            offset_dataset = fp[offset_group_name]
            for idx in range(len(meta_dataset)):
                predict_features.append({
                    "wp_start_offset": offset_dataset[idx][0].astype('int32'),
                    "wp_end_offset": offset_dataset[idx][1].astype('int32'),
                    "language_id": [meta_dataset[idx][2].astype('int8')],
                    "example_index": meta_dataset[idx][1],
                    "unique_ids": meta_dataset[idx][0]
                })

        logger.info("  Num candidate examples loaded (includes all shards): %d",
                    len(candidates_dict))
        logger.info("  Num candidate features loaded: %d", len(predict_features))
        logger.info("  Num prediction result features: %d", len(all_results))

        tydi_pred_dict = postproc.compute_pred_dict(
            candidates_dict,
            predict_features,
            [r._asdict() for r in all_results],
            candidate_beam=self.candidate_beam,
            max_answer_length=self.max_answer_length)

        logging.info("  Num post-processed results: %d", len(tydi_pred_dict))

        full_tydi_pred_dict = {}
        for key, value in tydi_pred_dict.items():
            if key in full_tydi_pred_dict:
                logger.warning("ERROR: '%s' already in full_tydi_pred_dict!", key)
            full_tydi_pred_dict[key] = value

        logger.info("Prediction finished for all shards.")
        logger.info("  Total output predictions: %d", len(full_tydi_pred_dict))

        with open(self.output_prediction_file, "w") as output_file:
            for prediction in full_tydi_pred_dict.values():
                output_file.write((json.dumps(prediction) + "\n"))

    # copied form language/language/canine/tydiqa/run_tydi/read_candidates
    def read_candidates(self, dev_jsonl_file):
        """Read candidates from an input pattern."""
        logger.info("Reading: %s", dev_jsonl_file)
        candidates_dict = {}
        count = 0
        with gzip.GzipFile(dev_jsonl_file, "rb") as input_file:  # pytype: disable=wrong-arg-types
            for line in input_file:
                json_dict = json.loads(line, object_pairs_hook=collections.OrderedDict)
                candidates_dict[json_dict["example_id"]] = json_dict["passage_answer_candidates"]
                count += 1
        assert count == len(candidates_dict), f"load prediction candidates failed"
        return candidates_dict


# This represents the raw predictions coming out of the neural model.
RawResult = collections.namedtuple(
    "RawResult",
    ["unique_id", "start_logits", "end_logits", "answer_type_logits"])
