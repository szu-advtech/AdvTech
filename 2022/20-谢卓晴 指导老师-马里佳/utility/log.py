#!/usr/bin/env python3
# coding=utf-8

from utility.loading_bar import LoadingBar
import time
import torch


class Log:
    def __init__(self, dataset, model, optimizer, args, directory, log_each: int, initial_epoch=-1, log_wandb=True):
        self.dataset = dataset
        self.model = model
        self.args = args
        self.optimizer = optimizer

        self.loading_bar = LoadingBar(length=27)
        self.best_f1_score = 0.0
        self.log_each = log_each
        self.epoch = initial_epoch
        self.log_wandb = log_wandb
        if self.log_wandb:
            globals()["wandb"] = __import__("wandb")  # ugly way to not require wandb if not needed

        self.directory = directory
        self.evaluation_results = f"{directory}/results_{{0}}_{{1}}.json"
        self.full_evaluation_results = f"{directory}/full_results_{{0}}_{{1}}.json"
        self.best_full_evaluation_results = f"{directory}/best_full_results_{{0}}_{{1}}.json"
        self.result_history = {epoch: {} for epoch in range(args.epochs)}

        self.best_checkpoint_filename = f"{self.directory}/best_checkpoint.h5"
        self.last_checkpoint_filename = f"{self.directory}/last_checkpoint.h5"

        self.step = 0
        self.total_batch_size = 0
        self.flushed = True

    def train(self, len_dataset: int) -> None:
        self.flush()

        self.epoch += 1
        if self.epoch == 0:
            self._print_header()

        self.is_train = True
        self._reset(len_dataset)

    def eval(self, len_dataset: int) -> None:
        self.flush()
        self.is_train = False
        self._reset(len_dataset)

    def __call__(self, batch_size, losses, grad_norm: float = None, learning_rates: float = None,) -> None:
        if self.is_train:
            self._train_step(batch_size, losses, grad_norm, learning_rates)
        else:
            self._eval_step(batch_size, losses)

        self.flushed = False

    def flush(self) -> None:
        if self.flushed:
            return
        self.flushed = True

        if self.is_train:
            print(f"\r┃{self.epoch:12d}  ┃{self._time():>12}  │", end="", flush=True)
        else:
            if self.losses is not None and self.log_wandb:
                dictionary = {f"validation/{key}": value / self.step for key, value in self.losses.items()}
                dictionary["epoch"] = self.epoch
                wandb.log(dictionary)

            self.losses = None
            # self._save_model(save_as_best=False, performance=None)

    def log_evaluation(self, scores, mode, epoch):
        f1_score = scores["sentiment_tuple/f1"]
        if self.log_wandb:
            scores = {f"{mode}/{k}": v for k, v in scores.items()}
            wandb.log({
                "epoch": epoch,
                **scores
            })

        if mode == "validation" and f1_score > self.best_f1_score:
            if self.log_wandb:
                wandb.run.summary["best sentiment tuple f1 score"] = f1_score
                self.best_f1_score = f1_score
                self._save_model(save_as_best=True, f1_score=f1_score)

    def _save_model(self, save_as_best: bool, f1_score: float):
        if not self.args.save_checkpoints:
            return

        state = {
            "epoch": self.epoch,
            "dataset": self.dataset.state_dict(),
            "f1_score": f1_score,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "args": self.args.state_dict(),
        }

        filename = self.best_checkpoint_filename if save_as_best else self.last_checkpoint_filename

        torch.save(state, filename)
        if self.log_wandb:
            wandb.save(filename)

    def _train_step(self, batch_size, losses, grad_norm: float, learning_rates) -> None:
        self.total_batch_size += batch_size
        self.step += 1

        if self.losses is None:
            self.losses = losses
        else:
            for key, values in losses.items():
                if key not in self.losses:
                    self.losses[key] = losses[key]
                    continue
                self.losses[key] += losses[key]

        if self.step % self.log_each == 0:
            progress = self.total_batch_size / self.len_dataset
            print(f"\r┃{self.epoch:12d}  │{self._time():>12}  {self.loading_bar(progress)}", end="", flush=True)

            if self.log_wandb:
                dictionary = {f"train/{key}" if not key.startswith("weight/") else key: value / self.log_each for key, value in self.losses.items()}
                dictionary["epoch"] = self.epoch
                dictionary["learning_rate/encoder"] = learning_rates[0]
                dictionary["learning_rate/decoder"] = learning_rates[-2]
                dictionary["learning_rate/grad_norm"] = learning_rates[-1]
                dictionary["gradient norm"] = grad_norm

                wandb.log(dictionary)

            self.losses = None

    def _eval_step(self, batch_size, losses) -> None:
        self.step += 1

        if self.losses is None:
            self.losses = losses
        else:
            for key, values in losses.items():
                if key not in self.losses:
                    self.losses[key] = losses[key]
                    continue
                self.losses[key] += losses[key]

    def _reset(self, len_dataset: int) -> None:
        self.start_time = time.time()
        self.step = 0
        self.total_batch_size = 0
        self.len_dataset = len_dataset
        self.losses = None

    def _time(self) -> str:
        time_seconds = int(time.time() - self.start_time)
        return f"{time_seconds // 60:02d}:{time_seconds % 60:02d} min"

    def _print_header(self) -> None:
        print(f"┏━━━━━━━━━━━━━━┳━━━╸S╺╸E╺╸M╺╸A╺╸N╺╸T╺╸I╺╸S╺╸K╺━━━━━━━━━━━━━━┓")
        print(f"┃              ┃              ╷                             ┃")
        print(f"┃       epoch  ┃     elapsed  │               progress bar  ┃")
        print(f"┠──────────────╂──────────────┼─────────────────────────────┨")
