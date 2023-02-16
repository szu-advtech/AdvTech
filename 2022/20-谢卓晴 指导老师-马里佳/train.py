#!/usr/bin/env python3
# coding=utf-8

import argparse
import os
import datetime
from tqdm import tqdm

import torch
import torch.utils.data
from transformers import AutoConfig

from model.model import Model
from data.dataset import Dataset
from utility.initialize import initialize
from utility.log import Log
from utility.schedule.multi_scheduler import multi_scheduler_wrapper
from utility.autoclip import AutoClip
from data.batch import Batch
from config.params import Params
from utility.predict import predict


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="path to config file")
    parser.add_argument("--output_directory", type=str, default="../outputs")
    parser.add_argument("--data_directory", type=str, default="../data")
    parser.add_argument("--dist_backend", default="nccl", type=str)
    parser.add_argument("--dist_url", default="localhost", type=str)
    parser.add_argument("--name", default="test", type=str, help="name of this run.")
    parser.add_argument("--save_checkpoints", dest="save_checkpoints", action="store_true", default=False)
    parser.add_argument("--seed", dest="seed", type=int, default=17181920)
    parser.add_argument("--log_wandb", dest="log_wandb", action="store_true", default=False)
    parser.add_argument("--validate_each", type=int, default=10, help="Validate every ${N}th epoch.")
    parser.add_argument("--wandb_log_mode", type=str, default=None, help="How to log the model weights, supported values: {'all', 'gradients', 'parameters', None}")
    parser.add_argument("--workers", type=int, default=1, help="number of CPU workers per GPU.")
    args = parser.parse_args()

    params = Params()
    params.load_state_dict(vars(args))
    params.load(args)

    encoder_config = AutoConfig.from_pretrained(params.encoder)
    params.hidden_size = encoder_config.hidden_size
    params.n_encoder_layers = encoder_config.num_hidden_layers

    return params


def main(directory, args):
    initialize(args, init_wandb=args.log_wandb)

    dataset = Dataset(args)

    model = Model(dataset, args)
    optimizer = torch.optim.AdamW(model.get_params_for_optimizer(args), betas=(0.9, args.beta_2))
    scheduler = multi_scheduler_wrapper(optimizer, args, len(dataset.train))
    autoclip = AutoClip(model.parameters())

    print(f"\nCONFIG:\n{args.state_dict()}")
    print(f"\n\nMODEL: {model}\n", flush=True)
    log = Log(dataset, model, optimizer, args, directory, log_each=10, log_wandb=args.log_wandb)

    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(args.epochs):

        #
        # TRAINING
        #

        model.train()
        log.train(len_dataset=dataset.train_size)

        model.zero_grad()

        for i, batch in tqdm(enumerate(dataset.train)):
            batch = Batch.to(batch, device)
            total_loss, stats = model(batch)
            total_loss.backward()

            if (i + 1) % args.accumulation_steps == 0:
                grad_norm = autoclip()

                scheduler(epoch)
                optimizer.step()
                model.zero_grad()

                with torch.no_grad():
                    batch_size = batch["every_input"][0].size(0) * args.accumulation_steps
                    log(batch_size, stats, grad_norm=grad_norm, learning_rates=scheduler.lr())

        if epoch < args.epochs - 5 and epoch % args.validate_each != (args.validate_each - 1):
            continue

        #
        # VALIDATION CROSS-ENTROPIES
        #
        model.eval()
        log.eval(len_dataset=dataset.val_size)

        with torch.no_grad():
            for batch in tqdm(dataset.val):
                _, stats = model(Batch.to(batch, device))

                batch_size = batch["every_input"][0].size(0)
                log(batch_size, stats)

        log.flush()

        #
        # VALIDATION MRP-SCORES
        #

        predict(model, dataset.train, args.training_data, args.raw_training_data, args, log, directory, device, mode="train", epoch=epoch)
        predict(model, dataset.val, args.validation_data, args.raw_validation_data, args, log, directory, device, mode="validation", epoch=epoch)

    #
    # TEST PREDICTION
    #
    os.makedirs(f"{directory}/test_predictions", exist_ok=True)
    predict(model, dataset.test, args.test_data, None, args, None, f"{directory}/test_predictions", device, mode="test")

    torch.save({
        "model": model.state_dict(),
        "vocabulary": dataset.state_dict(),
        "params": args.state_dict()
    }, f"{directory}/checkpoint.bin")


if __name__ == "__main__":
    args = parse_arguments()

    timestamp = f"{datetime.datetime.today():%m-%d-%y_%H-%M-%S}"
    directory = f"{args.output_directory}/{args.framework}_{args.language}_{timestamp}"
    os.makedirs(directory, exist_ok=True)

    main(directory, args)
