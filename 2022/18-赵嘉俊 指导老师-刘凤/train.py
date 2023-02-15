#! /usr/bin/env python

""" Top-level script for training models """

import numpy as np
import logging
import sys
import torch
import configargparse
import copy
from torch import optim

sys.path.append("../")

from training import losses, callbacks
from training import ForwardTrainer, ConditionalForwardTrainer, SCANDALForwardTrainer, AdversarialTrainer, ConditionalAdversarialTrainer, AlternatingTrainer

from datasets import load_simulator, SIMULATORS
from utils import create_filename, create_modelname, nat_to_bit_per_dim
from architectures import create_model
from architectures.create_model import ALGORITHMS
from manifold_flow.transforms.normalization import ActNorm

logger = logging.getLogger(__name__)


def parse_args():
    """ Parses command line arguments for the training """

    parser = configargparse.ArgumentParser()

    # What what what
    parser.add_argument("--modelname", type=str, default=None, help="Model name. Algorithm, latent dimension, dataset, and run are prefixed automatically.")
    parser.add_argument(
        "--algorithm", type=str, default="flow", choices=ALGORITHMS, help="Model: flow (AF), mf (FOM, M-flow), emf (Me-flow), pie (PIE), gamf (M-flow-OT), pae (PAE)...",
    )
    parser.add_argument("--dataset", type=str, default="spherical_gaussian", choices=SIMULATORS, help="Dataset: spherical_gaussian, power, lhc, lhc40d, lhc2d, and some others")
    parser.add_argument("-i", type=int, default=0, help="Run number")

    # Dataset details
    parser.add_argument("--truelatentdim", type=int, default=2, help="True manifold dimensionality (for datasets where that is variable)")
    parser.add_argument("--datadim", type=int, default=3, help="True data dimensionality (for datasets where that is variable)")
    parser.add_argument("--epsilon", type=float, default=0.01, help="Noise term (for datasets where that is variable)")

    # Model details
    parser.add_argument("--modellatentdim", type=int, default=512, help="Model manifold dimensionality")
    parser.add_argument("--specified", action="store_true", help="Prescribe manifold chart: FOM instead of M-flow")
    parser.add_argument("--outertransform", type=str, default="rq-coupling", help="Scalar base trf. for f: {affine | quadratic | rq}-{coupling | autoregressive}")
    parser.add_argument("--innertransform", type=str, default="rq-coupling", help="Scalar base trf. for h: {affine | quadratic | rq}-{coupling | autoregressive}")
    parser.add_argument("--lineartransform", type=str, default="permutation", help="Scalar linear trf: linear | permutation")
    parser.add_argument("--outerlayers", type=int, default=5, help="Number of transformations in f (not counting linear transformations)")
    parser.add_argument("--innerlayers", type=int, default=5, help="Number of transformations in h (not counting linear transformations)")
    parser.add_argument("--conditionalouter", action="store_true", help="If dataset is conditional, use this to make f conditional (otherwise only h is conditional)")
    parser.add_argument("--dropout", type=float, default=0.0, help="Use dropout")
    parser.add_argument("--pieepsilon", type=float, default=0.01, help="PIE epsilon term")
    parser.add_argument("--pieclip", type=float, default=None, help="Clip v in p(v), in multiples of epsilon")
    parser.add_argument("--encoderblocks", type=int, default=5, help="Number of blocks in Me-flow / PAE encoder")
    parser.add_argument("--encoderhidden", type=int, default=100, help="Number of hidden units in Me-flow / PAE encoder")
    parser.add_argument("--splinerange", default=3.0, type=float, help="Spline boundaries")
    parser.add_argument("--splinebins", default=8, type=int, help="Number of spline bins")
    parser.add_argument("--levels", type=int, default=3, help="Number of levels in multi-scale architectures for image data (for outer transformation f)")
    parser.add_argument("--actnorm", action="store_true", help="Use actnorm in convolutional architecture")
    parser.add_argument("--batchnorm", action="store_true", help="Use batchnorm in ResNets")
    parser.add_argument("--linlayers", type=int, default=2, help="Number of linear layers before the projection for M-flow and PIE on image data")
    parser.add_argument("--linchannelfactor", type=int, default=2, help="Determines number of channels in linear trfs before the projection for M-flow and PIE on image data")
    parser.add_argument("--intermediatensf", action="store_true", help="Use NSF rather than linear layers before projecting (for M-flows and PIE on image data)")
    parser.add_argument("--decoderblocks", type=int, default=5, help="Number of blocks in PAE encoder")
    parser.add_argument("--decoderhidden", type=int, default=100, help="Number of hidden units in PAE encoder")

    # Training
    parser.add_argument("--alternate", action="store_true", help="Use alternating M/D training algorithm")
    parser.add_argument("--sequential", action="store_true", help="Use sequential M/D training algorithm")
    parser.add_argument("--load", type=str, default=None, help="Model name to load rather than training from scratch, run is affixed automatically")
    parser.add_argument("--startepoch", type=int, default=0, help="Sets the first trained epoch for resuming partial training")
    parser.add_argument("--samplesize", type=int, default=None, help="If not None, number of samples used for training")
    parser.add_argument("--epochs", type=int, default=500, help="Maximum number of epochs")
    parser.add_argument("--subsets", type=int, default=1, help="Number of subsets per epoch in an alternating training")
    parser.add_argument("--batchsize", type=int, default=25, help="Batch size for everything except OT training")
    parser.add_argument("--genbatchsize", type=int, default=1000, help="Batch size for OT training")
    parser.add_argument("--lr", type=float, default=3.0e-4, help="Initial learning rate")
    parser.add_argument("--msefactor", type=float, default=1000.0, help="Reco error multiplier in loss")
    parser.add_argument("--addnllfactor", type=float, default=0.1, help="Negative log likelihood multiplier in loss for M-flow-S training")
    parser.add_argument("--nllfactor", type=float, default=1.0, help="Negative log likelihood multiplier in loss (except for M-flow-S training)")
    parser.add_argument("--sinkhornfactor", type=float, default=10.0, help="Sinkhorn divergence multiplier in loss")
    parser.add_argument("--weightdecay", type=float, default=1.0e-5, help="Weight decay")
    parser.add_argument("--clip", type=float, default=1.0, help="Gradient norm clipping parameter")
    parser.add_argument("--nopretraining", action="store_true", help="Skip pretraining in M-flow-S training")
    parser.add_argument("--noposttraining", action="store_true", help="Skip posttraining in M-flow-S training")
    parser.add_argument("--validationsplit", type=float, default=0.25, help="Fraction of train data used for early stopping")
    parser.add_argument("--scandal", type=float, default=None, help="Activates SCANDAL training and sets prefactor of score MSE in loss")
    parser.add_argument("--l1", action="store_true", help="Use smooth L1 loss rather than L2 (MSE) for reco error")
    parser.add_argument("--uvl2reg", type=float, default=None, help="Add L2 regularization term on the latent variables after the outer flow (M-flow-M/D only)")
    parser.add_argument("--seed", type=int, default=1357, help="Random seed (--i is always added to it)")
    parser.add_argument("--resume", type=int, default=None, help="Resume training at a given epoch (overwrites --load and --startepoch)")

    # Other settings
    parser.add_argument("-c", is_config_file=True, type=str, help="Config file path")
    parser.add_argument("--dir", type=str, default="experiments", help="Base directory of repo")
    parser.add_argument("--debug", action="store_true", help="Debug mode (more log output, additional callbacks)")

    args = parser.parse_args()
    return args


def make_training_kwargs(args, dataset):
    kwargs = {
        "dataset": dataset,
        "batch_size": args.batchsize,
        "initial_lr": args.lr,
        "scheduler": optim.lr_scheduler.CosineAnnealingLR,
        "clip_gradient": args.clip,
        "validation_split": args.validationsplit,
        "seed": args.seed + args.i,
    }
    if args.weightdecay is not None:
        kwargs["optimizer_kwargs"] = {"weight_decay": float(args.weightdecay)}
    scandal_loss = [losses.score_mse] if args.scandal is not None else []
    scandal_label = ["score MSE"] if args.scandal is not None else []
    scandal_weight = [args.scandal] if args.scandal is not None else []

    return kwargs, scandal_loss, scandal_label, scandal_weight


def train_manifold_flow(args, dataset, model, simulator):
    """ M-flow-S training """

    assert not args.specified

    trainer = ForwardTrainer(model) if simulator.parameter_dim() is None else ConditionalForwardTrainer(model) if args.scandal is None else SCANDALForwardTrainer(model)
    common_kwargs, scandal_loss, scandal_label, scandal_weight = make_training_kwargs(args, dataset)

    logger.info("Starting training MF, phase 1: pretraining on reconstruction error")
    learning_curves = trainer.train(
        loss_functions=[losses.mse],
        loss_labels=["MSE"],
        loss_weights=[args.msefactor],
        epochs=args.epochs // 3,
        callbacks=[callbacks.save_model_after_every_epoch(create_filename("checkpoint", "A", args))],
        forward_kwargs={"mode": "projection"},
        initial_epoch=args.startepoch,
        **common_kwargs,
    )
    learning_curves = np.vstack(learning_curves).T

    logger.info("Starting training MF, phase 2: mixed training")
    learning_curves_ = trainer.train(
        loss_functions=[losses.mse, losses.nll] + scandal_loss,
        loss_labels=["MSE", "NLL"] + scandal_label,
        loss_weights=[args.msefactor, args.addnllfactor * nat_to_bit_per_dim(args.modellatentdim)] + scandal_weight,
        epochs=args.epochs - 2 * (args.epochs // 3),
        parameters=list(model.parameters()),
        callbacks=[callbacks.save_model_after_every_epoch(create_filename("checkpoint", "B", args))],
        forward_kwargs={"mode": "mf"},
        initial_epoch=args.startepoch - (args.epochs // 3),
        **common_kwargs,
    )
    learning_curves_ = np.vstack(learning_curves_).T
    learning_curves = learning_curves_ if learning_curves is None else np.vstack((learning_curves, learning_curves_))

    logger.info("Starting training MF, phase 3: training only inner flow on NLL")
    learning_curves_ = trainer.train(
        loss_functions=[losses.mse, losses.nll] + scandal_loss,
        loss_labels=["MSE", "NLL"] + scandal_label,
        loss_weights=[0.0, args.nllfactor * nat_to_bit_per_dim(args.modellatentdim)] + scandal_weight,
        epochs=args.epochs // 3,
        parameters=list(model.inner_transform.parameters()),
        callbacks=[callbacks.save_model_after_every_epoch(create_filename("checkpoint", "C", args))],
        forward_kwargs={"mode": "mf-fixed-manifold"},
        initial_epoch=args.startepoch - (args.epochs - (args.epochs // 3)),
        **common_kwargs,
    )
    learning_curves_ = np.vstack(learning_curves_).T
    learning_curves = np.vstack((learning_curves, np.vstack(learning_curves_).T))

    return learning_curves


def train_specified_manifold_flow(args, dataset, model, simulator):
    """ FOM training """

    trainer = ForwardTrainer(model) if simulator.parameter_dim() is None else ConditionalForwardTrainer(model) if args.scandal is None else SCANDALForwardTrainer(model)
    common_kwargs, scandal_loss, scandal_label, scandal_weight = make_training_kwargs(args, dataset)

    logger.info("Starting training MF with specified manifold on NLL")
    learning_curves = trainer.train(
        loss_functions=[losses.mse, losses.nll] + scandal_loss,
        loss_labels=["MSE", "NLL"] + scandal_label,
        loss_weights=[0.0, args.nllfactor * nat_to_bit_per_dim(args.modellatentdim)] + scandal_weight,
        epochs=args.epochs,
        callbacks=[callbacks.save_model_after_every_epoch(create_filename("checkpoint", None, args))],
        forward_kwargs={"mode": "mf"},
        initial_epoch=args.startepoch,
        **common_kwargs,
    )
    learning_curves = np.vstack(learning_curves).T

    return learning_curves


def train_manifold_flow_alternating(args, dataset, model, simulator):
    """ M-flow-A training """

    assert not args.specified

    trainer1 = ForwardTrainer(model) if simulator.parameter_dim() is None else ConditionalForwardTrainer(model)
    trainer2 = ForwardTrainer(model) if simulator.parameter_dim() is None else ConditionalForwardTrainer(model) if args.scandal is None else SCANDALForwardTrainer(model)
    metatrainer = AlternatingTrainer(model, trainer1, trainer2)

    meta_kwargs = {"dataset": dataset, "initial_lr": args.lr, "scheduler": optim.lr_scheduler.CosineAnnealingLR, "validation_split": args.validationsplit}
    if args.weightdecay is not None:
        meta_kwargs["optimizer_kwargs"] = {"weight_decay": float(args.weightdecay)}
    _, scandal_loss, scandal_label, scandal_weight = make_training_kwargs(args, dataset)

    phase1_kwargs = {"forward_kwargs": {"mode": "projection"}, "clip_gradient": args.clip}
    phase2_kwargs = {"forward_kwargs": {"mode": "mf-fixed-manifold"}, "clip_gradient": args.clip}

    phase1_parameters = list(model.outer_transform.parameters()) + (list(model.encoder.parameters()) if args.algorithm == "emf" else [])
    phase2_parameters = list(model.inner_transform.parameters())

    logger.info("Starting training MF, alternating between reconstruction error and log likelihood")
    learning_curves_ = metatrainer.train(
        loss_functions=[losses.smooth_l1_loss if args.l1 else losses.mse, losses.nll] + scandal_loss,
        loss_function_trainers=[0, 1] + ([1] if args.scandal is not None else []),
        loss_labels=["L1" if args.l1 else "MSE", "NLL"] + scandal_label,
        loss_weights=[args.msefactor, args.nllfactor * nat_to_bit_per_dim(args.modellatentdim)] + scandal_weight,
        epochs=args.epochs // 2,
        subsets=args.subsets,
        batch_sizes=[args.batchsize, args.batchsize],
        parameters=[phase1_parameters, phase2_parameters],
        callbacks=[callbacks.save_model_after_every_epoch(create_filename("checkpoint", None, args))],
        trainer_kwargs=[phase1_kwargs, phase2_kwargs],
        **meta_kwargs,
    )
    learning_curves = np.vstack(learning_curves_).T

    return learning_curves


def train_manifold_flow_sequential(args, dataset, model, simulator):
    """ Sequential M-flow-M/D training """

    assert not args.specified

    if simulator.parameter_dim() is None:
        trainer1 = ForwardTrainer(model)
        trainer2 = ForwardTrainer(model)
    else:
        trainer1 = ConditionalForwardTrainer(model)
        if args.scandal is None:
            trainer2 = ConditionalForwardTrainer(model)
        else:
            trainer2 = SCANDALForwardTrainer(model)

    common_kwargs, scandal_loss, scandal_label, scandal_weight = make_training_kwargs(args, dataset)

    callbacks1 = [callbacks.save_model_after_every_epoch(create_filename("checkpoint", "A", args)), callbacks.print_mf_latent_statistics(), callbacks.print_mf_weight_statistics()]
    callbacks2 = [callbacks.save_model_after_every_epoch(create_filename("checkpoint", "B", args)), callbacks.print_mf_latent_statistics(), callbacks.print_mf_weight_statistics()]
    if simulator.is_image():
        callbacks1.append(
            callbacks.plot_sample_images(
                create_filename("training_plot", "sample_epoch_A", args), context=None if simulator.parameter_dim() is None else torch.zeros(30, simulator.parameter_dim()),
            )
        )
        callbacks2.append(
            callbacks.plot_sample_images(
                create_filename("training_plot", "sample_epoch_B", args), context=None if simulator.parameter_dim() is None else torch.zeros(30, simulator.parameter_dim()),
            )
        )
        callbacks1.append(callbacks.plot_reco_images(create_filename("training_plot", "reco_epoch_A", args)))
        callbacks2.append(callbacks.plot_reco_images(create_filename("training_plot", "reco_epoch_B", args)))

    logger.info("Starting training MF, phase 1: manifold training")
    learning_curves = trainer1.train(
        loss_functions=[losses.smooth_l1_loss if args.l1 else losses.mse] + ([] if args.uvl2reg is None else [losses.hiddenl2reg]),
        loss_labels=["L1" if args.l1 else "MSE"] + ([] if args.uvl2reg is None else ["L2_lat"]),
        loss_weights=[args.msefactor] + ([] if args.uvl2reg is None else [args.uvl2reg]),
        epochs=args.epochs // 2,
        parameters=list(model.outer_transform.parameters()) + list(model.encoder.parameters()) if args.algorithm == "emf" else list(model.outer_transform.parameters()),
        callbacks=callbacks1,
        forward_kwargs={"mode": "projection", "return_hidden": args.uvl2reg is not None},
        initial_epoch=args.startepoch,
        **common_kwargs,
    )
    learning_curves = np.vstack(learning_curves).T

    logger.info("Starting training MF, phase 2: density training")
    learning_curves_ = trainer2.train(
        loss_functions=[losses.nll] + scandal_loss,
        loss_labels=["NLL"] + scandal_label,
        loss_weights=[args.nllfactor * nat_to_bit_per_dim(args.modellatentdim)] + scandal_weight,
        epochs=args.epochs - (args.epochs // 2),
        parameters=list(model.inner_transform.parameters()),
        callbacks=callbacks2,
        forward_kwargs={"mode": "mf-fixed-manifold"},
        initial_epoch=args.startepoch - args.epochs // 2,
        **common_kwargs,
    )
    learning_curves = np.vstack((learning_curves, np.vstack(learning_curves_).T))

    return learning_curves


def train_pae_sequential(args, dataset, model, simulator):
    """ Sequential PAE training """

    assert not args.specified

    if simulator.parameter_dim() is None:
        trainer1 = ForwardTrainer(model)
        trainer2 = ForwardTrainer(model)
    else:
        trainer1 = ConditionalForwardTrainer(model)
        if args.scandal is None:
            trainer2 = ConditionalForwardTrainer(model)
        else:
            trainer2 = SCANDALForwardTrainer(model)

    common_kwargs, scandal_loss, scandal_label, scandal_weight = make_training_kwargs(args, dataset)

    callbacks1 = [callbacks.save_model_after_every_epoch(create_filename("checkpoint", "A", args)), callbacks.print_mf_latent_statistics(), callbacks.print_mf_weight_statistics()]
    callbacks2 = [callbacks.save_model_after_every_epoch(create_filename("checkpoint", "B", args)), callbacks.print_mf_latent_statistics(), callbacks.print_mf_weight_statistics()]
    if simulator.is_image():
        callbacks1.append(
            callbacks.plot_sample_images(
                create_filename("training_plot", "sample_epoch_A", args), context=None if simulator.parameter_dim() is None else torch.zeros(30, simulator.parameter_dim()),
            )
        )
        callbacks2.append(
            callbacks.plot_sample_images(
                create_filename("training_plot", "sample_epoch_B", args), context=None if simulator.parameter_dim() is None else torch.zeros(30, simulator.parameter_dim()),
            )
        )
        callbacks1.append(callbacks.plot_reco_images(create_filename("training_plot", "reco_epoch_A", args)))
        callbacks2.append(callbacks.plot_reco_images(create_filename("training_plot", "reco_epoch_B", args)))

    logger.info("Starting training PAE, phase 1: manifold training")
    learning_curves = trainer1.train(
        loss_functions=[losses.smooth_l1_loss if args.l1 else losses.mse] + ([] if args.uvl2reg is None else [losses.hiddenl2reg]),
        loss_labels=["L1" if args.l1 else "MSE"] + ([] if args.uvl2reg is None else ["L2_lat"]),
        loss_weights=[args.msefactor] + ([] if args.uvl2reg is None else [args.uvl2reg]),
        epochs=args.epochs // 2,
        parameters=list(model.decoder.parameters()) + list(model.encoder.parameters()),
        callbacks=callbacks1,
        forward_kwargs={"return_hidden": args.uvl2reg is not None},
        initial_epoch=args.startepoch,
        **common_kwargs,
    )
    learning_curves = np.vstack(learning_curves).T

    logger.info("Starting training PAE, phase 2: density training")
    learning_curves_ = trainer2.train(
        loss_functions=[losses.nll] + scandal_loss,
        loss_labels=["NLL"] + scandal_label,
        loss_weights=[args.nllfactor * nat_to_bit_per_dim(args.modellatentdim)] + scandal_weight,
        epochs=args.epochs - (args.epochs // 2),
        parameters=list(model.inner_transform.parameters()),
        callbacks=callbacks2,
        forward_kwargs={},
        initial_epoch=args.startepoch - args.epochs // 2,
        **common_kwargs,
    )
    learning_curves = np.vstack((learning_curves, np.vstack(learning_curves_).T))

    return learning_curves


def train_generative_adversarial_manifold_flow(args, dataset, model, simulator):
    """ M-flow-OT training """

    gen_trainer = AdversarialTrainer(model) if simulator.parameter_dim() is None else ConditionalAdversarialTrainer(model)
    common_kwargs, scandal_loss, scandal_label, scandal_weight = make_training_kwargs(args, dataset)
    common_kwargs["batch_size"] = args.genbatchsize

    logger.info("Starting training GAMF: Sinkhorn-GAN")

    callbacks_ = [callbacks.save_model_after_every_epoch(create_filename("checkpoint", None, args))]
    if args.debug:
        callbacks_.append(callbacks.print_mf_weight_statistics())

    learning_curves_ = gen_trainer.train(
        loss_functions=[losses.make_sinkhorn_divergence()],
        loss_labels=["GED"],
        loss_weights=[args.sinkhornfactor],
        epochs=args.epochs,
        callbacks=callbacks_,
        compute_loss_variance=True,
        initial_epoch=args.startepoch,
        **common_kwargs,
    )

    learning_curves = np.vstack(learning_curves_).T
    return learning_curves


def train_generative_adversarial_manifold_flow_alternating(args, dataset, model, simulator):
    """ M-flow-OTA training """

    assert not args.specified

    gen_trainer = AdversarialTrainer(model) if simulator.parameter_dim() is None else ConditionalAdversarialTrainer(model)
    likelihood_trainer = ForwardTrainer(model) if simulator.parameter_dim() is None else ConditionalForwardTrainer(model) if args.scandal is None else SCANDALForwardTrainer(model)
    metatrainer = AlternatingTrainer(model, gen_trainer, likelihood_trainer)

    meta_kwargs = {"dataset": dataset, "initial_lr": args.lr, "scheduler": optim.lr_scheduler.CosineAnnealingLR, "validation_split": args.validationsplit}
    if args.weightdecay is not None:
        meta_kwargs["optimizer_kwargs"] = {"weight_decay": float(args.weightdecay)}
    _, scandal_loss, scandal_label, scandal_weight = make_training_kwargs(args, dataset)

    phase1_kwargs = {"clip_gradient": args.clip}
    phase2_kwargs = {"forward_kwargs": {"mode": "mf-fixed-manifold"}, "clip_gradient": args.clip}

    phase1_parameters = list(model.parameters())
    phase2_parameters = list(model.inner_transform.parameters())

    logger.info("Starting training GAMF, alternating between Sinkhorn divergence and log likelihood")
    learning_curves_ = metatrainer.train(
        loss_functions=[losses.make_sinkhorn_divergence(), losses.nll] + scandal_loss,
        loss_function_trainers=[0, 1] + [1] if args.scandal is not None else [],
        loss_labels=["GED", "NLL"] + scandal_label,
        loss_weights=[args.sinkhornfactor, args.nllfactor * nat_to_bit_per_dim(args.modellatentdim)] + scandal_weight,
        batch_sizes=[args.genbatchsize, args.batchsize],
        epochs=args.epochs // 2,
        parameters=[phase1_parameters, phase2_parameters],
        callbacks=[callbacks.save_model_after_every_epoch(create_filename("checkpoint", None, args))],
        trainer_kwargs=[phase1_kwargs, phase2_kwargs],
        subsets=args.subsets,
        subset_callbacks=[callbacks.print_mf_weight_statistics()] if args.debug else None,
        **meta_kwargs,
    )
    learning_curves = np.vstack(learning_curves_).T

    return learning_curves


def train_flow(args, dataset, model, simulator):
    """ AF training """
    trainer = ForwardTrainer(model) if simulator.parameter_dim() is None else ConditionalForwardTrainer(model) if args.scandal is None else SCANDALForwardTrainer(model)
    common_kwargs, scandal_loss, scandal_label, scandal_weight = make_training_kwargs(args, dataset)
    callbacks_ = [callbacks.save_model_after_every_epoch(create_filename("checkpoint", None, args))]
    if simulator.is_image():
        callbacks_.append(
            callbacks.plot_sample_images(
                create_filename("training_plot", None, args), context=None if simulator.parameter_dim() is None else torch.zeros(30, simulator.parameter_dim())
            )
        )

    logger.info("Starting training standard flow on NLL")
    learning_curves = trainer.train(
        loss_functions=[losses.nll] + scandal_loss,
        loss_labels=["NLL"] + scandal_label,
        loss_weights=[args.nllfactor * nat_to_bit_per_dim(args.datadim)] + scandal_weight,
        epochs=args.epochs,
        callbacks=callbacks_,
        initial_epoch=args.startepoch,
        **common_kwargs,
    )

    learning_curves = np.vstack(learning_curves).T
    return learning_curves


def train_pie(args, dataset, model, simulator):
    """ PIE training """
    trainer = ForwardTrainer(model) if simulator.parameter_dim() is None else ConditionalForwardTrainer(model) if args.scandal is None else SCANDALForwardTrainer(model)
    common_kwargs, scandal_loss, scandal_label, scandal_weight = make_training_kwargs(args, dataset)
    callbacks_ = [callbacks.save_model_after_every_epoch(create_filename("checkpoint", None, args))]
    if simulator.is_image():
        callbacks_.append(
            callbacks.plot_sample_images(
                create_filename("training_plot", None, args), context=None if simulator.parameter_dim() is None else torch.zeros(30, simulator.parameter_dim())
            )
        )
        callbacks_.append(callbacks.plot_reco_images(create_filename("training_plot", "reco_epoch", args)))

    logger.info("Starting training PIE on NLL")
    learning_curves = trainer.train(
        loss_functions=[losses.nll] + scandal_loss,
        loss_labels=["NLL"] + scandal_label,
        loss_weights=[args.nllfactor * nat_to_bit_per_dim(args.datadim)] + scandal_weight,
        epochs=args.epochs,
        callbacks=callbacks_,
        forward_kwargs={"mode": "pie"},
        initial_epoch=args.startepoch,
        **common_kwargs,
    )
    learning_curves = np.vstack(learning_curves).T
    return learning_curves


def train_model(args, dataset, model, simulator):
    """ Starts appropriate training """

    if args.algorithm == "pie":
        learning_curves = train_pie(args, dataset, model, simulator)
    elif args.algorithm == "flow":
        learning_curves = train_flow(args, dataset, model, simulator)
    elif args.algorithm in ["mf", "emf"]:
        if args.alternate:
            learning_curves = train_manifold_flow_alternating(args, dataset, model, simulator)
        elif args.sequential:
            learning_curves = train_manifold_flow_sequential(args, dataset, model, simulator)
        elif args.specified:
            learning_curves = train_specified_manifold_flow(args, dataset, model, simulator)
        else:
            learning_curves = train_manifold_flow(args, dataset, model, simulator)
    elif args.algorithm == "pae":
        assert args.sequential
        learning_curves = train_pae_sequential(args, dataset, model, simulator)
    elif args.algorithm == "gamf":
        if args.alternate:
            learning_curves = train_generative_adversarial_manifold_flow_alternating(args, dataset, model, simulator)
        else:
            learning_curves = train_generative_adversarial_manifold_flow(args, dataset, model, simulator)
    else:
        raise ValueError("Unknown algorithm %s", args.algorithm)

    return learning_curves


def fix_act_norm_issue(model):
    """Fixing initialization state of actnorm layer"""
    if isinstance(model, ActNorm):
        logger.debug("Fixing initialization state of actnorm layer")
        model.initialized = True

    for _, submodel in model._modules.items():
        fix_act_norm_issue(submodel)


if __name__ == "__main__":
    # Logger
    args = parse_args()
    # 设置日志的基础配置
    # %(asctime)s: 打印日志的时间；%(message)s: 打印日志信息；%(levelname)s: 打印日志级别名称#
    logging.basicConfig(format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s", datefmt="%H:%M", level=logging.DEBUG if args.debug else logging.INFO)
    # 默认的级别是level,也就意味着只有级别大于等于level的才会被看到
    # logger.info()确定输出的信息是info级别（level=20）
    # 级别大小：DEBUG<INFO<WARNING<ERROR<CRITICAL#
    logger.info("Hi!")
    logger.debug("Starting train.py with arguments %s", args)

    create_modelname(args)

    if args.resume is not None:
        resume_filename = create_filename("resume", "A_", args)
        args.startepoch = args.resume
        logger.info("Resuming training. Loading file %s and continuing with epoch %s.", resume_filename, args.resume + 1)
    elif args.load is None:
        logger.info("Training model %s with algorithm %s on data set %s", args.modelname, args.algorithm, args.dataset)
    else:
        logger.info("Loading model %s and training it as %s with algorithm %s on data set %s", args.load, args.modelname, args.algorithm, args.dataset)

    # Bug fix related to some num_workers > 1 and CUDA. Bad things happen otherwise!
    #  .set_start_method()用于指定创建child process的方式，可选的值为fork、spawn和forkserver。使用spawn，child process仅会继承parent process的必要resource，file descriptor和handle均不会继承#
    torch.multiprocessing.set_start_method("spawn", force=True)

    # Data
    simulator = load_simulator(args)
    dataset = simulator.load_dataset(train=True, dataset_dir=create_filename("dataset", None, args), limit_samplesize=args.samplesize, joint_score=args.scandal is not None)

    # Model
    model = create_model(args, simulator)

    # Maybe load pretrained model
    if args.resume is not None:
        model.load_state_dict(torch.load(resume_filename, map_location=torch.device("cpu")))
        fix_act_norm_issue(model)
    elif args.load is not None:
        args_ = copy.deepcopy(args)
        args_.modelname = args.load
        if args_.i > 0:
            args_.modelname += "_run{}".format(args_.i)
        logger.info("Loading model %s", args_.modelname)
        model.load_state_dict(torch.load(create_filename("model", None, args_), map_location=torch.device("cpu")))
        fix_act_norm_issue(model)

    # Train and save
    learning_curves = train_model(args, dataset, model, simulator)

    # Save
    logger.info("Saving model")
    # 需要对create_filename方法进行修改#
    torch.save(model.state_dict(), create_filename("model", None, args))
    np.save(create_filename("learning_curve", None, args), learning_curves)

    logger.info("All done! Have a nice day!")
