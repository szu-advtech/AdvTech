from torch import nn
from torch.nn import functional as F
import logging
import torch
import numpy as np

from manifold_flow import nn as nn_, transforms
from manifold_flow.utils import various
from .vector_transforms import create_vector_transform

logger = logging.getLogger(__name__)


class PreprocessingEncoder(nn.Module):
    def __init__(self, encoder, preprocessor):
        super().__init__()
        self.encoder = encoder
        self.preprocessor = preprocessor

    def forward(self, inputs, context=None):
        temp = self.preprocessor(inputs)[0]
        outputs = self.encoder(temp, context=context)
        return outputs


def create_image_encoder(
    c, h, w, latent_dim, context_features=None, preprocessing="glow", alpha=0.05, num_bits=8,
):
    assert context_features is None
    preprocessing_transform = _create_preprocessing(alpha, c, h, num_bits, preprocessing, w)
    encoder = nn_.ModifiedConvEncoder(h, w, channels_in=c, channels_multiplier=1, levels=4, out_features=latent_dim, activation=F.relu,)
    encoder = PreprocessingEncoder(encoder, preprocessing_transform)

    return encoder


def _create_image_transform_step(
    num_channels,
    hidden_channels=96,
    context_channels=None,
    actnorm=True,
    coupling_layer_type="rational_quadratic_spline",
    num_res_blocks=3,
    resnet_batchnorm=True,
    dropout_prob=0.0,
    num_bins=8,
    tail_bound=3.0,
):
    def create_convnet(in_channels, out_channels):
        net = nn_.ConvResidualNet(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            context_channels=context_channels,
            num_blocks=num_res_blocks,
            use_batch_norm=resnet_batchnorm,
            dropout_probability=dropout_prob,
        )
        return net

    mask = various.create_mid_split_binary_mask(num_channels)

    if coupling_layer_type == "cubic_spline":
        coupling_layer = transforms.PiecewiseCubicCouplingTransform(
            mask=mask,
            transform_net_create_fn=create_convnet,
            tails="linear",
            tail_bound=tail_bound,
            num_bins=num_bins,
            apply_unconditional_transform=False,
            min_bin_width=0.001,
            min_bin_height=0.001,
        )
    elif coupling_layer_type == "quadratic_spline":
        coupling_layer = transforms.PiecewiseQuadraticCouplingTransform(
            mask=mask,
            transform_net_create_fn=create_convnet,
            tails="linear",
            tail_bound=tail_bound,
            num_bins=num_bins,
            apply_unconditional_transform=False,
            min_bin_width=0.001,
            min_bin_height=0.001,
        )
    elif coupling_layer_type == "rational_quadratic_spline":
        coupling_layer = transforms.PiecewiseRationalQuadraticCouplingTransform(
            mask=mask,
            transform_net_create_fn=create_convnet,
            tails="linear",
            tail_bound=tail_bound,
            num_bins=num_bins,
            apply_unconditional_transform=False,
            min_bin_width=0.001,
            min_bin_height=0.001,
            min_derivative=0.001,
        )
    elif coupling_layer_type == "affine":
        coupling_layer = transforms.AffineCouplingTransform(mask=mask, transform_net_create_fn=create_convnet)
    elif coupling_layer_type == "additive":
        coupling_layer = transforms.AdditiveCouplingTransform(mask=mask, transform_net_create_fn=create_convnet)
    else:
        raise RuntimeError("Unknown coupling_layer_type")

    step_transforms = []

    if actnorm:
        step_transforms.append(transforms.ActNorm(num_channels))

    step_transforms.extend([transforms.OneByOneConvolution(num_channels), coupling_layer])

    logger.debug("  Flow based on %s", coupling_layer_type)

    return transforms.CompositeTransform(step_transforms)


def create_image_transform(
    c,
    h,
    w,
    levels=3,
    hidden_channels=96,
    steps_per_level=7,
    alpha=0.05,
    num_bits=8,
    preprocessing="glow",
    multi_scale=True,
    dropout_prob=0.0,
    num_res_blocks=3,
    coupling_layer_type="rational_quadratic_spline",
    use_batchnorm=True,
    use_actnorm=True,
    postprocessing="permutation",
    postprocessing_layers=2,
    postprocessing_channel_factor=2,
    context_features=None,
    num_bins=8,
    tail_bound=3.0,
):
    assert h == w
    res = h
    dim = c * h * w

    if not isinstance(hidden_channels, list):
        hidden_channels = [hidden_channels] * levels

    preprocess_transform = _create_preprocessing(alpha, c, h, num_bits, preprocessing, w)

    # Main part
    if multi_scale:
        logger.debug("Input: c, h, w = %s, %s, %s", c, h, w)
        mct = transforms.MultiscaleCompositeTransform(num_transforms=levels)
        for level, level_hidden_channels in zip(range(levels), hidden_channels):
            logger.debug("Level %s", level)
            # transforms.SqueezeTransform()将h,w都缩小2倍#
            squeeze_transform = transforms.SqueezeTransform()
            c, h, w = squeeze_transform.get_output_shape(c, h, w)
            logger.debug("  c, h, w = %s, %s, %s", c, h, w)

            logger.debug("  SqueezeTransform()")
            transform_level = transforms.CompositeTransform(
                [squeeze_transform]
                + [
                    _create_image_transform_step(
                        c,
                        level_hidden_channels,
                        actnorm=use_actnorm,
                        coupling_layer_type=coupling_layer_type,
                        num_bins=num_bins,
                        tail_bound=tail_bound,
                        num_res_blocks=num_res_blocks,
                        resnet_batchnorm=use_batchnorm,
                        dropout_prob=dropout_prob,
                        context_channels=context_features,
                    )
                    for _ in range(steps_per_level)
                ]
                + [transforms.OneByOneConvolution(c)]  # End each level with a linear transformation.
            )
            logger.debug("  OneByOneConvolution(%s)", c)

            new_shape = mct.add_transform(transform_level, (c, h, w))
            if new_shape:  # If not last layer
                c, h, w = new_shape
                logger.debug("  new_shape = %s, %s, %s", c, h, w)
    else:
        all_transforms = []

        for level, level_hidden_channels in zip(range(levels), hidden_channels):
            squeeze_transform = transforms.SqueezeTransform()
            c, h, w = squeeze_transform.get_output_shape(c, h, w)

            transform_level = transforms.CompositeTransform(
                [squeeze_transform]
                + [
                    _create_image_transform_step(
                        c,
                        level_hidden_channels,
                        actnorm=use_actnorm,
                        coupling_layer_type=coupling_layer_type,
                        num_res_blocks=num_res_blocks,
                        resnet_batchnorm=use_batchnorm,
                        dropout_prob=dropout_prob,
                        context_channels=context_features,
                    )
                    for _ in range(steps_per_level)
                ]
                + [transforms.OneByOneConvolution(c)]  # End each level with a linear transformation.
            )
            all_transforms.append(transform_level)

        all_transforms.append(transforms.ReshapeTransform(input_shape=(c, h, w), output_shape=(c * h * w,)))
        mct = transforms.CompositeTransform(all_transforms)

    # Final transformation
    final_transform = _create_postprocessing(
        dim, multi_scale, postprocessing, postprocessing_channel_factor, postprocessing_layers, res, context_features, num_bins=num_bins, tail_bound=tail_bound
    )

    return transforms.CompositeTransform([preprocess_transform, mct, final_transform])


def _create_postprocessing(dim, multi_scale, postprocessing, postprocessing_channel_factor, postprocessing_layers, res, context_features, tail_bound, num_bins):
    # TODO: take context_features into account here

    if postprocessing == "linear":
        final_transform = transforms.LULinear(dim, identity_init=True)
        logger.debug("LULinear(%s)", dim)

    elif postprocessing == "partial_linear":
        if multi_scale:
            mask = various.create_mlt_channel_mask(dim, channels_per_level=postprocessing_channel_factor * np.array([1, 2, 4, 8], dtype=np.int), resolution=res)
            partial_dim = torch.sum(mask.to(dtype=torch.int)).item()
        else:
            partial_dim = postprocessing_channel_factor * 1024
            mask = various.create_split_binary_mask(dim, partial_dim)

        partial_transform = transforms.LULinear(partial_dim, identity_init=True)
        final_transform = transforms.PartialTransform(mask, partial_transform)
        logger.debug("PartialTransform (LULinear) (%s)", partial_dim)

    elif postprocessing == "partial_mlp":
        if multi_scale:
            mask = various.create_mlt_channel_mask(dim, channels_per_level=postprocessing_channel_factor * np.array([1, 2, 4, 8], dtype=np.int), resolution=res)
            partial_dim = torch.sum(mask.to(dtype=torch.int)).item()
        else:
            partial_dim = postprocessing_channel_factor * 1024
            mask = various.create_split_binary_mask(dim, partial_dim)

        partial_transforms = [transforms.LULinear(partial_dim, identity_init=True)]
        logger.debug("PartialTransform (LULinear) (%s)", partial_dim)
        for _ in range(postprocessing_layers - 1):
            partial_transforms.append(transforms.LogTanh(cut_point=1))
            logger.debug("PartialTransform (LogTanh) (%s)", partial_dim)
            partial_transforms.append(transforms.LULinear(partial_dim, identity_init=True))
            logger.debug("PartialTransform (LULinear) (%s)", partial_dim)
        partial_transform = transforms.CompositeTransform(partial_transforms)

        final_transform = transforms.CompositeTransform([transforms.PartialTransform(mask, partial_transform), transforms.MaskBasedPermutation(mask)])
        logging.debug("MaskBasedPermutation (%s)", mask)

    elif postprocessing == "partial_nsf":
        if multi_scale:
            mask = various.create_mlt_channel_mask(dim, channels_per_level=postprocessing_channel_factor * np.array([1, 2, 4, 16], dtype=np.int), resolution=res)
            partial_dim = torch.sum(mask.to(dtype=torch.int)).item()
        else:
            partial_dim = postprocessing_channel_factor * 1024
            mask = various.create_split_binary_mask(dim, partial_dim)

        partial_transform = create_vector_transform(
            dim=partial_dim, flow_steps=postprocessing_layers, linear_transform_type="permutation", tail_bound=tail_bound, num_bins=num_bins
        )
        logging.debug("RQ-NSF transform on %s features with %s steps", partial_dim, postprocessing_layers)

        final_transform = transforms.CompositeTransform([transforms.PartialTransform(mask, partial_transform), transforms.MaskBasedPermutation(mask)])
        logging.debug("MaskBasedPermutation (%s)", mask)

    elif postprocessing == "permutation":
        # Random permutation
        final_transform = transforms.RandomPermutation(dim)
        logger.debug("RandomPermutation(%s)", dim)

    elif postprocessing == "none":
        final_transform = transforms.IdentityTransform()

    else:
        raise NotImplementedError(postprocessing)
    return final_transform


def _create_preprocessing(alpha, c, h, num_bits, preprocessing, w):
    # Preprocessing
    # Inputs to the model in [0, 2 ** num_bits]
    if preprocessing == "glow":
        # Map to [-0.5,0.5]
        preprocess_transform = transforms.AffineScalarTransform(scale=(1.0 / 2 ** num_bits), shift=-0.5)
        logger.debug("Preprocessing: Glow")
    elif preprocessing == "realnvp":
        preprocess_transform = transforms.CompositeTransform(
            [
                # Map to [0,1]
                transforms.AffineScalarTransform(scale=(1.0 / 2 ** num_bits)),
                # Map into unconstrained space as done in RealNVP
                transforms.AffineScalarTransform(shift=alpha, scale=(1 - alpha)),
                transforms.Logit(),
            ]
        )
        logger.debug("Preprocessing: RealNVP")
    elif preprocessing == "realnvp_2alpha":
        preprocess_transform = transforms.CompositeTransform(
            [transforms.AffineScalarTransform(scale=(1.0 / 2 ** num_bits)), transforms.AffineScalarTransform(shift=alpha, scale=(1 - 2.0 * alpha)), transforms.Logit(),]
        )
        logger.debug("Preprocessing: RealNVP2alpha")
    elif preprocessing == "unflatten":
        preprocess_transform = transforms.ReshapeTransform(input_shape=(c * h * w,), output_shape=(c, h, w))
        logger.debug("Preprocessing: Unflattening from %s to (%s, %s, %s)", c * h * w, c, h, w)
    else:
        raise RuntimeError("Unknown preprocessing type: {}".format(preprocessing))
    return preprocess_transform
