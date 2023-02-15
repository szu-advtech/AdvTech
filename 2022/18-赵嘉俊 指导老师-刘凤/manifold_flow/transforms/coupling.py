"""Implementations of various coupling layers."""
import warnings
import logging
import numpy as np
import torch

from manifold_flow import transforms
from manifold_flow.transforms import splines
from manifold_flow.utils import various

logger = logging.getLogger(__name__)


class CouplingTransform(transforms.Transform):
    """A base class for coupling layers. Supports 2D inputs (NxD), as well as 4D inputs for
    images (NxCxHxW). For images the splitting is done on the channel dimension, using the
    provided 1D mask."""

    def __init__(self, mask, transform_net_create_fn, unconditional_transform=None):
        """
        Constructor.

        Args:
            mask: a 1-dim tensor, tuple or list. It indexes inputs as follows:
                * If `mask[i] > 0`, `input[i]` will be transformed.
                * If `mask[i] <= 0`, `input[i]` will be passed unchanged.
        """
        mask = torch.as_tensor(mask)
        if mask.dim() != 1:
            raise ValueError("Mask must be a 1-dim tensor.")
        if mask.numel() <= 0:
            raise ValueError("Mask can't be empty.")

        super().__init__()
        self.features = len(mask)
        features_vector = torch.arange(self.features)

        self.register_buffer("identity_features", features_vector.masked_select(mask <= 0))
        self.register_buffer("transform_features", features_vector.masked_select(mask > 0))

        assert self.num_identity_features + self.num_transform_features == self.features

        self.transform_net = transform_net_create_fn(self.num_identity_features, self.num_transform_features * self._transform_dim_multiplier())

        if unconditional_transform is None:
            self.unconditional_transform = None
        else:
            self.unconditional_transform = unconditional_transform(features=self.num_identity_features)

    @property
    def num_identity_features(self):
        return len(self.identity_features)

    @property
    def num_transform_features(self):
        return len(self.transform_features)

    def forward(self, inputs, context=None, full_jacobian=False):
        if inputs.dim() not in [2, 4]:
            raise ValueError("Inputs must be a 2D or a 4D tensor.")

        if inputs.shape[1] != self.features:
            raise ValueError("Expected features = {}, got {}.".format(self.features, inputs.shape[1]))

        identity_split = inputs[:, self.identity_features, ...]
        transform_split = inputs[:, self.transform_features, ...]

        # logger.debug("identity_split depends on inputs: %s", utils.check_dependence(identity_split, inputs))
        # logger.debug("transform_split depends on inputs: %s", utils.check_dependence(transform_split, inputs))

        if full_jacobian:
            transform_params = self.transform_net(identity_split, context)

            # logger.debug("transform_params depends on inputs: %s", utils.check_dependence(transform_params, inputs))

            transform_split, _ = self._coupling_transform_forward(inputs=transform_split, transform_params=transform_params)
            # logger.debug("transform_split depends on inputs: %s", utils.check_dependence(transform_split, inputs))
            # logger.debug("Jacobian: %s", utils.calculate_jacobian(transform_split, inputs))
            # logger.debug("Batch Jacobian: %s", utils.batch_jacobian(transform_split, inputs))

            # timer.timer(start="Jacobian coupling transform")

            jacobian_transform = various.batch_jacobian(transform_split, inputs)

            if self.unconditional_transform is not None:
                identity_split, jacobian_identity = self.unconditional_transform(identity_split, context)
                raise NotImplementedError()
            # else:
            #     jacobian_identity = torch.eye(self.num_identity_features).unsqueeze(0)  # (1, n, n)
            #     logger.debug("Identity Jacobian: %s", jacobian_identity[0])

            # Put together full Jacobian
            batchsize = inputs.size(0)
            jacobian = torch.zeros((batchsize,) + inputs.size()[1:] + inputs.size()[1:])
            jacobian[:, self.identity_features, self.identity_features] = 1.0
            jacobian[:, self.transform_features, :] = jacobian_transform

            outputs = torch.empty_like(inputs)
            outputs[:, self.identity_features, ...] = identity_split
            outputs[:, self.transform_features, ...] = transform_split

            # logger.debug("Jacobian from coupling layer (identity features = %s): \n %s", self.identity_features, jacobian[0])
            # timer.timer(stop="Jacobian coupling transform")

            return outputs, jacobian

        else:
            transform_params = self.transform_net(identity_split, context)

            transform_split, logabsdet = self._coupling_transform_forward(inputs=transform_split, transform_params=transform_params)

            if self.unconditional_transform is not None:
                identity_split, logabsdet_identity = self.unconditional_transform(identity_split, context)
                logabsdet += logabsdet_identity

            outputs = torch.empty_like(inputs)
            outputs[:, self.identity_features, ...] = identity_split
            outputs[:, self.transform_features, ...] = transform_split

            # # debugging
            # check_inputs, _ = self.inverse(outputs, context=context)
            # diff = torch.sum((check_inputs - inputs)**2, dim=1)
            # if torch.max(diff > 0.1):
            #     logger.debug("Coupling trf inversion imprecise!")
            #
            #     inputs_ = inputs[diff > 0.1]
            #     outputs_ = self.forward(inputs_, context=context)

            return outputs, logabsdet

    def inverse(self, inputs, context=None, full_jacobian=False):
        if inputs.dim() not in [2, 4]:
            raise ValueError("Inputs must be a 2D or a 4D tensor.")

        if inputs.shape[1] != self.features:
            raise ValueError("Expected features = {}, got {}.".format(self.features, inputs.shape[1]))

        identity_split = inputs[:, self.identity_features, ...]
        transform_split = inputs[:, self.transform_features, ...]

        if full_jacobian:
            # timer.timer(start="Jacobian inverse coupling transform")

            if self.unconditional_transform is not None:
                # identity_split_after_unconditional_transform, jacobian_identity = self.unconditional_transform.inverse(
                #     identity_split, context, full_jacobian=True
                # )
                raise NotImplementedError()
            else:
                identity_split_after_unconditional_transform = identity_split
            #     jacobian_identity = torch.eye(self.num_identity_features).unsqueeze(0)  # (1, n, n)

            transform_params = self.transform_net(identity_split, context)

            transform_split, _ = self._coupling_transform_inverse(inputs=transform_split, transform_params=transform_params)
            jacobian_transform = various.batch_jacobian(transform_split, inputs)

            # Put together full Jacobian
            batchsize = inputs.size(0)
            jacobian = torch.zeros((batchsize,) + inputs.size()[1:] + inputs.size()[1:])
            jacobian[:, self.identity_features, self.identity_features] = 1.0
            jacobian[:, self.transform_features, :] = jacobian_transform

            outputs = torch.empty_like(inputs)
            outputs[:, self.identity_features] = identity_split_after_unconditional_transform
            outputs[:, self.transform_features] = transform_split

            # timer.timer(stop="Jacobian inverse coupling transform")

            return outputs, jacobian

        else:
            logabsdet = 0.0
            if self.unconditional_transform is not None:
                identity_split, logabsdet = self.unconditional_transform.inverse(identity_split, context)

            transform_params = self.transform_net(identity_split, context)
            transform_split, logabsdet_split = self._coupling_transform_inverse(inputs=transform_split, transform_params=transform_params)
            logabsdet += logabsdet_split

            outputs = torch.empty_like(inputs)
            outputs[:, self.identity_features] = identity_split
            outputs[:, self.transform_features] = transform_split

            return outputs, logabsdet

    def _transform_dim_multiplier(self):
        """Number of features to output for each transform dimension."""
        raise NotImplementedError()

    def _coupling_transform_forward(self, inputs, transform_params, full_jacobian=False):
        """Forward pass of the coupling transform."""
        raise NotImplementedError()

    def _coupling_transform_inverse(self, inputs, transform_params, full_jacobian=False):
        """Inverse of the coupling transform."""
        raise NotImplementedError()


class AffineCouplingTransform(CouplingTransform):
    """An affine coupling layer that scales and shifts part of the variables.

    Reference:
    > L. Dinh et al., Density estimation using Real NVP, ICLR 2017.
    """

    def _transform_dim_multiplier(self):
        return 2

    def _scale_and_shift(self, transform_params):
        unconstrained_scale = transform_params[:, self.num_transform_features :, ...]
        shift = transform_params[:, : self.num_transform_features, ...]
        # scale = (F.softplus(unconstrained_scale) + 1e-3).clamp(0, 3)
        scale = torch.sigmoid(unconstrained_scale + 2) + 1e-3
        return scale, shift

    def _coupling_transform_forward(self, inputs, transform_params, full_jacobian=False):
        scale, shift = self._scale_and_shift(transform_params)
        log_scale = torch.log(scale)
        outputs = inputs * scale + shift
        if full_jacobian:
            jacobian = various.batch_diagonal(scale)
            return outputs, jacobian
        else:
            logabsdet = various.sum_except_batch(log_scale, num_batch_dims=1)
            return outputs, logabsdet

    def _coupling_transform_inverse(self, inputs, transform_params, full_jacobian=False):
        scale, shift = self._scale_and_shift(transform_params)
        log_scale = torch.log(scale)
        outputs = (inputs - shift) / scale
        if full_jacobian:
            jacobian = -various.batch_diagonal(scale)
            return outputs, jacobian
        else:
            logabsdet = -various.sum_except_batch(log_scale, num_batch_dims=1)
            return outputs, logabsdet


class AdditiveCouplingTransform(AffineCouplingTransform):
    """An additive coupling layer, i.e. an affine coupling layer without scaling.

    Reference:
    > L. Dinh et al., NICE:  Non-linear  Independent  Components  Estimation,
    > arXiv:1410.8516, 2014.
    """

    def _transform_dim_multiplier(self):
        return 1

    def _scale_and_shift(self, transform_params):
        shift = transform_params
        scale = torch.ones_like(shift)
        return scale, shift


class PiecewiseCouplingTransform(CouplingTransform):
    def _coupling_transform_forward(self, inputs, transform_params, full_jacobian=False):
        return self._coupling_transform(inputs, transform_params, inverse=False, full_jacobian=full_jacobian)

    def _coupling_transform_inverse(self, inputs, transform_params, full_jacobian=False):
        return self._coupling_transform(inputs, transform_params, inverse=True, full_jacobian=full_jacobian)

    def _coupling_transform(self, inputs, transform_params, inverse=False, full_jacobian=False):
        if inputs.dim() == 4:
            b, c, h, w = inputs.shape
            # For images, reshape transform_params from Bx(C*?)xHxW to BxCxHxWx?
            transform_params = transform_params.reshape(b, c, -1, h, w).permute(0, 1, 3, 4, 2)
        elif inputs.dim() == 2:
            b, d = inputs.shape
            # For 2D data, reshape transform_params from Bx(D*?) to BxDx?
            transform_params = transform_params.reshape(b, d, -1)

        if full_jacobian:
            outputs, jacobian = self._piecewise_cdf(inputs, transform_params, inverse, full_jacobian=True)
            return outputs, jacobian
        else:
            outputs, logabsdet = self._piecewise_cdf(inputs, transform_params, inverse)
            return outputs, various.sum_except_batch(logabsdet)

    def _piecewise_cdf(self, inputs, transform_params, inverse=False, full_jacobian=False):
        raise NotImplementedError()


class PiecewiseLinearCouplingTransform(PiecewiseCouplingTransform):
    """
    Reference:
    > Müller et al., Neural Importance Sampling, arXiv:1808.03856, 2018.
    """

    def __init__(self, mask, transform_net_create_fn, num_bins=10, tails=None, tail_bound=1.0, apply_unconditional_transform=False, img_shape=None):
        self.num_bins = num_bins
        self.tails = tails
        self.tail_bound = tail_bound

        if apply_unconditional_transform:
            unconditional_transform = lambda features: transforms.PiecewiseLinearCDF(
                shape=[features] + (img_shape if img_shape else []), num_bins=num_bins, tails=tails, tail_bound=tail_bound
            )
        else:
            unconditional_transform = None

        super().__init__(mask, transform_net_create_fn, unconditional_transform=unconditional_transform)

    def _transform_dim_multiplier(self):
        return self.num_bins

    def _piecewise_cdf(self, inputs, transform_params, inverse=False, full_jacobian=False):
        unnormalized_pdf = transform_params

        if self.tails is None:
            return splines.linear_spline(inputs=inputs, unnormalized_pdf=unnormalized_pdf, inverse=inverse, full_jacobian=full_jacobian)
        else:
            return splines.unconstrained_linear_spline(
                inputs=inputs, unnormalized_pdf=unnormalized_pdf, inverse=inverse, tails=self.tails, tail_bound=self.tail_bound, full_jacobian=full_jacobian
            )


class PiecewiseQuadraticCouplingTransform(PiecewiseCouplingTransform):
    """
    Reference:
    > Müller et al., Neural Importance Sampling, arXiv:1808.03856, 2018.
    """

    def __init__(
        self,
        mask,
        transform_net_create_fn,
        num_bins=10,
        tails=None,
        tail_bound=1.0,
        apply_unconditional_transform=False,
        img_shape=None,
        min_bin_width=splines.quadratic.DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=splines.quadratic.DEFAULT_MIN_BIN_HEIGHT,
    ):
        self.num_bins = num_bins
        self.tails = tails
        self.tail_bound = tail_bound
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height

        if apply_unconditional_transform:
            unconditional_transform = lambda features: transforms.PiecewiseQuadraticCDF(
                shape=[features] + (img_shape if img_shape else []),
                num_bins=num_bins,
                tails=tails,
                tail_bound=tail_bound,
                min_bin_width=min_bin_width,
                min_bin_height=min_bin_height,
            )
        else:
            unconditional_transform = None

        super().__init__(mask, transform_net_create_fn, unconditional_transform=unconditional_transform)

    def _transform_dim_multiplier(self):
        if self.tails == "linear":
            return self.num_bins * 2 - 1
        else:
            return self.num_bins * 2 + 1

    def _piecewise_cdf(self, inputs, transform_params, inverse=False, full_jacobian=False):
        unnormalized_widths = transform_params[..., : self.num_bins]
        unnormalized_heights = transform_params[..., self.num_bins :]

        if hasattr(self.transform_net, "hidden_features"):
            unnormalized_widths /= np.sqrt(self.transform_net.hidden_features)
            unnormalized_heights /= np.sqrt(self.transform_net.hidden_features)

        if self.tails is None:
            spline_fn = splines.quadratic_spline
            spline_kwargs = {}
        else:
            spline_fn = splines.unconstrained_quadratic_spline
            spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}

        return spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            full_jacobian=full_jacobian,
            **spline_kwargs
        )


class PiecewiseCubicCouplingTransform(PiecewiseCouplingTransform):
    def __init__(
        self,
        mask,
        transform_net_create_fn,
        num_bins=10,
        tails=None,
        tail_bound=1.0,
        apply_unconditional_transform=False,
        img_shape=None,
        min_bin_width=splines.cubic.DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=splines.cubic.DEFAULT_MIN_BIN_HEIGHT,
    ):

        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.tails = tails
        self.tail_bound = tail_bound

        if apply_unconditional_transform:
            unconditional_transform = lambda features: transforms.PiecewiseCubicCDF(
                shape=[features] + (img_shape if img_shape else []),
                num_bins=num_bins,
                tails=tails,
                tail_bound=tail_bound,
                min_bin_width=min_bin_width,
                min_bin_height=min_bin_height,
            )
        else:
            unconditional_transform = None

        super().__init__(mask, transform_net_create_fn, unconditional_transform=unconditional_transform)

    def _transform_dim_multiplier(self):
        return self.num_bins * 2 + 2

    def _piecewise_cdf(self, inputs, transform_params, inverse=False, full_jacobian=False):
        unnormalized_widths = transform_params[..., : self.num_bins]
        unnormalized_heights = transform_params[..., self.num_bins : 2 * self.num_bins]
        unnorm_derivatives_left = transform_params[..., 2 * self.num_bins][..., None]
        unnorm_derivatives_right = transform_params[..., 2 * self.num_bins + 1][..., None]

        if hasattr(self.transform_net, "hidden_features"):
            unnormalized_widths /= np.sqrt(self.transform_net.hidden_features)
            unnormalized_heights /= np.sqrt(self.transform_net.hidden_features)

        if self.tails is None:
            spline_fn = splines.cubic_spline
            spline_kwargs = {}
        else:
            spline_fn = splines.unconstrained_cubic_spline
            spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}

        return spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnorm_derivatives_left=unnorm_derivatives_left,
            unnorm_derivatives_right=unnorm_derivatives_right,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            full_jacobian=full_jacobian,
            **spline_kwargs
        )


class PiecewiseRationalQuadraticCouplingTransform(PiecewiseCouplingTransform):
    def __init__(
        self,
        mask,
        transform_net_create_fn,
        num_bins=10,
        tails=None,
        tail_bound=1.0,
        apply_unconditional_transform=False,
        img_shape=None,
        min_bin_width=splines.rational_quadratic.DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=splines.rational_quadratic.DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=splines.rational_quadratic.DEFAULT_MIN_DERIVATIVE,
    ):

        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.tails = tails
        self.tail_bound = tail_bound

        if apply_unconditional_transform:
            unconditional_transform = lambda features: transforms.PiecewiseRationalQuadraticCDF(
                shape=[features] + (img_shape if img_shape else []),
                num_bins=num_bins,
                tails=tails,
                tail_bound=tail_bound,
                min_bin_width=min_bin_width,
                min_bin_height=min_bin_height,
                min_derivative=min_derivative,
            )
        else:
            unconditional_transform = None

        super().__init__(mask, transform_net_create_fn, unconditional_transform=unconditional_transform)

    def _transform_dim_multiplier(self):
        if self.tails == "linear":
            return self.num_bins * 3 - 1
        else:
            return self.num_bins * 3 + 1

    def _piecewise_cdf(self, inputs, transform_params, inverse=False, full_jacobian=False):
        unnormalized_widths = transform_params[..., : self.num_bins]
        unnormalized_heights = transform_params[..., self.num_bins : 2 * self.num_bins]
        unnormalized_derivatives = transform_params[..., 2 * self.num_bins :]

        if hasattr(self.transform_net, "hidden_features"):
            unnormalized_widths /= np.sqrt(self.transform_net.hidden_features)
            unnormalized_heights /= np.sqrt(self.transform_net.hidden_features)
        elif hasattr(self.transform_net, "hidden_channels"):
            unnormalized_widths /= np.sqrt(self.transform_net.hidden_channels)
            unnormalized_heights /= np.sqrt(self.transform_net.hidden_channels)
        else:
            warnings.warn("Inputs to the softmax are not scaled down: initialization might be bad.")

        if self.tails is None:
            spline_fn = splines.rational_quadratic_spline
            spline_kwargs = {}
        else:
            spline_fn = splines.unconstrained_rational_quadratic_spline
            spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}

        return spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            full_jacobian=full_jacobian,
            **spline_kwargs
        )
