"""
Implementation of the LEAF audio frontend based on the paper

[LEAF: A LEARNABLE FRONTEND FOR AUDIO CLASSIFICATION](https://openreview.net/forum?id=jM76BCb6F9m),
based on the official implementation here https://github.com/google-research/leaf-audio

Written for audax by / Copyright 2022, Sarthak Yadav
"""
import flax.linen.initializers
import jax
from jax import dtypes
import math
from functools import partial
from typing import Union, Callable, Any, Iterable, Optional
from jax import numpy as jnp, lax
from flax import linen as nn
import warnings
from ..commons import utils
from ..core import functional


PRNGKey = Any
Shape = Iterable[int]
Dtype = Any  # this could be a real type?
Array = Any


def squared_modulus_activation(inputs):
    """
    Squared Modules Activation function as used in Leaf
    :param inputs:
    :return:
    """
    inputs = inputs.transpose((0, 2, 1))
    outputs = 2 * nn.avg_pool(inputs ** 2, (2,), strides=(2,))
    return outputs.transpose((0, 2, 1))


def gabor_impulse_response(t, center, fwhm):
    denominator = 1. / (jnp.sqrt(2.0 * math.pi) * fwhm)
    gaussian = jnp.exp(jnp.tensordot(1.0 / (2. * fwhm**2), -t**2, axes=0))
    center_frequency_complex = center.astype(jnp.complex64)
    t_complex = t.astype(jnp.complex64)
    sinusoid = jnp.exp(1j*jnp.tensordot(center_frequency_complex,t_complex,axes=0))
    denominator = denominator.astype(jnp.complex64)[:, jnp.newaxis]
    gaussian = gaussian.astype(jnp.complex64)
    return denominator * sinusoid * gaussian


def gabor_filters(kernel, size: int = 401):
    t = jnp.arange(-(size//2), (size+1)//2, dtype=jnp.float32)
    return gabor_impulse_response(t, center=kernel[:, 0], fwhm=kernel[:, 1])


def gaussian_lowpass_kernel(sigma, filter_size: int):
    sigma = jnp.clip(sigma, a_min=(2. / filter_size), a_max=0.5)
    t = jnp.arange(0, filter_size, dtype=jnp.float32)
    t = t.reshape((1, filter_size, 1, 1))
    numerator = t - 0.5 * (filter_size - 1)
    denominator = sigma * 0.5 * (filter_size - 1)
    return jnp.exp(-0.5 * (numerator / denominator) ** 2)


def constant_init(key, shape, dtype=jnp.float_, constant=0.04):
    return jnp.ones(shape, dtypes.canonicalize_dtype(dtype)) * constant


# TODO: Tidy up gabor_init and this mess, can be really simplified

class GaborFilterHelper:
    def __init__(self,
                 n_filters: int = 40,
                 min_freq: float = 0.,
                 max_freq: float = 8000.,
                 sample_rate: int = 16000,
                 window_len: int = 401,
                 n_fft: int = 512,
                 normalize_energy: bool = False
                 ):
        super(GaborFilterHelper, self).__init__()
        self.n_filters = n_filters
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.sample_rate = sample_rate
        self.window_len = window_len
        self.n_fft = n_fft
        self.normalize_energy = normalize_energy
        mel_filters = functional.melscale_fbanks(n_freqs=(self.n_fft // 2) + 1, n_mels=self.n_filters,
                                                      sample_rate=self.sample_rate, f_min=self.min_freq,
                                                      f_max=self.max_freq, norm=None)
        mel_filters = mel_filters.transpose((1, 0))
        if self.normalize_energy:
            mel_filters = mel_filters / self._mel_filters_areas(mel_filters)
        self.mel_filters = mel_filters

    @property
    def gabor_params_from_mels(self):
        coeff = jnp.sqrt(2. * jnp.log(2.)) * self.n_fft
        sqrt_filters = jnp.sqrt(self.mel_filters)
        center_frequencies = jnp.argmax(sqrt_filters, axis=1)
        peaks = jnp.max(sqrt_filters, axis=1, keepdims=True)
        half_magnitudes = peaks / 2.
        fwhms = jnp.sum(
            (sqrt_filters >= half_magnitudes).astype(jnp.float32), axis=1)
        return jnp.stack([center_frequencies * 2 * jnp.pi / self.n_fft, coeff / (jnp.pi * fwhms)], axis=1)

    def _mel_filters_areas(self, filters):
        peaks = jnp.max(filters, axis=1, keepdims=True)
        return peaks * (jnp.sum((filters > 0).astype(jnp.float32), axis=1, keepdims=True) + 2) * jnp.pi / self.n_fft

    # @property
    # def mel_filters(self):
    #     """
    #     this is called only once at initialization
    #     """
    #     mel_filters = functional.melscale_fbanks(n_freqs=(self.n_fft // 2) + 1, n_mels=self.n_filters,
    #                                              sample_rate=self.sample_rate, f_min=self.min_freq,
    #                                              f_max=self.max_freq, norm=None)
    #     mel_filters = mel_filters.transpose((1, 0))
    #     if self.normalize_energy:
    #         mel_filters = mel_filters / self._mel_filters_areas(mel_filters)
    #     return mel_filters

    # def get_gabor_filters(self):
    #     gabor_params = self.gabor_params_from_mels()
    #     gabor_filters = gabor_filters(gabor_params, size=self.window_len)
    #     output = gabor_filters * jnp.sqrt(
    #         self._mel_filters_areas(self.mel_filters()) * 2 * math.sqrt(math.pi) * gabor_params[:, 1:2]
    #     ).astype(jnp.complex64)
    #     return output


class GaborInit:
    def __init__(self, default_window_len=401, **kwargs):
        super(GaborInit, self).__init__()
        self.def_win_len = default_window_len
        self._kwargs = kwargs

    def __call__(self, shape, dtype=None):
        n_filters = shape[0] if len(shape) == 2 else shape[-1] // 2
        window_len = self.def_win_len if len(shape) == 2 else shape[0]
        gabor_filters = GaborFilterHelper(n_filters=n_filters, window_len=window_len, **self._kwargs)
        if len(shape) == 2:
            return gabor_filters.gabor_params_from_mels
        else:
            # only needed in case of > 2-dim weights
            # even_indices = torch.arange(start=0, end=shape[2], step=2)
            # odd_indices = torch.arange(start=1, end=shape[2], step=2)
            # filters = gabor_filters.gabor_filters()
            raise NotImplementedError("implementation incomplete. Use even valued filter dimensions")


def gabor_init(key, shape, dtype=jnp.float_,
               window_len=401,
               sample_rate=16000, min_freq=60., max_freq=7800.,
               n_fft=512,
               normalize_energy: bool = False):
    init = GaborInit(default_window_len=window_len,
                     sample_rate=sample_rate, min_freq=min_freq, max_freq=max_freq,
                     n_fft=n_fft,
                     normalize_energy=normalize_energy)
    return init(shape, dtype=dtype)


class GaussianLowPassPooling(nn.Module):
    """
    Gaussian Lowpass pooling as found in original LEAF implementation
    """
    kernel_size: int
    stride: int
    in_channels: int
    padding: Union[str, int] = 'SAME'
    use_bias: bool = False
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32
    precision: Any = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = None
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros

    @nn.compact
    def __call__(self, inputs):
        inputs = jnp.asarray(inputs, self.dtype)

        kernel_size = [self.kernel_size]

        def maybe_broadcast(x):
            if x is None:
                # backward compatibility with using None as sentinel for
                # broadcast 1
                x = 1
            if isinstance(x, int):
                return (x,) * len(kernel_size)
            return x

        is_single_input = False
        if inputs.ndim == len(kernel_size) + 1:
            is_single_input = True
            inputs = jnp.expand_dims(inputs, axis=0)

        kernel_shape = (1, 1, self.in_channels, 1)
        kernel = self.param('kernel', self.kernel_init, kernel_shape, self.param_dtype)
        kernel = jnp.asarray(kernel, self.dtype)

        kernel = gaussian_lowpass_kernel(kernel, self.kernel_size)
        kernel = kernel.reshape(-1, self.kernel_size, self.in_channels)
        kernel = kernel.transpose(1, 0, 2)  # kernel needs to be of this shape
        padding_lax = self.padding
        dimension_numbers = utils.conv_dimension_numbers(inputs.shape)
        lhs_dilation = maybe_broadcast(1)
        rhs_dilation = maybe_broadcast(1)
        strides = maybe_broadcast(self.stride)

        y = lax.conv_general_dilated(
            inputs,
            kernel,
            strides,
            padding_lax,
            lhs_dilation=lhs_dilation,
            rhs_dilation=rhs_dilation,
            dimension_numbers=dimension_numbers,
            feature_group_count=self.in_channels,
            precision=self.precision)
        if self.use_bias:
            bias = self.param('bias', self.bias_init, (self.in_channels,), self.param_dtype)
            bias = jnp.asarray(bias, self.dtype)
            y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
        return y


class ExponentialMovingAverage(nn.Module):
    """
    ExponentialMovingAverage as used in LEAF implementation
    Usage:

    >>> init_func = partial(constant_init, constant=0.04)
    >>> exp = ExponentialMovingAverage(in_channels=40, per_channel=True, kernel_init=init_func)
    >>> # now initialize and use like regular linen Module

    """
    in_channels: int
    per_channel: bool = False
    trainable: bool = True
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = None

    @nn.compact
    def __call__(self, inputs):
        if self.kernel_init is None:
            raise ValueError("self.kernel_init is none.")
        inputs = jnp.asarray(inputs, self.dtype)
        kernel_shape = (self.in_channels,) if self.per_channel else (1,)
        if self.trainable:
            kernel = self.param('smooth', self.kernel_init, kernel_shape, self.param_dtype)
        else:
            kernel = self.kernel_init(0, kernel_shape, dtype=self.param_dtype)

        kernel = jnp.clip(kernel, a_min=0., a_max=1.)
        initial_state = inputs[:, 0, :]

        def func(a, x):
            a = kernel * x + (1.0 - kernel) * a
            return a, a

        _, result = lax.scan(func, initial_state, inputs.transpose((1, 0, 2)))
        return result.transpose((1, 0, 2))


class PCENLayer(nn.Module):
    in_channels: int
    alpha_coef: float = 0.96
    smooth_coef: float = 0.04
    delta_coef: float = 2.0
    root_coef: float = 2.0
    floor_coef: float = 1e-6
    trainable: bool = True
    learn_smooth_coef: bool = True
    per_channel_smooth_coef: bool = False
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs) -> Any:
        inputs = inputs.astype(self.dtype)
        assert self.learn_smooth_coef, "Only sPCEN is supported"

        if self.trainable:
            alpha = self.param("alpha", partial(constant_init, constant=self.alpha_coef), [self.in_channels],
                               self.param_dtype)
            delta = self.param("delta", partial(constant_init, constant=self.delta_coef), [self.in_channels],
                               self.param_dtype)
            root = self.param("root", partial(constant_init, constant=self.root_coef), [self.in_channels],
                              self.param_dtype)
        else:
            alpha = constant_init(0, [self.in_channels], constant=self.alpha_coef, dtype=self.param_dtype)
            delta = constant_init(0, [self.in_channels], constant=self.delta_coef, dtype=self.param_dtype)
            root = constant_init(0, [self.in_channels], constant=self.root_coef, dtype=self.param_dtype)

        alpha = jnp.minimum(alpha, 1.0)
        root = jnp.maximum(root, 1.0)

        ema = ExponentialMovingAverage(self.in_channels, self.per_channel_smooth_coef,
                                       trainable=self.trainable, dtype=self.dtype, param_dtype=self.param_dtype,
                                       kernel_init=partial(constant_init, constant=self.smooth_coef))

        ema_smoother = ema(inputs)
        one_over_root = 1. / root
        output = ((inputs / (
                    self.floor_coef + ema_smoother) ** alpha + delta) ** one_over_root - delta ** one_over_root)
        output = output.astype(self.dtype)
        return output


class GaborConstraint:
    def __init__(self, kernel_size: int):
        self.kernel_size = kernel_size

    def __call__(self, kernel):
        mu_lower = 0.
        mu_upper = math.pi
        sigma_lower = 4 * math.sqrt(2 * math.log(2)) / math.pi
        sigma_upper = self.kernel_size * math.sqrt(2 * math.log(2)) / math.pi
        clipped_mu = jnp.clip(kernel[:, 0], mu_lower, mu_upper)
        clipped_sigma = jnp.clip(kernel[:, 1], sigma_lower, sigma_upper)
        return jnp.stack([clipped_mu, clipped_sigma], axis=1)


class GaborConv1D(nn.Module):
    filters: int
    kernel_size: int
    stride: int
    padding: str = "SAME"
    use_bias: bool = False
    sort_filter: bool = False
    constraint: Optional[Callable] = None
    kernel_init: Optional[Callable[[PRNGKey, Shape, Dtype], Array]] = None
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    precision: Any = None

    @nn.compact
    def __call__(self, inputs):
        inputs = jnp.asarray(inputs, self.dtype)
        kernel_size = [self.kernel_size]
        num_filters = self.filters // 2

        if not self.kernel_init:
            raise ValueError("Initialized without kernel_init argument")

        def maybe_broadcast(x):
            if x is None:
                # backward compatibility with using None as sentinel for
                # broadcast 1
                x = 1
            if isinstance(x, int):
                return (x,) * len(kernel_size)
            return x

        kernel = self.param('gabor_kernel', self.kernel_init, [num_filters, 2], self.param_dtype)
        if self.constraint is None:
            warnings.warn("GaborConv1d was initialized without a constraint..")
        else:
            kernel = self.constraint(kernel)
        if self.sort_filter:
            # TODO add sort filter functionality
            raise NotImplementedError("filter sorting not implemented yet")
        filters = gabor_filters(kernel, self.kernel_size)
        real_filters = jnp.real(filters)
        imag_filters = jnp.imag(filters)
        stacked_filters = jnp.stack([real_filters, imag_filters], axis=1)
        stacked_filters = stacked_filters.reshape((2 * num_filters, self.kernel_size))
        stacked_filters = jnp.expand_dims(stacked_filters.transpose((1, 0)), axis=1)

        padding_lax = self.padding
        dimension_numbers = utils.conv_dimension_numbers(inputs.shape)
        lhs_dilation = maybe_broadcast(1)
        rhs_dilation = maybe_broadcast(1)
        stride = maybe_broadcast(self.stride)
        y = lax.conv_general_dilated(
            inputs,
            stacked_filters,
            stride,
            padding_lax,
            lhs_dilation=lhs_dilation,
            rhs_dilation=rhs_dilation,
            dimension_numbers=dimension_numbers,
            feature_group_count=1,
            precision=self.precision)

        if self.use_bias:
            bias = self.param('gabor_bias', self.bias_init, (num_filters * 2,), self.param_dtype)
            bias = jnp.asarray(bias, self.dtype)
            y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
        return y


class Leaf(nn.Module):
    n_filters: int = 40
    sample_rate: int = 16000
    window_len: float = 25.
    window_stride: float = 10.
    min_freq: float = 60.
    max_freq: float = 7800.
    complex_conv_init: Optional[Union[Callable, str]] = "default"
    train_pcen: Optional[bool] = True
    pooling_param = 0.4
    use_bias: bool = False
    spec_augment: bool = False
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32
    precision: Any = None

    def setup(self):
        window_size = int(self.sample_rate * self.window_len // 1000 + 1)
        window_stride = int(self.sample_rate * self.window_stride // 1000)
        if type(self.complex_conv_init) == str:
            if self.complex_conv_init.lower() == "default":
                conv_init_func = partial(gabor_init,
                                         window_len=window_size,
                                         sample_rate=self.sample_rate,
                                         min_freq=self.min_freq,
                                         max_freq=self.max_freq,
                                         n_fft=512,
                                         normalize_energy=False)
            elif self.complex_conv_init.lower() == "randn":
                conv_init_func = flax.linen.initializers.normal()
            elif self.complex_conv_init.lower() == "lecun_normal":
                conv_init_func = flax.linen.initializers.lecun_normal()
            elif self.complex_conv_init.lower() == "kaiming_normal":
                conv_init_func = flax.linen.initializers.kaiming_normal()
            else:
                raise ValueError("Unsupported 'str' valued argument {}, should be one of [randn, lecun_normal, kaiming_normal]. Pass a callable instead")
        else:
            conv_init_func = self.complex_conv_init

        self.complex_conv = GaborConv1D(filters=self.n_filters*2,
                                        kernel_size=window_size, stride=1,
                                        use_bias=self.use_bias,
                                        constraint=GaborConstraint(window_size),
                                        kernel_init=conv_init_func)

        self.pooling = GaussianLowPassPooling(kernel_size=window_size, stride=window_stride,
                                              in_channels=self.n_filters, use_bias=False, dtype=self.dtype,
                                              param_dtype=self.param_dtype, precision=self.precision,
                                              kernel_init=partial(constant_init,
                                                                   constant=self.pooling_param))
        self.compression = PCENLayer(in_channels=self.n_filters, alpha_coef=0.96,
                                     smooth_coef=0.04, delta_coef=2.0, floor_coef=1e-12,
                                     trainable=self.train_pcen, learn_smooth_coef=True,
                                     per_channel_smooth_coef=True, dtype=self.dtype)
        if self.spec_augment:
            raise NotImplementedError("spec augment functionality not yet added.")

    def __call__(self, inputs):
        outputs = inputs[:, :, jnp.newaxis] if inputs.ndim < 3 else inputs
        outputs = outputs.astype(self.dtype)
        outputs = self.complex_conv(outputs)
        outputs = squared_modulus_activation(outputs)
        outputs = self.pooling(outputs)
        outputs = jnp.maximum(outputs, 1e-5)
        outputs = self.compression(outputs)
        outputs = outputs.astype(self.dtype)
        return outputs
