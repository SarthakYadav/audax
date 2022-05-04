"""
Implementation of the SincNet audio frontend based on the paper

Ravanelli, Mirco, and Yoshua Bengio. "Speaker recognition from raw waveform with sincnet."
In 2018 IEEE Spoken Language Technology Workshop (SLT), pp. 1021-1028. IEEE, 2018.

Written for audax by / Copyright 2022, Sarthak Yadav
"""
import jax
import math
from functools import partial
from typing import Union, Callable, Any, Iterable, Optional
from jax import numpy as jnp, lax
from flax import linen as nn
from ..commons import utils

PRNGKey = Any
Shape = Iterable[int]
Dtype = Any  # this could be a real type?
Array = Any


def sinc_impulse_response(t: jnp.array, frequency: jnp.array) -> jnp.array:
    """Computes the sinc impulse response."""
    return jnp.sin(2 * math.pi * frequency * t) / (2 * math.pi * frequency * t)


def sinc_filters(cutoff_freq_low: jnp.array,
                 cutoff_freq_high: jnp.array,
                 size: int = 401,
                 sample_rate: int = 16000) -> jnp.array:
    """Computes the sinc filters from its parameters for a given size.

    Sinc is not defined in zero so we need to separately compute negative
    (left_range) and positive part (right_range).

    Args:
        cutoff_freq_low: tf.Tensor<float>[1, filters] the lower cutoff frequencies
        of the bandpass.
        cutoff_freq_high: tf.Tensor<float>[1, filters] the upper cutoff frequencies
        of the bandpass.
        size: the size of the output tensor.
        sample_rate: audio sampling rate

    Returns:
        A jnp.tensor<float>[size, filters].
    """
    left_range = jnp.arange(-(size // 2), 0, dtype=jnp.float32)[:, jnp.newaxis] / float(sample_rate)
    right_range = jnp.arange(1, (size // 2) + 1, dtype=jnp.float32)[:, jnp.newaxis] / float(sample_rate)

    high_pass_left_range = 2 * cutoff_freq_high * sinc_impulse_response(left_range, cutoff_freq_high)
    high_pass_right_range = 2 * cutoff_freq_high * sinc_impulse_response(right_range, cutoff_freq_high)

    low_pass_left_range = 2 * cutoff_freq_low * sinc_impulse_response(left_range, cutoff_freq_low)
    low_pass_right_range = 2 * cutoff_freq_low * sinc_impulse_response(right_range, cutoff_freq_low)

    high_pass = jnp.concatenate([high_pass_left_range, 2 * cutoff_freq_high, high_pass_right_range], axis=0)
    low_pass = jnp.concatenate([low_pass_left_range, 2 * cutoff_freq_low, low_pass_right_range], axis=0)

    band_pass = high_pass - low_pass
    return band_pass / jnp.max(band_pass, axis=0, keepdims=True)


def sinc_init(key, shape, dtype=jnp.float_,
              sample_rate=16000, min_low_hz=50., min_band_hz=50.,
              sampling="default"):
    init = SincInit(sample_rate=sample_rate, min_low_hz=min_low_hz, 
                    min_band_hz=min_band_hz, sampling=sampling)
    return init(key, shape, dtype=dtype)


class SincInit:
    def __init__(self,
                 sample_rate: int = 16000,
                 min_low_hz: float = 50.,
                 min_band_hz: float = 50.,
                 sampling="default"):
        self._sample_rate = sample_rate
        self._min_low_hz = min_low_hz
        self._min_band_hz = min_band_hz
        self._sampling = sampling

    def __call__(self, key, shape, dtype=jnp.float32):
        filters = shape[0]
        low_hz = self._min_low_hz
        high_hz = 0.5 * self._sample_rate - (low_hz + self._min_band_hz)
        low_mel = utils.hz2mel(low_hz)
        high_mel = utils.hz2mel(high_hz)

        if self._sampling == "uniform":
            bandpass_mel = jax.random.uniform(key, (filters+1,), dtype=dtype, 
                                             minval=low_mel, maxval=high_mel)
            bandpass_mel = bandpass_mel.sort()
            bandpass_hz = utils.mel2hz(bandpass_mel)
        elif self._sampling == "truncated_normal":
            # sample from truncated normal between zero and one
            # multiply by high_mel
            bandpass_mel = jax.random.truncated_normal(key, 0., 1., (filters+1,), dtype=dtype)
            # a simple min max scale
            bandpass_mel_std = (bandpass_mel - bandpass_mel.min(axis=0)) / (bandpass_mel.max(axis=0) - bandpass_mel.min(axis=0))
            bandpass_mel = bandpass_mel_std * (high_mel - low_mel) + low_mel
            bandpass_mel = bandpass_mel.sort()
            bandpass_hz = utils.mel2hz(bandpass_mel)
        elif self._sampling == "uniform_v2":
            bandpass_hz = jax.random.uniform(key, (filters+1,), dtype=dtype,
                                            minval=low_hz, maxval=high_hz)
            bandpass_hz = bandpass_hz.sort()
        elif self._sampling == "truncated_normal_v2":
            bandpass_hz = jax.random.truncated_normal(key, 0., 1., (filters+1,), dtype=dtype)
            bandpass_hz_std = (bandpass_hz - bandpass_hz.min(axis=0)) / (bandpass_hz.max(axis=0) - bandpass_hz.min(axis=0))
            bandpass_hz = bandpass_hz_std * (high_hz - low_hz) + low_hz
            bandpass_hz = bandpass_hz.sort()
        else:
            bandpass_mel = jnp.linspace(low_mel, high_mel, filters + 1)
            bandpass_hz = utils.mel2hz(bandpass_mel)

        left_edge = bandpass_hz[:-1]
        filter_width = bandpass_hz[1:] - bandpass_hz[:-1]
        return jnp.stack([left_edge, filter_width], axis=1).astype(dtype)


class SincConv1D(nn.Module):
    filters: int
    kernel_size: int
    stride: int
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array]
    window_fn: Optional[Callable] = jnp.hamming
    trainable: bool = True
    padding: str = "SAME"
    sample_rate: int = 16000
    min_low_hz: float = 50.
    min_band_hz: float = 50.
    use_bias: bool = False
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    precision: Any = None

    @nn.compact
    def __call__(self, inputs):
        inputs = jnp.asarray(inputs, self.dtype)
        kernel_size = [self.kernel_size]
        num_filters = self.filters // 2

        if not self.kernel_init:
            raise ValueError("initialized without kernel_init")

        # if not self.window:
        #     raise ValueError("initialized without a window")

        # if len(self.window) != self.kernel_size:
        #     raise ValueError("window size: {} is not equal to kernel_size: {}".format(len(self.window), kernel_size))

        def maybe_broadcast(x):
            if x is None:
                # backward compatibility with using None as sentinel for
                # broadcast 1
                x = 1
            if isinstance(x, int):
                return (x,) * len(kernel_size)
            return x

        kernel = self.param('sinc_kernel', self.kernel_init, [num_filters, 2], self.param_dtype)
        window = self.window_fn(self.kernel_size)
        left_edge = self.min_low_hz + jnp.abs(kernel[:, 0])
        right_edge = jnp.clip(
            left_edge + self.min_band_hz + jnp.abs(kernel[:, 1]),
            a_min=self.min_low_hz,
            a_max=self.sample_rate / 2
        )
        filters = sinc_filters(
            left_edge[jnp.newaxis, :], right_edge[jnp.newaxis, :],
            self.kernel_size) * window[:, jnp.newaxis]
        filters = jnp.expand_dims(filters, axis=1)
        padding_lax = self.padding
        dimension_numbers = utils.conv_dimension_numbers(inputs.shape)
        lhs_dilation = maybe_broadcast(1)
        rhs_dilation = maybe_broadcast(1)
        stride = maybe_broadcast(self.stride)

        y = lax.conv_general_dilated(
            inputs,
            filters,
            stride,
            padding_lax,
            lhs_dilation=lhs_dilation,
            rhs_dilation=rhs_dilation,
            dimension_numbers=dimension_numbers,
            feature_group_count=1,
            precision=self.precision)

        if self.use_bias:
            bias = self.param('sinc_bias', self.bias_init, (num_filters,), self.param_dtype)
            bias = jnp.asarray(bias, self.dtype)
            y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
        return y


class SincNet(nn.Module):
    n_filters: int = 40
    sample_rate: int = 16000
    window_len: float = 25.
    window_stride: float = 10.
    min_low_hz = 50.
    min_band_hz = 50.
    pooling_param = 0.4
    use_bias: bool = False
    spec_augment: bool = False
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32
    complex_conv_init: Optional[Union[Callable, str]] = "default"
    precision: Any = None

    def setup(self):
        window_size = int(self.sample_rate * self.window_len // 1000 + 1)
        window_stride = int(self.sample_rate * self.window_stride // 1000)
        self.complex_conv = SincConv1D(filters=self.n_filters * 2,
                                       kernel_size=window_size, stride=1,
                                       use_bias=self.use_bias,
                                       sample_rate=self.sample_rate,
                                       kernel_init=partial(sinc_init,
                                                           sample_rate=self.sample_rate,
                                                           min_low_hz=self.min_low_hz,
                                                           min_band_hz=self.min_band_hz,
                                                           sampling=self.complex_conv_init))

        self.pooling = partial(nn.max_pool, window_shape=(window_size,), strides=(window_stride,), padding="SAME")
        self.compression = nn.LayerNorm()

        if self.spec_augment:
            raise NotImplementedError("spec augment functionality not yet added.")

    def __call__(self, inputs):
        outputs = inputs[:, :, jnp.newaxis] if inputs.ndim < 3 else inputs
        outputs = outputs.astype(self.dtype)
        outputs = self.complex_conv(outputs)
        outputs = nn.leaky_relu(outputs, negative_slope=0.2)
        outputs = self.pooling(outputs)
        outputs = jnp.maximum(outputs, 1e-5)
        outputs = self.compression(outputs)
        outputs = outputs.astype(self.dtype)
        return outputs
