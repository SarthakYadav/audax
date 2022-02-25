"""
helper functions for EfficientNet CNN models

Written for audax by / Copyright 2022, Sarthak Yadav
"""
import jax
from jax import numpy as jnp
from flax import linen as nn
from typing import Callable, Any, Optional
from functools import partial


class SqueezeAndExcitation(nn.Module):
    """
    Squeeze-and-Excitation block based on reference implementation in torchvision
    https://github.com/pytorch/vision/blob/4bf6c6e4b4cf2dea9c7f9952aa30d5820ab8ae33/torchvision/ops/misc.py
    """
    input_channels: int
    squeeze_channels: int
    activation: Callable = nn.relu
    scale_activation: Callable = nn.sigmoid
    kernel_init: Callable = nn.initializers.kaiming_normal()
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        conv = partial(nn.Conv, kernel_size=(1, 1), padding="VALID",
                       use_bias=False, dtype=self.dtype, kernel_init=self.kernel_init)

        # AdaptiveAvgPool
        scale = jnp.asarray(x, jnp.float32)
        scale = scale.mean((1, 2), keepdims=True)
        scale = jnp.asarray(scale, self.dtype)

        scale = conv(self.squeeze_channels, name="reduce")(scale)
        scale = self.activation(scale)
        scale = conv(self.input_channels, name="expand")(scale)
        scale = self.scale_activation(scale)
        return x * scale


class ConvNormActivation(nn.Module):
    out_channels: int
    kernel_size: int = 3
    stride: int = 1
    padding: Optional[int] = None
    groups: int = 1
    norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm
    activation_layer: Callable = nn.silu
    dilation: int = 1
    bias: Optional[bool] = None
    dtype: Any = jnp.float32
    conv_kernel_init: Callable = nn.initializers.kaiming_normal()

    @nn.compact
    def __call__(self, x, train: bool = True):

        if type(self.kernel_size) == int:
            kernel_size = (self.kernel_size,) * 2
        else:
            kernel_size = self.kernel_size

        if self.padding is None:
            # self.padding = (self.kernel_size - 1) // 2 * self.dilation
            padding = "SAME"
        if self.bias is None:
            bias = self.norm_layer is None

        x = nn.Conv(self.out_channels, kernel_size,
                    strides=self.stride, padding=padding,
                    kernel_dilation=self.dilation, feature_group_count=self.groups,
                    use_bias=bias, dtype=self.dtype, kernel_init=self.conv_kernel_init)(x)

        if self.norm_layer:
            if self.norm_layer == nn.BatchNorm:
                norm = partial(self.norm_layer,
                               use_running_average=not train,
                               momentum=0.9,
                               epsilon=1e-5,
                               dtype=self.dtype)
            else:
                norm = partial(self.norm_layer, dtype=self.dtype)
            x = norm()(x)
        if self.activation_layer:
            x = self.activation_layer(x)
        x = jnp.asarray(x, dtype=self.dtype)
        return x
