"""
An implementation of the EfficientNet CNN architecture in jax

Based on the simplified implementation of EfficientNet as found in torchvision (https://pytorch.org/vision/main/_modules/torchvision/models/efficientnet.html)

Written/modified for audax by / Copyright 2022, Sarthak Yadav
"""
import copy
import jax
import math
from typing import Callable, Optional, Any, List, Sequence
from jax import numpy as jnp
from flax import linen as nn
from functools import partial
from .utils import _make_divisible
from .layers.efficientnet_helpers import SqueezeAndExcitation, ConvNormActivation
from .layers.drop import DropPath


# __all__ = [
#     "EfficientNet",
#     "efficientnet_b0",
#     "efficientnet_b1",
#     "efficientnet_b2",
#     "efficientnet_b3",
#     "efficientnet_b4",
#     "efficientnet_b5",
#     "efficientnet_b6",
#     "efficientnet_b7",
# ]


class MBConvConfig:
    # Stores information listed at Table 1 of the EfficientNet paper
    def __init__(
            self,
            expand_ratio: float,
            kernel: int,
            stride: int,
            input_channels: int,
            out_channels: int,
            num_layers: int,
            width_mult: float,
            depth_mult: float,
    ) -> None:
        self.expand_ratio = expand_ratio
        self.kernel = kernel
        self.stride = stride
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.num_layers = self.adjust_depth(num_layers, depth_mult)

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "expand_ratio={expand_ratio}"
        s += ", kernel={kernel}"
        s += ", stride={stride}"
        s += ", input_channels={input_channels}"
        s += ", out_channels={out_channels}"
        s += ", num_layers={num_layers}"
        s += ")"
        return s.format(**self.__dict__)

    @staticmethod
    def adjust_channels(channels: int, width_mult: float, min_value: Optional[int] = None) -> int:
        return _make_divisible(channels * width_mult, 8, min_value)

    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float):
        return int(math.ceil(num_layers * depth_mult))


class MBConv(nn.Module):
    cnf: MBConvConfig
    drop_path_prob: float
    norm_layer: Callable[..., nn.Module]
    se_layer: Callable[..., nn.Module] = SqueezeAndExcitation
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, inputs, train: bool = True):
        if not (1 <= self.cnf.stride <= 2):
            raise ValueError("wrong stride value")

        input_channels = inputs.shape[-1]
        use_res_connect = self.cnf.stride == 1 and input_channels == self.cnf.out_channels
        activation_layer = nn.silu

        expanded_channels = self.cnf.adjust_channels(input_channels, self.cnf.expand_ratio)
        outputs = inputs
        if expanded_channels != input_channels:
            outputs = ConvNormActivation(
                expanded_channels,
                kernel_size=1,
                norm_layer=self.norm_layer,
                activation_layer=activation_layer
            )(outputs, train=train)

        # depthwise
        outputs = ConvNormActivation(
            expanded_channels,
            kernel_size=self.cnf.kernel,
            stride=self.cnf.stride,
            groups=expanded_channels,
            norm_layer=self.norm_layer,
            activation_layer=activation_layer,
        )(outputs, train=train)

        # squeeze and excitation
        squeeze_channels = max(1, input_channels // 4)
        outputs = self.se_layer(expanded_channels, squeeze_channels, activation=activation_layer)(outputs)

        # project
        outputs = ConvNormActivation(
            self.cnf.out_channels, kernel_size=1,
            norm_layer=self.norm_layer, activation_layer=None
        )(outputs, train=train)

        if use_res_connect:
            if self.drop_path_prob != 0.:
                outputs = DropPath(self.drop_path_prob)(outputs, train=train)
            outputs = outputs + inputs
        return outputs


class EfficientNet(nn.Module):
    inverted_residual_setting: List[MBConvConfig]
    drop_rate: float
    drop_path_prob: float = 0.2
    num_classes: Optional[int] = 1000
    block: Optional[Callable[..., nn.Module]] = None
    norm_layer: Optional[Callable[..., nn.Module]] = None
    global_pool: str = "avg"
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, inputs, train: bool = True):
        if not self.inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
                isinstance(self.inverted_residual_setting, Sequence)
                and all([isinstance(s, MBConvConfig) for s in self.inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[MBConvConfig]")

        if self.block is None:
            block = MBConv
        else:
            block = self.block

        if self.norm_layer is None:
            norm_layer = nn.BatchNorm
        else:
            norm_layer = self.norm_layer

        activation_layer = nn.silu

        firstconv_output_channels = self.inverted_residual_setting[0].input_channels
        outputs = ConvNormActivation(firstconv_output_channels, kernel_size=3, stride=2,
                                     norm_layer=norm_layer, activation_layer=activation_layer)(inputs, train=train)

        total_stage_blocks = sum(cnf.num_layers for cnf in self.inverted_residual_setting)
        stage_block_id = 0

        for cnf in self.inverted_residual_setting:
            for ix in range(cnf.num_layers):

                block_cnf = copy.copy(cnf)

                if ix > 0:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                sd_prob = self.drop_path_prob * float(stage_block_id) / total_stage_blocks

                outputs = block(block_cnf, norm_layer=norm_layer, drop_path_prob=sd_prob, dtype=self.dtype)(outputs,
                                                                                                            train=train)
                stage_block_id += 1

        lastconv_output_channels = 4 * self.inverted_residual_setting[-1].out_channels

        outputs = ConvNormActivation(lastconv_output_channels, kernel_size=1, norm_layer=norm_layer,
                                     activation_layer=activation_layer, dtype=self.dtype)(outputs, train=train)

        if self.global_pool == "avg":
            outputs = jnp.mean(outputs, axis=(1, 2))
        elif self.global_pool == "max":
            outputs = jnp.max(outputs, axis=(1, 2))
        else:
            raise ValueError("Unsupported pool type value : {}".format(self.global_pool))

        if self.num_classes is not None:
            if self.drop_rate != 0.:
                outputs = nn.Dropout(self.drop_rate, deterministic=not train, name="fc_dropout")(outputs)
            outputs = nn.Dense(self.num_classes, name="fc")(outputs)

        return outputs


def _efficientnet(
        arch: str,
        width_mult: float,
        depth_mult: float,
        dropout: float,
        **kwargs: Any,
) -> EfficientNet:
    bneck_conf = partial(MBConvConfig, width_mult=width_mult, depth_mult=depth_mult)
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 32, 16, 1),
        bneck_conf(6, 3, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 2, 40, 80, 3),
        bneck_conf(6, 5, 1, 80, 112, 3),
        bneck_conf(6, 5, 2, 112, 192, 4),
        bneck_conf(6, 3, 1, 192, 320, 1),
    ]
    model = EfficientNet(inverted_residual_setting, dropout, **kwargs)
    return model


def efficientnet_b0(**kwargs: Any):
    return _efficientnet("efficientnet_b0", 1.0, 1.0, 0.2, **kwargs)


def efficientnet_b1(**kwargs: Any) -> EfficientNet:
    return _efficientnet("efficientnet_b1", 1.0, 1.1, 0.2, **kwargs)


def efficientnet_b2(**kwargs: Any) -> EfficientNet:
    return _efficientnet("efficientnet_b2", 1.1, 1.2, 0.3, **kwargs)


def efficientnet_b3(**kwargs: Any) -> EfficientNet:
    return _efficientnet("efficientnet_b3", 1.2, 1.4, 0.3, **kwargs)


def efficientnet_b4(**kwargs: Any) -> EfficientNet:
    return _efficientnet("efficientnet_b4", 1.4, 1.8, 0.4, **kwargs)


def efficientnet_b5(**kwargs: Any) -> EfficientNet:
    return _efficientnet(
        "efficientnet_b5",
        1.6,
        2.2,
        0.4,
        norm_layer=partial(nn.BatchNorm, epsilon=0.001, momentum=0.01),
        **kwargs,
    )


def efficientnet_b6(**kwargs: Any) -> EfficientNet:
    return _efficientnet(
        "efficientnet_b6",
        1.8,
        2.6,
        0.5,
        norm_layer=partial(nn.BatchNorm, epsilon=0.001, momentum=0.01),
        **kwargs,
    )


def efficientnet_b7(**kwargs: Any) -> EfficientNet:
    return _efficientnet(
        "efficientnet_b7",
        2.0,
        3.1,
        0.5,
        norm_layer=partial(nn.BatchNorm, epsilon=0.001, momentum=0.01),
        **kwargs,
    )
