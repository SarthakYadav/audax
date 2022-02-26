""" ConvNeXt
Paper: `A ConvNet for the 2020s` - https://arxiv.org/pdf/2201.03545.pdf

Adapted from the original PyTorch code (https://github.com/facebookresearch/ConvNeXt),
and Ross Wightman's implementation for timm (https://github.com/rwightman/pytorch-image-models)

Modifications and additions for audax by / Copyright 2022, Sarthak Yadav
"""
from functools import partial
import jax.numpy as jnp
from jax import lax, random
from flax import linen as nn
from .layers.drop import DropPath
from .layers.mlp import Mlp
from jax import dtypes
from typing import Optional, Tuple, Union, Callable, Any


def constant_init(key, shape, dtype=jnp.float_, constant=0.04):
    return jnp.ones(shape, dtypes.canonicalize_dtype(dtype)) * constant


def convnext_truncated_init(key, shape, dtype=jnp.float_, std=0.02):
    return random.truncated_normal(key, -2, 2, shape, dtype) * std


default_init = partial(convnext_truncated_init, std=0.02)


class ConvNeXtBlock(nn.Module):
    dim: int
    mlp_ratio: int = 4
    ls_init_value: float = 1e-6
    norm_layer: Callable = None
    dtype = jnp.float32
    drop_path: float = 0.

    @nn.compact
    def __call__(self, x, train: bool = True):
        shortcut = x
        x = nn.Conv(self.dim, kernel_size=(7, 7), padding="SAME",
                    feature_group_count=self.dim, dtype=self.dtype, kernel_init=default_init)(x)
        if self.norm_layer:
            x = self.norm_layer(dtype=self.dtype)(x)
        else:
            x = nn.LayerNorm(dtype=self.dtype)(x)
        x = Mlp(hidden_features=int(self.mlp_ratio*self.dim), out_features=self.dim,
                activation=nn.gelu, kernel_init=default_init)(x)

        if self.ls_init_value:
            gamma = self.param("gamma", partial(constant_init, constant=self.ls_init_value), [self.dim])
        else:
            gamma = None

        if gamma is not None:
            x = x * gamma.reshape((1,) * (x.ndim-1) + (-1,))
        
        if self.drop_path != 0.:
            x = DropPath(self.drop_path)(x, train=train) + shortcut
        else:
            x = x + shortcut
        return x


class ConvNeXtStage(nn.Module):
    out_channels: int
    stride: int = 2
    depth: int = 2
    dp_rates: Optional[float] = None
    ls_init_value: float = 1.0
    norm_layer: Callable = None
    cross_stage: Optional[float] = False
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, train: bool = True):
        in_chs = x.shape[-1]
        output = x
        if self.norm_layer is None:
            norm_layer = partial(nn.LayerNorm, dtype=self.dtype)
        else:
            norm_layer = partial(self.norm_layer, dtype=self.dtype)

        if in_chs != self.out_channels:
            output = norm_layer()(output)
            output = nn.Conv(self.out_channels, kernel_size=(self.stride, self.stride),
                             strides=(self.stride, self.stride), padding="VALID",
                             dtype=self.dtype, kernel_init=default_init)(output)

        dp_rates = self.dp_rates or [0.] * self.depth
        for j in range(self.depth):
            output = ConvNeXtBlock(dim=self.out_channels, drop_path=dp_rates[j],
                                   ls_init_value=self.ls_init_value,
                                   norm_layer=norm_layer)(output, train=train)
        return output


class ConvNeXt(nn.Module):
    r"""
    ConvNeXt
    A Jax impl of : `A ConvNet for the 2020s`  - https://arxiv.org/pdf/2201.03545.pdf

    Args:
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (tuple(int)): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_rate (float): Head dropout rate
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    num_classes: Optional[int] = 1000
    global_pool: str = "avg"
    depths: Optional[Tuple] = (3, 3, 9, 3)
    dims: Optional[Tuple] = (96, 192, 384, 768)
    output_stride: int = 32
    patch_size: int = 4
    ls_init_value: float = 1e-6
    head_init_scale: float = 1.
    norm_layer: Optional[Union[Callable, nn.Module]] = None
    drop_rate: float = 0.
    drop_path_rate: float = 0.
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, inputs, train: bool = True):
        assert self.output_stride == 32
        if self.norm_layer is None:
            norm_layer = partial(nn.LayerNorm, dtype=self.dtype)
        else:
            norm_layer = partial(self.norm_layer, dtype=self.dtype)

        # STEM
        outputs = nn.Conv(self.dims[0], kernel_size=(self.patch_size, self.patch_size),
                          strides=(self.patch_size, self.patch_size),
                          padding="VALID", dtype=self.dtype, kernel_init=default_init)(inputs)
        outputs = norm_layer()(outputs)
        # END STEM

        curr_stride = self.patch_size

        for i in range(4):
            stride = 2 if i > 0 else 1
            curr_stride *= stride
            out_chs = self.dims[i]
            outputs = ConvNeXtStage(
                out_channels=out_chs, stride=stride, depth=self.depths[i],
                ls_init_value=self.ls_init_value, norm_layer=norm_layer, dtype=self.dtype)(outputs, train=train)

        if self.global_pool == "avg":
            outputs = jnp.mean(outputs, axis=(1, 2))
        elif self.global_pool == "max":
            outputs = jnp.max(outputs, axis=(1, 2))
        else:
            raise ValueError("Unsupported pool type value : {}".format(self.global_pool))

        if self.num_classes is not None:
            outputs = norm_layer(name="fc_norm")(outputs)
            if self.drop_rate != 0.:
                outputs = nn.Dropout(self.drop_rate, deterministic=not train, name="fc_dropout")(outputs)
            outputs = nn.Dense(self.num_classes, name="fc", kernel_init=default_init)(outputs)

        return outputs


# drop_path_rate corresponds to Stochastic Depth parameter
# for ConvNeXT-T/S/B/L, Stochastic Depths corresponding to ImageNet-1k training are 0.1/0.4/0.5/0.5 resp.
# We don't use them by default

convnext_tiny = partial(ConvNeXt, depths=(3, 3, 9, 3), dims=(96, 192, 384, 768), drop_path_rate=0.1)
convnext_small = partial(ConvNeXt, depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], drop_path_rate=0.4)
convnext_base = partial(ConvNeXt, depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])
convnext_large = partial(ConvNeXt, depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536])
convnext_xlarge = partial(ConvNeXt, depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048])
