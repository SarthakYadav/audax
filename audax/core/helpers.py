"""
some helpers utility functions

_conv_dimension_numbers is taken from flax (https://flax.readthedocs.io/en/latest/_modules/flax/linen/linear.html)

Written/hacked together for audax by / Copyright 2022, Sarthak Yadav
"""
import jax
from functools import partial
from jax import numpy as jnp, lax


def batch_pad(xs, pad: int, mode="reflect"):
    if xs.ndim == 2:
        pad_arg = [(0, 0), (pad, pad)]
    elif xs.ndim == 3:
        pad_arg = [(0, 0), (pad, pad), (0, 0)]
    else:
        pad_arg = [(pad, pad)]
    xs = jnp.pad(xs, pad_arg, mode=mode)
    return xs


def _conv_dimension_numbers(input_shape):
    """Computes the dimension numbers based on the input shape."""
    ndim = len(input_shape)
    lhs_spec = (0, ndim - 1) + tuple(range(1, ndim - 1))
    rhs_spec = (ndim - 1, ndim - 2) + tuple(range(0, ndim - 2))
    out_spec = lhs_spec
    return lax.ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)
