"""
Similarity layers

Written/modified for audax by / Copyright 2022, Sarthak Yadav
"""
from jax import numpy as jnp, lax
from flax import linen as nn
from typing import Callable, Any, Iterable, Optional
from ..utils import l2_normalize

PRNGKey = Any
Shape = Iterable[int]
Dtype = Any  # this could be a real type?
Array = Any


class DotProduct(nn.Module):
    precision: lax.Precision = lax.Precision.HIGH

    @nn.compact
    def __call__(self, anchor, positive):
        anchor = l2_normalize(anchor, axis=-1)
        positive = l2_normalize(positive, axis=-1)
        return jnp.matmul(anchor, positive.T, precision=self.precision)


class BilinearProduct(nn.Module):
    dim: int
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, anchor, positive):
        w = self.param('bilinear_product_weight', nn.initializers.normal(), [self.dim, self.dim], self.dtype)
        projection_positive = jnp.matmul(w, positive.T)
        return jnp.matmul(anchor, projection_positive)
