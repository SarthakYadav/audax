"""
Implementation of DropPath (Stochastic Depth) regularization

Inspired by the PyTorch implementation in timm (https://github.com/rwightman/pytorch-image-models)
by Ross Wightman, 2022

Modifications and additions for audax by / Copyright 2022, Sarthak Yadav
"""

from flax import linen as nn
from jax import lax, numpy as jnp, random
from typing import Any


def drop_path(x: jnp.array, rng, drop_prob: float = 0.) -> jnp.array:
    if drop_prob == 0.:
        return x
    keep_prob = 1 - drop_prob
    mask = random.bernoulli(key=rng, p=keep_prob, shape=(x.shape[0],) + (1,)*(x.ndim-1))
    mask = jnp.broadcast_to(mask, x.shape)
    return lax.select(mask, x / keep_prob, jnp.zeros_like(x))


class DropPath(nn.Module):
    rate: float = 0.

    @nn.compact
    def __call__(self, x, train: bool = False, rng: Any = None):
        if not train or self.rate == 0.:
            return x
        if rng is None:
            rng = self.make_rng('drop_path')
        return drop_path(x, rng, drop_prob=self.rate)
