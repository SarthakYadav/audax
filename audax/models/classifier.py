"""
General classifier wrapper for audax models

Written for audax by / Copyright 2022, Sarthak Yadav
"""
import jax.numpy as jnp
from jax import lax, random
import flax.linen as nn
from typing import Callable, Any, Optional


Dtype = Any


class Classifier(nn.Module):
    model_cls: Callable
    num_classes: int
    frontend_cls: Optional[Callable] = None
    drop_rate: float = 0.0
    spec_aug: Optional[Callable] = None
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs, train: bool = True):
        outputs = inputs
        if self.frontend_cls is not None:
            outputs = self.frontend_cls(dtype=jnp.float32, name="frontend")(outputs)
            outputs = outputs[Ellipsis, jnp.newaxis]
            outputs = outputs.astype(self.dtype)

            if self.spec_aug is not None and train:
                print("Doing spec_aug")
                spec_aug_rng = self.make_rng('spec_aug')
                outputs, _ = self.spec_aug(outputs, spec_aug_rng)

        outputs = self.model_cls(num_classes=None, dtype=self.dtype, name="encoder")(outputs, train=train)
        if self.drop_rate != 0:
            outputs = nn.Dropout(self.drop_rate, deterministic=not train, name="fc_dropout")(outputs)
        outputs = nn.Dense(self.num_classes, name="fc")(outputs)
        return outputs
