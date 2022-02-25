"""
Implementation for the paper

Park, D.S., Chan, W., Zhang, Y., Chiu, C.C., Zoph, B., Cubuk, E.D. and Le, Q.V., 2019.
"Specaugment: A simple data augmentation method for automatic speech recognition."
arXiv preprint arXiv:1904.08779.

Written for audax by / Copyright 2022, Sarthak Yadav
"""
from typing import Any
import jax.numpy as jnp
from jax import random, jit
import flax.linen as nn


class SpecAugment:
    """
    SpecAugment, without time-stretching
    """
    def __init__(self, freq_param: int, time_param: int, num_masks: int = 2):
        self.freq_param = freq_param
        self.time_param = time_param
        self.num_masks = num_masks

    def augment(self, inputs, rng):
        # if rng is None:
        #     rng = self.make_rng('spec_augment')
        rng, *keys = random.split(rng, 3)

        def mask_along_axis_iid(key_index, inputs: jnp.array, mask_param, axis, mask_value=0.):
            """
            :param inputs: Input tensor of shape (batch, time, freq, 1)
            :param mask_param:
            :param mask_value:
            :param axis:
            :return:
            """
            # if axis not in [1, 2]:
            #     raise ValueError('Only Frequency and Time masking are supported')

            dtype = inputs.dtype

            shape_ = (inputs.shape[0], inputs.shape[-1])

            value = random.uniform(keys[key_index], shape_) * mask_param
            min_value = random.uniform(keys[key_index], shape_) * (inputs.shape[axis] - value)
            mask_start = min_value.reshape(-1, 1, 1, 1)
            mask_end = (min_value + value).reshape(-1, 1, 1, 1)
            mask = jnp.arange(0, inputs.shape[axis])
            if axis == 1:
                inputs = inputs.transpose(0, 2, 3, 1)
            else:
                inputs = inputs.transpose(0, 1, 3, 2)
            o = (mask >= mask_start) & (mask < mask_end)
            o = jnp.repeat(o, inputs.shape[1], axis=1)
            inputs = jnp.where(o, 0, inputs)# inputs.at[o].set(mask_value)

            if axis == 1:
                inputs = inputs.transpose(0, 3, 1, 2)
            else:
                inputs = inputs.transpose(0, 1, 3, 2)
            return inputs

        outputs = inputs
        if self.freq_param != 0:
            outputs = mask_along_axis_iid(0, outputs, self.freq_param, 2)
        if self.time_param != 0:
            outputs = mask_along_axis_iid(1, outputs, self.time_param, 1)
        return outputs, rng

    def __call__(self, inputs, rng):
        outputs = inputs
        for _ in range(self.num_masks):
            outputs, rng = self.augment(outputs, rng)
        return outputs, rng
