"""
Contrastive training utilities

Should work out of the box with any contrastive ssl losses that have anchor-positive pairs.

However, primarily designed to work with SimCLR [1, 2] & COLA [3], so some modifications might be necessary
for other use cases

References
----------

[1] Chen, T., Kornblith, S., Norouzi, M. and Hinton, G., 2020, November.
    "A simple framework for contrastive learning of visual representations".
    In International conference on machine learning (pp. 1597-1607). PMLR.


[2] Chen, T., Kornblith, S., Swersky, K., Norouzi, M. and Hinton, G., 2020.
    "Big self-supervised models are strong semi-supervised learners".
    arXiv preprint arXiv:2006.10029.

[3] Saeed, A., Grangier, D. and Zeghidour, N., 2021, June.
    "Contrastive learning of general-purpose audio representations".
    In ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)
    (pp. 3875-3879). IEEE.

Written for audax by / Copyright 2022, Sarthak Yadav
"""
import functools
import time
from typing import Any

from absl import logging
from clu import metric_writers
from clu import periodic_actions
import flax
from flax import jax_utils
from flax import optim
from flax.training import checkpoints
from flax.training import common_utils
from flax.training import train_state
import jax
from jax import lax
import jax.numpy as jnp
from jax import random
import ml_collections
import optax
import tensorflow as tf
from sklearn.metrics import average_precision_score

try:
    # Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

from .misc import TrainingMode, Features, DataSplit
from . import training_utilities
from .data_v2.helpers import prepare_datasets_v2

# import encoders
# from cola import COLA, SimilarityMeasure
import wandb


# def create_COLA_model(*, encoder_cls, half_precision, similarity_measure, embedding_dim,
#                       temperature=0.1):
#     platform = jax.local_devices()[0].platform
#     if half_precision:
#         if platform == 'tpu':
#             model_dtype = jnp.bfloat16
#         else:
#             model_dtype = jnp.float16
#     else:
#         model_dtype = jnp.float32
#     return COLA(model_cls=encoder_cls, similarity_measure=similarity_measure,
#                 embedding_dim=embedding_dim, temperature=temperature, dtype=model_dtype)


def train_step(state, batch, learning_rate_fn,
               cost_func,
               mode=TrainingMode.MULTICLASS):
    """Perform a single training step."""

    def loss_fn(params):
        anchors = batch['anchor']
        positives = batch['positive']
        labels = batch['label']
        logits, new_model_state = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            jnp.concatenate([anchors, positives]),
            mutable=['batch_stats'],
            rngs=state.aux_rng_keys
        )
        labels = common_utils.onehot(jnp.arange(0, logits.shape[0]), num_classes=logits.shape[0])
        loss = cost_func(logits, labels)
        return loss, (new_model_state, logits, labels)

    step = state.step
    dynamic_scale = state.dynamic_scale
    lr = learning_rate_fn(step)

    if dynamic_scale:
        grad_fn = dynamic_scale.value_and_grad(
            loss_fn, has_aux=True, axis_name='batch')
        dynamic_scale, is_fin, aux, grads = grad_fn(state.params)
        # dynamic loss takes care of averaging gradients across replicas
    else:
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        aux, grads = grad_fn(state.params)
        # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
        grads = lax.pmean(grads, axis_name='batch')

    new_model_state, logits, labels = aux[1]
    metrics = training_utilities.compute_metrics(logits, labels, mode=mode, cost_fn=cost_func)
    metrics['learning_rate'] = lr
    new_state = state.apply_gradients(
        grads=grads, batch_stats=new_model_state['batch_stats'])
    if dynamic_scale:
        new_state = new_state.replace(
            opt_state=jax.tree_multimap(
                functools.partial(jnp.where, is_fin),
                new_state.opt_state,
                state.opt_state),
            params=jax.tree_multimap(
                functools.partial(jnp.where, is_fin),
                new_state.params,
                state.params))
        metrics['scale'] = dynamic_scale.scale
    return new_state, metrics


def eval_step(state, batch, cost_func, mode=TrainingMode.MULTICLASS):
    variables = {'params': state.params, 'batch_stats': state.batch_stats}
    anchors = batch['anchor']
    positives = batch['positive']
    logits = state.apply_fn(
        variables, jnp.concatenate([anchors, positives]), train=False, mutable=False)
    labels = batch['label']
    labels = common_utils.onehot(jnp.arange(0, logits.shape[0]), num_classes=logits.shape[0])
    metrics = training_utilities.compute_metrics(logits, labels, mode=mode, cost_fn=cost_func)
    return metrics, logits, labels
