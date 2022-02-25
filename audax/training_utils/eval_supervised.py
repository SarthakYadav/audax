"""
Helper functions for evaluating a given supervised "Classifier" model

Written for audax by / Copyright 2022, Sarthak Yadav
"""
import json
import functools
import os
import time
from typing import Any
import tqdm
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
from numpy import var
import numpy as np
import optax
import tensorflow as tf
from sklearn.metrics import average_precision_score, accuracy_score
from ..transforms import mixup
from . import metrics_helper
from jax.config import config


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
from .. import models
from audax import frontends
from .train_supervised import create_model


def forward(state, batch):
    variables = {
        'params': state.get_all_params,                    # absolutely ok to just use state.get_all_params here
        'batch_stats': state.batch_stats
    }
    logits = state.apply_fn(
        variables, batch['audio'], train=False, mutable=False)
    return logits


def load_variables_from_checkpoint(workdir, prefix):
    pretrained_variables = checkpoints.restore_checkpoint(workdir, None, prefix=prefix)
    variables = {
        "params": pretrained_variables['params'],
        "batch_stats": pretrained_variables['batch_stats']
    }
    return variables, pretrained_variables['aux_rng_keys']


def evaluate(workdir: str,
             eval_signal_duration="AUTO",
             eval_manifest_override=None,
             eval_steps_override=None):
    config = training_utilities.read_config_from_json(workdir)
    if config.batch_size % jax.device_count() > 0:
        raise ValueError('Batch size must be divisible by the number of devices')
    # doing one sample at a time. Better supports VoxCeleb and other variable length sequences
    local_batch_size = 1 // jax.process_count()
    logging.info("Process count: {}".format(jax.process_count()))
    # eval is done on single device
    device = config.get("device", 1)
    if device:
        devices = [jax.local_devices()[device]]
    else:
        devices = jax.local_devices()
    platform = devices[0].platform
    if config.half_precision:
        if platform == 'tpu':
            input_dtype = tf.bfloat16
        else:
            input_dtype = tf.float16
    else:
        input_dtype = tf.float32
    mode = TrainingMode(config.model.type)
    if eval_signal_duration == "AUTO":
        if config.data.dataset_name == "audioset":
            config.audio_config.min_duration = 10.
        elif config.data.dataset_name == "speechcommandsv2":
            config.audio_config.min_duration = 1.
        elif config.data.dataset_name == "voxceleb1":
            config.audio_config.min_duration = 10.
        else:
            raise ValueError(f"Unsupported dataset {config.data.dataset_name} for eval_signal_duration == 'AUTO'")
    elif type(eval_signal_duration) == float and eval_signal_duration >= 1.0:
        config.audio_config.min_duration = eval_signal_duration
    else:
        raise ValueError(f"Unsupported dataset {config.data.dataset_name} for eval_signal_duration == 'AUTO'")
    if eval_manifest_override is not None:
        assert os.path.exists(eval_manifest_override), f"{eval_manifest_override} doesn't exist"
        logging.info("Overriding eval_manifest path {} in config file with {}".format(
            config.data.eval_manifest, eval_manifest_override
        ))
        if eval_steps_override is None or eval_steps_override == 0:
            raise ValueError(f"Incorrect value for eval_steps_override: {eval_steps_override}")
        config.data.eval_manifest = eval_manifest_override
        config.data.eval_samples = eval_steps_override

    rng = random.PRNGKey(0)
    _, eval_iter = prepare_datasets_v2(config, local_batch_size, input_dtype=input_dtype)
    eval_iter = training_utilities.create_input_iter(eval_iter, devices=devices)

    if config.data.jax_transforms:
        tfs = training_utilities.get_feature_functions(config)
        if len(tfs) != 0:
            p_feature_extract_fn = jax.pmap(
                functools.partial(
                    training_utilities.apply_audio_transforms, transforms=tfs, 
                    dtype=training_utilities.get_dtype(config.half_precision),
                ), axis_name='batch', devices=devices)
        else:
            p_feature_extract_fn = None
    else:
        p_feature_extract_fn = None
    model_cls, frontend_cls = training_utilities.get_model_frontend_cls(config)
    model = create_model(
        model_cls=model_cls, half_precision=config.half_precision,
        frontend_cls=frontend_cls,
        num_classes=config.model.num_classes,
        spec_aug=None,
        drop_rate=config.model.get("fc_drop_rate", 0.))
    
    # placeholder to just load the thing
    learning_rate_fn = training_utilities.create_learning_rate_fn(
        config, 0.1, 100)
    state = training_utilities.create_train_state(rng, config, model, learning_rate_fn)
    # state = training_utilities.restore_checkpoint(state, workdir)
    state = checkpoints.restore_checkpoint(workdir, state, prefix="best_")
    state = jax_utils.replicate(state, devices=devices)
    p_forward = jax.pmap(functools.partial(forward),
                           axis_name='batch', devices=devices)
    if config.steps_per_eval == -1:
        num_validation_examples = config.data.eval_samples
        steps_per_eval = num_validation_examples // 1
    else:
        steps_per_eval = config.steps_per_eval
    eval_logits = []
    eval_labels = []
    for _ in tqdm.tqdm(range(steps_per_eval)):
        eval_batch = next(eval_iter)
        if p_feature_extract_fn:
            eval_batch['audio'] = p_feature_extract_fn(eval_batch['audio'])
        # print(eval_batch['audio'].shape)
        logits = p_forward(state, eval_batch)
        # print(logits.shape)
        eval_logits.append(logits)
        eval_labels.append(eval_batch['label'])
    logging.info("Concatenating predictions and labels..")
    eval_logits = jnp.concatenate([jax.device_get(x) for x in eval_logits])
    eval_labels = jnp.concatenate([jax.device_get(x) for x in eval_labels])
    eval_logits = eval_logits.reshape(-1, eval_logits.shape[-1]).astype('float32')
    eval_labels = eval_labels.reshape(-1, eval_labels.shape[-1]).astype("float32")
    fp = open(os.path.join(workdir, "results.txt"), "w")
    if mode == TrainingMode.MULTILABEL:
        # macro_mAP = metrics_helper.calculate_mAP(eval_logits, eval_labels)
        stats = metrics_helper.calculate_stats(eval_logits, eval_labels)
        mAP = np.mean([stat['AP'] for stat in stats])
        mAUC = np.mean([stat['auc'] for stat in stats])
        dp = metrics_helper.d_prime(mAUC)
        s = "mAP: {:.5f}\n".format(mAP)
        s += "mAUC: {:.5f}\n".format(mAUC)

        s += "dprime: {:.5f}".format(dp)
    elif mode == TrainingMode.MULTICLASS:
        acc = accuracy_score(y_true=np.argmax(np.asarray(eval_labels), axis=1), 
                             y_pred=np.argmax(np.asarray(eval_logits), axis=1))
        s = "Accuracy: {:.4f}".format(acc)
    print(s)
    fp.write(s)
    fp.close()
