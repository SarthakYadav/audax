"""
Supervised training boilerplate code used for training a audio classifier in the provided recipes

Written for audax by / Copyright 2022, Sarthak Yadav
"""
import functools
import os
import time
from typing import Any
import copy
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
from ..transforms import mixup


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
# from encoders.classifier import Classifier
import wandb
from audax import frontends


def create_model(*, model_cls, half_precision, num_classes, 
                 drop_rate=None, spec_aug=None, **kwargs):
    platform = jax.local_devices()[0].platform
    if half_precision:
        if platform == 'tpu':
            model_dtype = jnp.bfloat16
        else:
            model_dtype = jnp.float16
    else:
        model_dtype = jnp.float32
    return models.Classifier(model_cls=model_cls, num_classes=num_classes,
                             dtype=model_dtype, drop_rate=drop_rate, spec_aug=spec_aug, 
                             **kwargs)


def train_step(state, batch, learning_rate_fn,
               cost_func,
               mode=TrainingMode.MULTICLASS,
               mixup_func=None,
               mixup_criterion_func=None):
    """Perform a single training step."""
    inputs = batch['audio']
    targets = batch["label"]
    if mixup_func is not None:
        inputs, y_a, y_b, lam = mixup_func(state.aux_rng_keys["mixup"], inputs, targets)

    def loss_fn(params):
        logits, new_model_state = state.apply_fn(
            {
                'params': {**params, **state.frozen_params},        # we don't use state.get_all_params here
                'batch_stats': state.batch_stats                    # since we'll override gradient computation
            },
            inputs,
            mutable=['batch_stats'],
            rngs=state.aux_rng_keys
        )
        if mixup_func is not None:
            loss = mixup_criterion_func(cost_func, logits, y_a, y_b, lam)
        else:
            loss = cost_func(logits, batch['label'])
        return loss, (new_model_state, logits)

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

    new_model_state, logits = aux[1]
    metrics = training_utilities.compute_metrics(logits, batch['label'], mode=mode, cost_fn=cost_func)
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
    variables = {
        'params': state.get_all_params,                    # absolutely ok to just use state.get_all_params here
        'batch_stats': state.batch_stats
    }
    logits = state.apply_fn(
        variables, batch['audio'], train=False, mutable=False)
    metrics = training_utilities.compute_metrics(logits, batch['label'], mode=mode, cost_fn=cost_func)
    return metrics, logits, batch['label']


def train_and_evaluate(config: ml_collections.ConfigDict,
                       workdir: str,
                       no_wandb: bool,
                       seed: int = 0):
    """Execute model training and evaluation loop.

      Args:
        config: Hyperparameter configuration for training and evaluation.
        workdir: Directory where the tensorboard summaries are written to.

      Returns:
        Final TrainState.
      """
    wandb_logger = None
    if not no_wandb:
        wandb_logger = wandb.init(project='{}'.format(config.wandb.get("project", "audax-supervised")),
                                  group="{}".format(config.data.dataset_name),
                                  config=config.to_dict(), name=workdir.split("/")[-1])
    writer = metric_writers.create_default_writer(
        logdir=workdir, just_logging=jax.process_index() != 0)
    # write config to workdir
    training_utilities.write_config_to_json(workdir, config)
    rng = random.PRNGKey(seed)
    if config.batch_size % jax.device_count() > 0:
        raise ValueError('Batch size must be divisible by the number of devices')
    local_batch_size = config.batch_size // jax.process_count()
    logging.info("Process count: {}".format(jax.process_count()))
    device = config.get("device", None)
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
    logging.info('Training mode is {}...'.format(mode))
    train_iter, eval_iter = prepare_datasets_v2(config, local_batch_size, input_dtype=input_dtype)
    train_iter = training_utilities.create_input_iter(train_iter, devices=devices)
    eval_iter = training_utilities.create_input_iter(eval_iter, devices=devices)

    if config.data.jax_transforms:
        tfs = training_utilities.get_feature_functions(config)
        if len(tfs) != 0:
            p_feature_extract_fn = jax.pmap(
                functools.partial(
                    training_utilities.apply_audio_transforms, transforms=tfs, 
                    dtype=training_utilities.get_dtype(config.half_precision),
                    normalize=config.data.get("normalize_batch", False),
                    add_new_axis=config.model.get("is_2d", True)
                ), axis_name='batch', devices=devices)
        else:
            p_feature_extract_fn = None
    else:
        p_feature_extract_fn = None

    in_model_spec_aug = config.get("in_model_spec_aug", False)
    p_spec_aug, spec_aug_rng = training_utilities.get_spec_augment(config, devices, pmapped=not in_model_spec_aug)
    if p_spec_aug:
        logging.info("Using spec augment")
    num_examples = config.data.tr_samples
    if config.get("steps_per_epoch", -1) == -1:
        steps_per_epoch = (num_examples // config.batch_size)
    else:
        steps_per_epoch = config.get("steps_per_epoch")

    if config.num_train_steps == -1:
        num_steps = int(steps_per_epoch * config.num_epochs)
        num_epochs = config.num_epochs
    else:
        num_steps = config.num_train_steps
        num_epochs = config.num_train_steps // steps_per_epoch
    logging.info("num_steps: {} | num_epochs: {}".format(num_steps, num_epochs))
    if config.steps_per_eval == -1:
        num_validation_examples = config.data.eval_samples
        steps_per_eval = num_validation_examples // config.batch_size
    else:
        steps_per_eval = config.steps_per_eval

    steps_per_checkpoint = steps_per_epoch
    base_learning_rate = config.opt.learning_rate * config.batch_size / 256.

    model_cls, frontend_cls = training_utilities.get_model_frontend_cls(config)
    model = create_model(
        model_cls=model_cls, half_precision=config.half_precision,
        frontend_cls=frontend_cls,
        num_classes=config.model.num_classes,
        spec_aug=p_spec_aug if in_model_spec_aug else None,
        drop_rate=config.model.get("fc_drop_rate", 0.))
    print(model)
    learning_rate_fn = training_utilities.create_learning_rate_fn(
        config, base_learning_rate, steps_per_epoch, num_epochs)

    if config.model.get("pretrained", None):
        state = training_utilities.create_train_state_from_pretrained(rng, config,
                                                                      model, learning_rate_fn,
                                                                      config.model.pretrained,
                                                                      config.model.get("pretrained_prefix",
                                                                                       "checkpoint_"),
                                                                      to_copy=['encoder'],
                                                                      fc_only=config.model.get("pretrained_fc_only",
                                                                                               False))
    else:
        state = training_utilities.create_train_state(rng, config, model, learning_rate_fn)
    state = training_utilities.restore_checkpoint(state, workdir)

    step_offset = int(state.step)
    state = jax_utils.replicate(state, devices=devices)

    label_smoothing_factor = config.opt.get("label_smoothing_factor", None)
    if label_smoothing_factor:
        logging.info("Training with Label Smoothing, alpha = {}".format(label_smoothing_factor))

    # setup funcs!
    if mode == TrainingMode.MULTICLASS:
        cost_fn = functools.partial(training_utilities.cross_entropy_loss, smoothing_factor=label_smoothing_factor)
    else:
        cost_fn = functools.partial(training_utilities.binary_xentropy_loss,  smoothing_factor=label_smoothing_factor)

    mixup_alpha = config.opt.get("mixup_alpha", 0.0)
    if mixup_alpha != 0.0 and (mode == TrainingMode.MULTILABEL or mode==TrainingMode.MULTICLASS):
        logging.info(f"Mixup with alpha {mixup_alpha} will be used for training")
        mixup_func = functools.partial(mixup.do_mixup, alpha=mixup_alpha, mode=mode)
        mixup_criterion_func = functools.partial(mixup.get_mixedup_xentropy, mode=mode)
    else:
        mixup_func = None
        mixup_criterion_func = None

    p_train_step = jax.pmap(
        functools.partial(train_step, learning_rate_fn=learning_rate_fn,
                          cost_func=cost_fn,
                          mode=mode, mixup_func=mixup_func, 
                          mixup_criterion_func=mixup_criterion_func),
        axis_name='batch', devices=devices)
    p_eval_step = jax.pmap(functools.partial(eval_step, mode=mode, cost_func=cost_fn),
                           axis_name='batch', devices=devices)

    train_metrics = []
    hooks = []
    if jax.process_index() == 0:
        hooks += [periodic_actions.Profile(num_profile_steps=5, logdir=workdir)]
    train_metrics_last_t = time.time()
    best_val_acc = 0.
    logging.info('Initial compilation, this might take some minutes...')

    for step, batch in zip(range(step_offset, num_steps), train_iter):
        if p_feature_extract_fn:
            batch['audio'] = p_feature_extract_fn(batch['audio'])
            if p_spec_aug and not in_model_spec_aug:
                batch['audio'], spec_aug_rng = p_spec_aug(batch['audio'], spec_aug_rng)
        if step == 0:
            print(batch['audio'].shape, batch['audio'].dtype)
            # jnp.save("test_{:05d}.npy".format(step), jax.device_get(batch['audio']))

        state, metrics = p_train_step(state, batch)
        for h in hooks:
            h(step)
        if step == step_offset:
            logging.info('Initial compilation completed.')

        if config.get('log_every_steps'):
            train_metrics.append(metrics)
            if (step + 1) % config.log_every_steps == 0:
                train_metrics = common_utils.get_metrics(train_metrics)
                summary = {
                    f'train_{k}': v
                    for k, v in jax.tree_map(lambda x: x.mean(), train_metrics).items()
                }
                summary['steps_per_second'] = config.log_every_steps / (
                        time.time() - train_metrics_last_t)
                writer.write_scalars(step + 1, copy.copy(summary))
                if wandb_logger:
                    wandb_logger.log(summary, step+1)
                train_metrics = []
                train_metrics_last_t = time.time()

        if (step + 1) % steps_per_epoch == 0:
            epoch = step // steps_per_epoch
            eval_metrics = []
            eval_logits = []
            eval_labels = []
            # sync batch statistics across replicas
            state = training_utilities.sync_batch_stats(state)

            for _ in range(steps_per_eval):
                eval_batch = next(eval_iter)
                if p_feature_extract_fn:
                    eval_batch['audio'] = p_feature_extract_fn(eval_batch['audio'])
                metrics, logits, labels = p_eval_step(state, eval_batch)
                eval_metrics.append(metrics)
                eval_logits.append(logits)
                eval_labels.append(labels)
                # print("logits.shape: {} | type: {}".format(logits.shape, type(logits)))

            eval_metrics = common_utils.get_metrics(eval_metrics)

            summary = jax.tree_map(lambda x: x.mean(), eval_metrics)
            if mode == TrainingMode.MULTILABEL:
                eval_logits = jnp.concatenate([jax.device_get(x) for x in eval_logits])
                eval_labels = jnp.concatenate([jax.device_get(x) for x in eval_labels])
                eval_logits = eval_logits.reshape(-1, eval_logits.shape[-1])
                eval_labels = eval_labels.reshape(-1, eval_labels.shape[-1])
                map_value = average_precision_score(eval_labels.astype('float32'), eval_logits.astype('float32'),
                                                    average="macro")
                summary['accuracy'] = map_value

            logging.info('eval epoch: %d, loss: %.4f, accuracy: %.4f',
                         epoch, summary['loss'], summary['accuracy'] * 100)

            if summary['accuracy'] >= best_val_acc:
                best_val_acc = summary['accuracy']
                state = training_utilities.sync_batch_stats(state)
                training_utilities.save_best_checkpoint(state, workdir, best_val_acc)

            writer.write_scalars(
                step + 1, {f'eval_{key}': val for key, val in summary.items()})
            writer.flush()
            if wandb_logger:
                wandb_logger.log({f'eval_{key}': val for key, val in summary.items()}, step + 1)

        if (step + 1) % steps_per_checkpoint == 0 or step + 1 == num_steps:
            state = training_utilities.sync_batch_stats(state)
            training_utilities.save_checkpoint(state, workdir)

    # Wait until computations are done before exiting
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
    if wandb_logger:
        wandb_logger.finish()
    return state
