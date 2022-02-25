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

"""

import time
import jax
import wandb
import functools
import ml_collections
import jax.numpy as jnp
from absl import logging
from clu import metric_writers
from clu import periodic_actions
from flax import jax_utils
from flax.training import common_utils
from contrastive_model import COLA
from jax import random
import tensorflow as tf
try:
    # Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

from audax.training_utils.misc import TrainingMode, Features, DataSplit
from audax.training_utils import training_utilities, train_contrastive
from audax.training_utils.data_v2.helpers import prepare_datasets_v2
from audax.models.utils import SimilarityMeasure


def create_COLA_model(*, encoder_cls, half_precision, similarity_measure, embedding_dim,
                      temperature=0.1, spec_aug=None, **kwargs):
    platform = jax.local_devices()[0].platform
    if half_precision:
        if platform == 'tpu':
            model_dtype = jnp.bfloat16
        else:
            model_dtype = jnp.float16
    else:
        model_dtype = jnp.float32
    return COLA(model_cls=encoder_cls, similarity_measure=similarity_measure,
                embedding_dim=embedding_dim, temperature=temperature, spec_aug=spec_aug, 
                dtype=model_dtype, **kwargs)


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
        wandb_logger = wandb.init(project='{}'.format(config.wandb.get("project", "audax-cola")),
                                  group="{}".format(config.data.dataset_name),
                                  config=config.to_dict(), name=workdir.split("/")[-1])
    writer = metric_writers.create_default_writer(
        logdir=workdir, just_logging=jax.process_index() != 0)
    training_utilities.write_config_to_json(workdir, config)
    rng = random.PRNGKey(seed)
    if config.batch_size % jax.device_count() > 0:
        raise ValueError('Batch size must be divisible by the number of devices')
    local_batch_size = config.batch_size // jax.process_count()
    logging.info("Process count: {}".format(jax.process_count()))
    device = config.get("device", None)
    if device is not None:
        devices = [jax.local_devices()[device]]
    else:
        devices = jax.local_devices()
    print("Training on the following devices: {}".format(devices))
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
                ), axis_name='batch', devices=devices)
        else:
            p_feature_extract_fn = None
    else:
        p_feature_extract_fn = None
    in_model_spec_aug = config.get("in_model_spec_aug", False)
    p_spec_aug, spec_aug_rng = training_utilities.get_spec_augment(config, devices, pmapped=not in_model_spec_aug)
    if p_spec_aug:
        logging.info("Using spec augment, in model: {}".format(str(in_model_spec_aug)))

    num_examples = config.data.tr_samples
    steps_per_epoch = (num_examples // config.batch_size)

    if config.num_train_steps == -1:
        num_steps = int(steps_per_epoch * config.num_epochs)
    else:
        num_steps = config.num_train_steps

    if config.steps_per_eval == -1:
        num_validation_examples = config.data.eval_samples
        steps_per_eval = num_validation_examples // config.batch_size
    else:
        steps_per_eval = config.steps_per_eval

    steps_per_checkpoint = steps_per_epoch
    base_learning_rate = config.opt.learning_rate * config.batch_size / 256.

    model_cls, frontend_cls = training_utilities.get_model_frontend_cls(config)
    if mode == TrainingMode.COLA:
        cost_fn = functools.partial(training_utilities.cross_entropy_loss, smoothing_factor=None)
        model = create_COLA_model(encoder_cls=model_cls, half_precision=config.half_precision,
                                  similarity_measure=SimilarityMeasure(config.model.similarity_measure),
                                  spec_aug=p_spec_aug if in_model_spec_aug else None,
                                  embedding_dim=config.model.embedding_dim, frontend_cls=frontend_cls)
    else:
        raise ValueError("Unsupported mode {}".format(mode))
    print(model)
    learning_rate_fn = training_utilities.create_learning_rate_fn(
        config, base_learning_rate, steps_per_epoch)

    state = training_utilities.create_train_state(rng, config, model, learning_rate_fn)
    state = training_utilities.restore_checkpoint(state, workdir)

    step_offset = int(state.step)
    state = jax_utils.replicate(state, devices=devices)

    label_smoothing_factor = config.opt.get("label_smoothing_factor", None)
    if label_smoothing_factor:
        logging.info("Training with Label Smoothing, alpha = {}".format(label_smoothing_factor))

    p_train_step = jax.pmap(
        functools.partial(train_contrastive.train_step, learning_rate_fn=learning_rate_fn,
                          cost_func=cost_fn,
                          mode=mode),
        axis_name='batch', devices=devices)
    p_eval_step = jax.pmap(
        functools.partial(train_contrastive.eval_step, mode=mode, 
                          cost_func=cost_fn),
        axis_name='batch', devices=devices)

    train_metrics = []
    hooks = []
    if jax.process_index() == 0:
        hooks += [periodic_actions.Profile(num_profile_steps=5, logdir=workdir)]
    train_metrics_last_t = time.time()
    best_val_loss = 1e5
    logging.info('Initial compilation, this might take some minutes...')
    for step, batch in zip(range(step_offset, num_steps), train_iter):
        is_best_ckpt = False
        if p_feature_extract_fn:
            batch['anchor'] = p_feature_extract_fn(batch['anchor'])
            batch['positive'] = p_feature_extract_fn(batch['positive'])
            if p_spec_aug and not in_model_spec_aug:
                # batch['anchor'], spec_aug_rng = p_spec_aug(batch['anchor'], spec_aug_rng)
                batch['positive'], spec_aug_rng = p_spec_aug(batch['positive'], spec_aug_rng)
            if step == 0:
                print(batch['anchor'].shape, batch['anchor'].dtype)
                print(batch['positive'].shape, batch['positive'].dtype)
                # jnp.save("./test.npy", jax.device_get(batch['audio']))
                # print("saved..")
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
                writer.write_scalars(step + 1, summary)
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
                    eval_batch['anchor'] = p_feature_extract_fn(eval_batch['anchor'])
                    eval_batch['positive'] = p_feature_extract_fn(eval_batch['positive'])
                metrics, logits, labels = p_eval_step(state, eval_batch)
                eval_metrics.append(metrics)
                eval_logits.append(logits)
                eval_labels.append(labels)
                # print("logits.shape: {} | type: {}".format(logits.shape, type(logits)))

            eval_metrics = common_utils.get_metrics(eval_metrics)
            summary = jax.tree_map(lambda x: x.mean(), eval_metrics)
            logging.info('eval epoch: %d, loss: %.4f',
                         epoch, summary['loss'])
            if summary['loss'] <= best_val_loss:
                best_val_loss = summary['loss']
                state = training_utilities.sync_batch_stats(state)
                training_utilities.save_best_checkpoint(state, workdir, best_val_loss)

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
