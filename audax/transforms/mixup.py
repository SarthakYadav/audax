"""
Implementation for the paper

Zhang, H., Cisse, M., Dauphin, Y.N. and Lopez-Paz, D., 2017.
"mixup: Beyond empirical risk minimization". arXiv preprint arXiv:1710.09412.

Written for audax by / Copyright 2022, Sarthak Yadav
"""
import jax
import jax.numpy as jnp
from ..training_utils.misc import TrainingMode


def do_mixup(key, inputs, targets, alpha=0.2, mode: TrainingMode=TrainingMode.MULTILABEL):
    """
    Applies Mixup to the input.
    :param inputs: input batch tensor of shape (bsize, C, H, W)
    :param targets: input gt tensor of shape (bsize, num_classes)
    :param alpha: alpha parameter for beta distribution generation
    :return: tuple of mixed_inputs, mixed_outputs
    """
    key, subkey = jax.random.split(key, 2)
    ndim = inputs.ndim
    bsize = inputs.shape[0]
    lam = jax.random.beta(key, alpha, alpha, shape=(bsize,))
    perms = jax.random.permutation(subkey, bsize)
    
    lam_desired_shape = (bsize,) + (1,) * (ndim-1)
    mixed_x = inputs * lam.reshape(lam_desired_shape) + inputs[perms] * (1 - lam.reshape(lam_desired_shape))

    if mode == TrainingMode.MULTILABEL:
        mixed_y = targets * lam.reshape((bsize, 1)) + targets[perms] * (1 - lam.reshape((bsize, 1)))
        return mixed_x, mixed_y, None, None
    else:
        y_a, y_b = targets, targets[perms]
        return mixed_x, y_a, y_b, lam


def get_mixedup_xentropy(criterion, pred, y_a, y_b, lam, mode: TrainingMode=TrainingMode.MULTILABEL):
    if mode == TrainingMode.MULTILABEL:
        return criterion(pred, y_a)
    elif mode == TrainingMode.MULTICLASS:
        return criterion(pred, y_a) * lam + criterion(pred, y_b) * (1 - lam)
    else:
        raise ValueError(f"Unsupported training mode {mode}, should be one of ['TrainingMode.MULTILABEL', 'TrainingMode.MULTICLASS']")
