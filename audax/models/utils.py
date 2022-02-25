import enum
from typing import Optional
import jax


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


@enum.unique
class SimilarityMeasure(enum.Enum):
    """Look up for similarity measure in contrastive model."""
    DOT = "dot_product"
    BILINEAR = "bilinear_product"


def l2_normalize(x, axis=None, eps=1e-12):
    """
    l2 normalize the input on axis dimension
    """
    return x * jax.lax.rsqrt((x * x).sum(axis=axis, keepdims=True) + eps)
