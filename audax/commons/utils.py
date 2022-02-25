from jax import lax, numpy as jnp


def hz2mel(f: float):
  """Hertz to mel scale conversion.

  Converts a frequency from hertz to mel values.

  Args:
    f: a frequency in Hz

  Returns:
    value of f on a mel scale
  """
  return 2595 * jnp.log10(1 + f / 700)


def mel2hz(m: float):
  """Mel to hertz conversion.

  Converts a frequency from mel to hertz values.

  Args:
    m: a frequency in mels

  Returns:
    value of m on the linear frequency scale
  """
  return 700 * (jnp.power(10, m / 2595) - 1)


def conv_dimension_numbers(input_shape):
    """
    Computes the dimension numbers based on the input shape.
    copied from flax for easy access
    """
    ndim = len(input_shape)
    lhs_spec = (0, ndim - 1) + tuple(range(1, ndim - 1))
    rhs_spec = (ndim - 1, ndim - 2) + tuple(range(0, ndim - 2))
    out_spec = lhs_spec
    return lax.ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)
