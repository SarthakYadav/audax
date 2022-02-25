"""
Short time fourier transform implementation in jax

Written for audax by / Copyright 2022, Sarthak Yadav
"""
import jax
from jax import numpy as jnp, lax
from typing import Optional, Callable
from .helpers import batch_pad, _conv_dimension_numbers


def stft(inputs: jnp.array, n_fft: int, hop_length: Optional[int] = None,
         win_length: Optional[int] = None, window: Optional[jnp.array] = None,
         center: bool = True,
         onesided: Optional[bool] = None) -> jnp.array:
    r"""Short-time fourier transform
    This implementation is designed to be consistent with stft implementation from torchaudio
    and does not have all the bells and whistles from scipy.signal.

    Get's the job done though, run's on GPUs and TPUs, and produces identical output to the torchaudio implementation
    in very few lines of code.

    It uses `lax.conv_general_dilated_patches` to generate overlapping patches.

    Returns a jnp.array of size :math:`(* \times T \times N)`, where :math:`*` is the optional batch size of
    :attr:`input`, :math:`N` is the number of frequencies where STFT is applied
    and :math:`T` is the total number of frames used.

    Args:
        input (jnp.array): input signal tensor of shape (T,), (N, T) or (N, T, 1)
        n_fft (int): size of Fourier transform
        hop_length (int, optional): the distance between neighboring sliding window
            frames. Default: ``None`` (treated as equal to ``floor(n_fft / 4)``)
        win_length (int, optional): the size of window frame and STFT filter.
            Default: ``None``  (treated as equal to :attr:`n_fft`)
        window (Tensor, optional): the optional window function.
            Default: ``None`` (treated as window of all :math:`1` s)
        center (bool, optional): whether to pad :attr:`input` on both sides via reflection padding
            so that the :math:`t`-th frame is centered at time :math:`t \times \text{hop\_length}`.
            Default: ``True``
        onesided (bool, optional): controls whether to return half of results to
            avoid redundancy for real inputs.
            Default: ``True`` for real :attr:`input` and :attr:`window`, ``False`` otherwise.
    Returns:
        jnp.array, as described above
    """

    is_real = inputs.dtype == jnp.float32

    assert window.shape[-1] == win_length
    win_length = win_length if win_length is not None else n_fft
    hop_length = hop_length if hop_length is not None else n_fft // 4

    input_dims = inputs.ndim
    single_input = False
    if input_dims == 1:
        inputs = inputs.reshape(1, -1, 1)
        single_input = True
    if input_dims == 2:
        # assuming first dimension is batch, second is time
        inputs = inputs[Ellipsis, jnp.newaxis]

    if center:
        pad = int(n_fft // 2)
        inputs = batch_pad(inputs, pad=pad)

    dimension_numbers = _conv_dimension_numbers(inputs.shape)
    strided = lax.conv_general_dilated_patches(lhs=inputs, filter_shape=(win_length,),
                        window_strides=(hop_length,), padding='VALID',
                        dimension_numbers=dimension_numbers)

    # apply window
    strided = strided * window.reshape(1, 1, -1)
    output = jnp.fft.fft(strided, n=n_fft)

    if onesided is None:
        if is_real:
            onesided = True

    if onesided:
        output = output[:, :, :(n_fft // 2) + 1]

    return output
