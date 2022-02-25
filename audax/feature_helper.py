from .core import functional
from functools import partial
from jax import numpy as jnp, lax
from typing import Optional, Callable


def spectrogram_helper(
        pad: int,
        n_fft: int,
        hop_length: int,
        win_length: int,
        power: Optional[float] = 2.0,
        normalized: bool = False,
        window_type: Callable = jnp.hanning,
        center: bool = True,
):
    window = window_type(win_length)
    return partial(functional.spectrogram, pad=pad, window=window, n_fft=n_fft,
                   hop_length=hop_length, win_length=win_length, power=power,
                   normalized=normalized, center=center, onesided=True)


def melscale_helper(
        n_mels: int,
        n_fft: int,
        sample_rate: int,
        f_min: float,
        f_max: float,
        norm=None,
        precision=lax.Precision.HIGH
):
    fb = functional.melscale_fbanks(n_freqs=(n_fft//2)+1, n_mels=n_mels,
                         sample_rate=sample_rate, f_min=f_min, f_max=f_max, norm=norm)
    return partial(functional.apply_melscale, melscale_filterbank=fb, precision=precision)
