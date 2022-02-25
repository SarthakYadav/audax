"""
functional utils for audio feature extraction

Written for audax by / Copyright 2022, Sarthak Yadav
"""
import jax
import warnings
import math
from jax import numpy as jnp, lax
from typing import Optional, Callable, Any
from .stft import stft
from .helpers import batch_pad


def _hz_to_mel(freq: float, mel_scale: str = "htk") -> float:
    r"""Convert Hz to Mels.

    Args:
        freqs (float): Frequencies in Hz
        mel_scale (str, optional): Scale to use: ``htk`` or ``slaney``. (Default: ``htk``)

    Returns:
        mels (float): Frequency in Mels
    """
    if mel_scale not in ['slaney', 'htk']:
        raise ValueError('mel_scale should be one of "htk" or "slaney".')

    if mel_scale == "htk":
        return 2595.0 * math.log10(1.0 + (freq / 700.0))

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (freq - f_min) / f_sp
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0

    if freq >= min_log_hz:
        mels = min_log_mel + math.log(freq / min_log_hz) / logstep
    return mels


def _mel_to_hz(mels: jnp.array, mel_scale: str = "htk") -> jnp.array:
    """Convert mel bin numbers to frequencies.

    Args:
        mels (jnp.array): Mel frequencies
        mel_scale (str, optional): Scale to use: ``htk`` or ``slaney``. (Default: ``htk``)

    Returns:
        freqs (Tensor): Mels converted in Hz
    """
    if mel_scale not in ['slaney', 'htk']:
        raise ValueError('mel_scale should be one of "htk" or "slaney".')
    if mel_scale == "htk":
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0

    log_t = (mels >= min_log_mel)
    freqs.at[log_t].set(min_log_hz * jnp.exp(logstep * (mels[log_t] - min_log_mel)))
    return freqs


def _create_triangular_filterbank(
        all_freqs: jnp.array,
        f_pts: jnp.array,
) -> jnp.array:
    """Create a triangular filter bank.

    Args:
        all_freqs (Tensor): STFT freq points of size (`n_freqs`).
        f_pts (Tensor): Filter mid points of size (`n_filter`).

    Returns:
        fb (Tensor): The filter bank of size (`n_freqs`, `n_filter`).
    """
    # Adopted from Librosa
    # calculate the difference between each filter mid point and each stft freq point in hertz
    f_diff = f_pts[1:] - f_pts[:-1]  # (n_filter + 1)
    slopes = f_pts[jnp.newaxis, Ellipsis] - all_freqs[Ellipsis, jnp.newaxis]  # (n_freqs, n_filter + 2)
    # create overlapping triangles
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]  # (n_freqs, n_filter)
    up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_freqs, n_filter)
    fb = jnp.maximum(0., jnp.minimum(down_slopes, up_slopes))
    return fb


def melscale_fbanks(
        n_freqs: int,
        n_mels: int,
        sample_rate: int,
        f_min: float,
        f_max: Optional[float] = None,
        norm: Optional[str] = None,
        mel_scale: str = "htk",
) -> jnp.array:
    r"""Create a frequency bin conversion matrix.

    Args:
        n_freqs (int): Number of frequencies to highlight/apply
        f_min (float): Minimum frequency (Hz)
        f_max (float): Maximum frequency (Hz)
        n_mels (int): Number of mel filterbanks
        sample_rate (int): Sample rate of the audio waveform
        norm (str or None, optional): If 'slaney', divide the triangular mel weights by the width of the mel band
            (area normalization). (Default: ``None``)
        mel_scale (str, optional): Scale to use: ``htk`` or ``slaney``. (Default: ``htk``)

    Returns:
        Tensor: Triangular filter banks (fb matrix) of size (``n_freqs``, ``n_mels``)
        meaning number of frequencies to highlight/apply to x the number of filterbanks.
        Each column is a filterbank so that assuming there is a matrix A of
        size (..., ``n_freqs``), the applied result would be
        ``A * melscale_fbanks(A.size(-1), ...)``.

    """
    f_max = f_max if f_max is not None else float(sample_rate // 2)
    assert f_min <= f_max, 'Require f_min: {} < f_max: {}'.format(f_min, f_max)
    if norm is not None and norm != "slaney":
        raise ValueError("norm must be one of None or 'slaney'")
    all_freqs = jnp.linspace(0, sample_rate // 2, n_freqs)

    # calculate mel freq bins
    m_min = _hz_to_mel(f_min, mel_scale=mel_scale)
    m_max = _hz_to_mel(f_max, mel_scale=mel_scale)

    m_pts = jnp.linspace(m_min, m_max, n_mels + 2)
    f_pts = _mel_to_hz(m_pts, mel_scale=mel_scale)

    # create filterbank
    fb = _create_triangular_filterbank(all_freqs, f_pts)

    if norm is not None and norm == "slaney":
        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (f_pts[2:n_mels + 2] - f_pts[:n_mels])
        fb *= enorm[jnp.newaxis, Ellipsis]

    # if (fb.max(axis=0) == 0.).any():
    #     warnings.warn(
    #         "At least one mel filterbank has all zero values. "
    #         f"The value for `n_mels` ({n_mels}) may be set too high. "
    #         f"Or, the value for `n_freqs` ({n_freqs}) may be set too low."
    #     )

    return fb


def spectrogram(
        inputs: jnp.array,
        pad: int,
        window: jnp.array,
        n_fft: int,
        hop_length: int,
        win_length: int,
        power: Optional[float] = 2.0,
        normalized: bool = False,
        center: bool = True,
        onesided: bool = True,
        return_complex: bool = True
) -> jnp.array:
    r"""Create a spectrogram or a batch of spectrograms from a raw audio signal.
        The spectrogram can be either magnitude-only or complex.

        Args:
            inputs (Tensor): Tensor of audio of dimension `(..., time)`
            pad (int): Two sided padding of signal
            window (Tensor): Window tensor that is applied/multiplied to each frame/window
            n_fft (int): Size of FFT
            hop_length (int): Length of hop between STFT windows
            win_length (int): Window size
            power (float or None): Exponent for the magnitude spectrogram,
                (must be > 0) e.g., 1 for energy, 2 for power, etc.
                If None, then the complex spectrum is returned instead.
            normalized (bool): Whether to normalize by magnitude after stft
            center (bool, optional): whether to pad :attr:`waveform` on both sides so
                that the :math:`t`-th frame is centered at time :math:`t \times \text{hop\_length}`.
                Default: ``True``
            pad_mode (string, optional): controls the padding method used when
                :attr:`center` is ``True``. Default: ``"reflect"``
            onesided (bool, optional): controls whether to return half of results to
                avoid redundancy. Default: ``True``
            return_complex (bool, optional):
                Indicates whether the underlying complex valued spectrogram is to be returned.
                If False, absolute value of the spectrogram is returned
                This argument is only effective when ``power=None``. It is ignored for
                cases where ``power`` is a number as in those cases, the returned tensor is
                power spectrogram, which is a real-valued tensor.

        Returns:
            Tensor: Dimension `(..., freq, time)`, freq is
            ``n_fft // 2 + 1`` and ``n_fft`` is the number of
            Fourier bins, and time is the number of window hops (n_frame).
        """
    if power is None and not return_complex:
        raise ValueError(f"in correct combination of power(={power}) and return_complex(={return_complex}) provided.")

    if pad > 0:
        inputs = batch_pad(inputs, pad, "zeros")

    spec_f = stft(
        inputs,
        n_fft,
        hop_length,
        win_length,
        window,
        center=center,
        onesided=onesided
    )

    if normalized:
        spec_f /= jnp.sqrt(jnp.sum(jnp.power(window, 2.)))

    if power is not None:
        if power == 1.0:
            return jnp.abs(spec_f)
        return jnp.power(jnp.abs(spec_f), power)
    if not return_complex:
        return jnp.abs(spec_f)
    return spec_f


def apply_melscale(spectrogram: jnp.array,
                   melscale_filterbank: jnp.array,
                   precision: Any = lax.Precision.HIGHEST) -> jnp.array:
    r"""

    Args:
        spectrogram (jnp.array): A spectrogram STFT of dimension (..., time, freq)
        melscale_filterbank (jnp.array): MelScale filterbank of dimension (``n_freqs``, ``n_mels``)
        precision (jax.lax.Precision): Precision enum. Default value is  Precision.HIGHEST, which corresponds to FP32

    Returns
        jnp.array: A melspectrogram of dimension (..., time, ``n_mels``)
    """
    melspec = jnp.matmul(spectrogram, melscale_filterbank, precision=precision)
    return melspec
