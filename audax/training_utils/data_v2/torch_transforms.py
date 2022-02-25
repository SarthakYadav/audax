"""
Tensorflow FFT is painfully slow
https://github.com/tensorflow/tensorflow/issues/6541

This file provides a sample of how to use torch transforms in tf.dataset pipelines. It's much faster than using
tensorflow fft, but much slower than doing everything in jax after batches have been created.

Additional benefit: switching frameworks won't affect input features


The transforms here should be used following these general guidelines


- SpectrogramParser, SpectrogramPostProcess and ToMelScale should be called on batched signals
  of shape (N, T), and thus should be used after batching has been done in your pipeline

- Cropping functions should be called early in the pipeline, on unbatched signals of shape (T,)

Written for audax by / Copyright 2022, Sarthak Yadav
"""
import functools
import torch
from audio_utils.common.transforms import Compose, CenterCrop, RandomCrop
from audio_utils.common.feature_transforms import SpectrogramParser, SpectrogramPostProcess, ToMelScale


def apply_feature_extractor(audios, feature_extractor):
    """
    Applies given feature extractor to a batch of audios

    This function can be easily wrapped in tf.Dataset pipeline as

    >>> extractor_features = functools.partial(apply_feature_extractor, feature_extractor)
    >>> # where `feature_extractor` is the callable transform
    >>> tf.numpy_function(extractor_features, [input_batch], Tout=tf.float32)

    Parameters
    ----------
    audios: float32 array of shape (N, T)
    feature_extractor: callable object for features

    Returns
    -------
    np array of shape (N, frequency_bins, timesteps, 1)
    """
    x = torch.from_numpy(audios).unsqueeze(1)
    return feature_extractor(x).squeeze(1).unsqueeze(-1).numpy()


def apply_cropper(audio, cropper):
    """
    Applies given cropper to a batch of audios

    This function can be easily wrapped in tf.Dataset pipeline as

    >>> crop_features = functools.partial(apply_cropper, cropper)
    >>> # where `cropper` is the callable crop transform
    >>> tf.numpy_function(crop_features, [input_audio], Tout=tf.float32)

    Parameters
    ----------
    audio: float32 waveform array of shape (T,)
    cropper: callable cropper object that returns a (M,) shaped crop of input, M <= T

    Returns
    -------
    float32 np.array of shape (N, M)
    """
    x = torch.from_numpy(audio).unsqueeze(0)
    return cropper(x).squeeze(0).numpy()


def get_spectrogram_extractor(frame_length=400,
                              frame_step=160,
                              fft_length=1024,
                              log_compress=True,
                              normalize=False,
                              power=2,
                              **kwargs):
    tfs = Compose([
        SpectrogramParser(window_length=frame_length, hop_length=frame_step, n_fft=fft_length, mode="after_batch"),
        SpectrogramPostProcess(window_length=frame_length, normalize=normalize,
                               log_compress=log_compress, power=power, mode="after_batch")
    ])
    return functools.partial(apply_feature_extractor, feature_extractor=tfs)


def get_log_mel_extractor(
        sample_rate=16000,
        frame_length=400,
        frame_step=160,
        fft_length=1024,
        n_mels=64,
        fmin=60.0,
        fmax=7800.0,
        power=2,
        normalize=False,
        **kwargs):
    tfs = Compose([
        SpectrogramParser(window_length=frame_length, hop_length=frame_step, n_fft=fft_length, mode="after_batch"),
        SpectrogramPostProcess(window_length=frame_length, power=power, normalize=normalize,
                               log_compress=False, mode="after_batch"),
        ToMelScale(sample_rate=sample_rate, hop_length=frame_step, n_fft=fft_length, n_mels=n_mels,
                   fmin=fmin, fmax=fmax, norm=None)
    ])
    return functools.partial(apply_feature_extractor, feature_extractor=tfs)


def get_random_cropper(seq_length):
    cropper = RandomCrop(seq_length)
    return functools.partial(apply_cropper, cropper=cropper)


def get_center_cropper(seq_length):
    cropper = CenterCrop(seq_length)
    return functools.partial(apply_cropper, cropper=cropper)
