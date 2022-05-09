"""
Audio dataset and parsing utilities for audax

Written for audax by / Copyright 2022, Sarthak Yadav
"""
import os
import tensorflow as tf
import tensorflow_io as tfio
from typing import Callable, Optional
from functools import partial
from ..misc import TrainingMode, DataSplit
from .transforms import pad_waveform, contrastive_labels, loudness_normalization


def parse_tfrecord_fn_v2(example,
                         label_parser=None,
                         cropper=None,
                         seg_length=16000):
    feature_description = {
        "audio": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.VarLenFeature(tf.int64),
        "duration": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)
    a = tf.io.parse_tensor(example['audio'], tf.float32)
    a.set_shape([None])
    a = pad_waveform(a, seg_length=seg_length)
    if cropper:
        a = tf.numpy_function(cropper, [a], tf.float32)
    example['audio'] = a
    if label_parser:
        example = label_parser(example)
    return example


def parse_tfrecord_fn_tfio(example,
                           label_parser=None,
                           cropper=None,
                           normalize_waveform=False,
                           normalize_loudness=False,
                           seg_length=16000,
                           file_format="flac"):
    feature_description = {
        "audio": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.VarLenFeature(tf.int64),
        "duration": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)
    if file_format == "flac":
        a = tfio.audio.decode_flac(example['audio'], dtype=tf.int16)
        a = tf.cast(a, tf.float32) / 32768.
    elif file_format == "ogg":
        a = tfio.audio.decode_vorbis(example['audio'])
    elif file_format == "wav":
        a = tfio.audio.decode_wav(example['audio'], dtype=tf.int16)
        a = tf.cast(a, tf.float32) / 32768.
    a = tf.reshape(a, (-1,))
    if normalize_loudness:
        a = loudness_normalization(a)
    a = pad_waveform(a, seg_length=seg_length)
    if cropper:
        a = cropper(a)
    if normalize_waveform:
        a = (a - tf.reduce_mean(a)) / (tf.math.reduce_std(a) + 1e-8)
    example['audio'] = a
    if label_parser:
        example = label_parser(example)
    return example


def parse_tf_record_tfio_contrastive(example, cropper,
                                     label_parser=None,
                                     normalize_waveform=False,
                                     normalize_loudness=False,
                                     noise=0.001,
                                     seg_length=16000,
                                     file_format="flac"):
    feature_description = {
        "audio": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.VarLenFeature(tf.int64),
        "duration": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)
    if file_format == "flac":
        a = tfio.audio.decode_flac(example['audio'], dtype=tf.int16)
        a = tf.cast(a, tf.float32) / 32768.
    elif file_format == "ogg":
        a = tfio.audio.decode_vorbis(example['audio'])
        # vorbis is already float, and in correct range
    elif file_format == "wav":
        a = tfio.audio.decode_wav(example['audio'], dtype=tf.int16)
        a = tf.cast(a, tf.float32) / 32768.
    a = tf.reshape(a, (-1,))
    
    if normalize_loudness:
        a = loudness_normalization(a)
    a = pad_waveform(a, seg_length=seg_length)

    waveform_a = cropper(a)
    waveform_p = cropper(a)

    if normalize_waveform:
        waveform_a = tf.math.l2_normalize(waveform_a, epsilon=1e-9)
        waveform_p = tf.math.l2_normalize(waveform_p, epsilon=1e-9)

    waveform_p = waveform_p + (noise * tf.random.normal(tf.shape(waveform_p)))
    output = {
        "anchor": waveform_a,
        "positive": waveform_p
    }
    return output


def map_contrastive_labels(example):
    labels = tf.range(0, limit=example['anchor'].shape[0], dtype=tf.int64)
    labels = tf.one_hot(labels, depth=example['anchor'].shape[0])
    example['label'] = labels
    return example


def get_dataset_v2(filenames,
                batch_size,
                parse_example,
                num_classes,
                data_split=DataSplit.TRAIN,
                file_ext="tfrec",
                compression="ZLIB",
                feature_extraction_func=None,
                cacheable=False,
                is_contrastive=False
                ):
    options = tf.data.Options()
    options.autotune.enabled = True
    options.threading.private_threadpool_size = 96  # 0=automatically determined
    options.deterministic = False
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.with_options(options)
    # shuffle filenames every epoch
    dataset = dataset.shuffle(len(filenames), seed=0, reshuffle_each_iteration=True)    
    # dataset = tf.data.TFRecordDataset(filenames, compression_type=compression,
    #                                       num_parallel_reads=tf.data.AUTOTUNE)
    # dataset = dataset.with_options(options)
    # dataset = dataset.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    if cacheable:
        dataset = dataset.interleave(
            lambda x: tf.data.TFRecordDataset(x, compression_type=compression,
                                            num_parallel_reads=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE),
            cycle_length=tf.data.AUTOTUNE, block_length=32,
            num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).prefetch(tf.data.AUTOTUNE)
        dataset = dataset.cache()
        dataset = dataset.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        # gives better data throughput for AudioSet, which can't really be cached anyway
        dataset = dataset.interleave(
            lambda x: tf.data.TFRecordDataset(x, compression_type=compression,
                                              num_parallel_reads=tf.data.AUTOTUNE).map(parse_example, num_parallel_calls=tf.data.AUTOTUNE),
            cycle_length=tf.data.AUTOTUNE, block_length=32,
            num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).prefetch(tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE, drop_remainder=True)
    if feature_extraction_func:
        dataset = dataset.map(feature_extraction_func,
                              num_parallel_calls=tf.data.AUTOTUNE)
    if is_contrastive:
        dataset = dataset.map(contrastive_labels, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
