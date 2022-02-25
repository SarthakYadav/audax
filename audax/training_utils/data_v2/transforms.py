"""
Data transforms in tensorflow for the tf.data based datasets
Kept separate from core audax (since it's tensorflow based)

Written for audax by / Copyright 2022, Sarthak Yadav
"""
import tensorflow as tf


def pad_waveform(waveform, seg_length=16000):
    padding = tf.maximum(seg_length - tf.shape(waveform)[0], 0)
    left_pad = padding // 2
    right_pad = padding - left_pad
    padded_waveform = tf.pad(waveform, paddings=[[left_pad, right_pad]])
    return padded_waveform


def random_crop_signal(audio, slice_length):
    data_length = tf.shape(audio, out_type=tf.dtypes.int64)[0]
    max_offset = data_length - slice_length
    if max_offset == 0:
        return audio
    random_offset = tf.random.uniform((), minval=0, maxval=max_offset, dtype=tf.dtypes.int64)
    slice_indices = tf.range(0, slice_length, dtype=tf.dtypes.int64)
    random_slice = tf.gather(audio, slice_indices + random_offset, axis=0)
    return random_slice


def center_crop_signal(audio, slice_length):
    data_length = tf.shape(audio, out_type=tf.dtypes.int64)[0]
    if data_length == slice_length:
        return audio
    center_offset = data_length // 2
    slice_indices = tf.range(0, slice_length, dtype=tf.dtypes.int64)
    return tf.gather(audio, slice_indices + center_offset, axis=0)


def label_parser(example, mode="multiclass", num_classes=527):
    label = tf.sparse.to_dense(example['label'])
    if mode == "multilabel":
        example['label'] = tf.reduce_sum(tf.one_hot(label, num_classes, on_value=1., axis=-1), axis=0)
    else:
        example['label'] = tf.one_hot(label[0], num_classes, on_value=1.)
    return example


def contrastive_labels(example):
    labels = tf.range(0, example['anchor'].shape[0])
    labels = tf.one_hot(labels, example['anchor'].shape[0], on_value=1.)
    example['label'] = labels
    return example


def map_torch_batched_feature_extractor(example, feature_extractor):
    example['audio'] = tf.numpy_function(feature_extractor, [example['audio']], Tout=tf.float32)
    return example


def map_dtype(example, desired=tf.float32):
    example['audio'] = tf.cast(example['audio'], desired)
    example['label'] = tf.cast(example['label'], desired)
    return example