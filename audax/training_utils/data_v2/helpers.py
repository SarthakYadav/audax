"""
Helper utilities for parsing experiment configs and creating data splits.

Written for audax by / Copyright 2022, Sarthak Yadav
"""
import ml_collections
import pandas as pd
import functools
from . import dataset
from . import transforms
from ..misc import DataSplit, TrainingMode, Features


def get_tfrecord_parser(config: ml_collections.ConfigDict, cropper,
                        label_parser_func, is_val=False,
                        is_contrastive=False):
    if config.audio_config.min_duration is not None:
        desired_seq_len = int(config.audio_config.sample_rate * config.audio_config.min_duration)
    else:
        desired_seq_len = int(config.audio_config.sample_rate * 10.)
    if is_contrastive and config.data.reader != "tfio":
        raise NotImplementedError("Only tfio data reader supports contrastive data parsing." +
                                  "Provided config.data.reader == {}".format(config.data.reader))
    if config.data.reader == "tfio":
        print("using tfio bruh")
        if is_contrastive:
            parser_fn = functools.partial(dataset.parse_tf_record_tfio_contrastive,
                                          label_parser=label_parser_func,
                                          cropper=cropper,
                                          noise=config.model.get("contrastive_noise", 0.001),
                                          normalize_waveform=config.audio_config.get("normalize_waveform", False),
                                          normalize_loudness=config.audio_config.get("normalize_loudness", False),
                                          seg_length=desired_seq_len * 2,
                                          file_format=config.data.get("file_format","flac"))
        else:
            parser_fn = functools.partial(dataset.parse_tfrecord_fn_tfio,
                                          label_parser=label_parser_func,
                                          cropper=cropper,
                                          normalize_waveform=config.audio_config.get("normalize_waveform", False),
                                          normalize_loudness=config.audio_config.get("normalize_loudness", False),
                                          seg_length=desired_seq_len,
                                          file_format=config.data.get("file_format","flac"))
    else:
        parser_fn = functools.partial(dataset.parse_tfrecord_fn_v2,
                                      label_parser=label_parser_func,
                                      cropper=cropper,
                                      seg_length=desired_seq_len)
    return parser_fn


def prepare_datasets_v2(config: ml_collections.ConfigDict, batch_size, input_dtype):
    train_files = pd.read_csv(config.data.tr_manifest)['files'].values
    val_files = pd.read_csv(config.data.eval_manifest)['files'].values

    if config.audio_config.min_duration is not None:
        desired_seq_len = int(config.audio_config.sample_rate * config.audio_config.min_duration)
        random_cropper = functools.partial(transforms.random_crop_signal, slice_length=desired_seq_len)
        center_cropper = functools.partial(transforms.center_crop_signal, slice_length=desired_seq_len)
    else:
        desired_seq_len = int(config.audio_config.sample_rate * 10.)
        random_cropper = None
        center_cropper = None

    label_parser_func = functools.partial(transforms.label_parser, mode=config.model.type,
                                          num_classes=config.model.num_classes)


    if config.model.type.lower() in ["contrastive", "cola"]:
        is_contrastive = True
    else:
        is_contrastive = False
    print("prepare_datasets_v2:", config.model.type.lower(), is_contrastive)

    parse_record_train = get_tfrecord_parser(config, cropper=random_cropper,
                                             label_parser_func=label_parser_func,
                                             is_contrastive=is_contrastive)
    parse_record_val = get_tfrecord_parser(config, cropper=center_cropper,
                                           label_parser_func=label_parser_func,
                                           is_val=True,
                                           is_contrastive=is_contrastive)
    fe = None
    # TODO: remove torch transforms
    if not config.data.jax_transforms:
        from . import torch_transforms
        if config.audio_config.features == "log_mel":
            fe = torch_transforms.get_log_mel_extractor(
                sample_rate=config.audio_config.sample_rate,
                frame_length=config.audio_config.win_len,
                frame_step=config.audio_config.hop_len,
                fft_length=config.audio_config.n_fft,
                n_mels=config.audio_config.n_mels,
                fmin=config.audio_config.fmin,
                fmax=config.audio_config.fmax
            )
        elif config.audio_config.features == "spectrogram":
            fe = torch_transforms.get_spectrogram_extractor(
                frame_length=config.audio_config.win_len,
                frame_step=config.audio_config.hop_len,
                fft_length=config.audio_config.n_fft
            )
        else:
            fe = None
    if fe and not config.data.jax_transforms:
        batched_feature_extractor = functools.partial(transforms.map_torch_batched_feature_extractor,
                                                      feature_extractor=fe)
    else:
        batched_feature_extractor = None

    train_dataset = dataset.get_dataset_v2(
        train_files, batch_size, parse_example=parse_record_train, num_classes=config.model.num_classes,
        data_split=DataSplit.TRAIN, compression=config.data.get("compression", "ZLIB"),
        feature_extraction_func=batched_feature_extractor, cacheable=config.data.cacheable,
        is_contrastive=is_contrastive
    )

    val_dataset = dataset.get_dataset_v2(
        val_files, batch_size, parse_example=parse_record_val, num_classes=config.model.num_classes,
        data_split=DataSplit.EVAL, compression=config.data.get("compression", "ZLIB"),
        feature_extraction_func=batched_feature_extractor, cacheable=config.data.cacheable,
        is_contrastive=is_contrastive
    )

    if not config.data.jax_transforms:
        dtype_map_func = functools.partial(transforms.map_dtype, desired=input_dtype)
        train_dataset = train_dataset.map(dtype_map_func)
        val_dataset = val_dataset.map(dtype_map_func)
    return train_dataset, val_dataset
