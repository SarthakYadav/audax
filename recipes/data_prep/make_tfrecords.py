import tensorflow as tf
import json
import numpy as np
import soundfile as sf
import pandas as pd
import tqdm
import argparse
import os
import io

try:
    # Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass


def get_int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def load_audio_tf(f):
    x, sr = sf.read(f)
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    return x


parser = argparse.ArgumentParser()
parser.add_argument("--manifest_path", type=str, help="Path to manifest csv file")
parser.add_argument("--labels_map", type=str, help="Path to label map file")
parser.add_argument("--labels_delim", type=str, default=",", help="labels delimiter")
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--split_name", type=str)
parser.add_argument("--multiproc_threads", type=int,
                    default=None, help="Number of multiprocessing threads, None by default (no multiprocessing)")
parser.add_argument("--files_per_record", type=int, default=1,
                    help="How many blocks worth of data should a single tfrecord contain")
parser.add_argument("--compression", default="ZLIB", type=str,
                    help="compression format for tfrecords (Default=ZLIB)")
parser.add_argument("--desired_duration", type=float, default=1)
parser.add_argument("--clip_larger_files", action="store_true")

args = parser.parse_args()
df = pd.read_csv(args.manifest_path)
print(df)
# block_paths = df['files'].values


segments = []
for ix in range(0, len(df), args.files_per_record):
    segments.append((ix, ix + args.files_per_record))
num_tfrecords = len(segments)

with open(args.labels_map, "r") as fd:
    lbl_map = json.load(fd)


def readfile(f):
    with open(f, "rb") as stream:
        return stream.read()


def load_and_pad_data(fpath):
    x, sr = sf.read(fpath)
    orig_duration = int(len(x) / sr)
    desired_samples = int(args.desired_duration * sr)
    if desired_samples is not None:
        if len(x) < desired_samples:
            tile_size = (desired_samples // x.shape[0]) + 1
            x = np.tile(x, tile_size)[:desired_samples]
        if args.clip_larger_files:
            if len(x) > desired_samples:
                x = x[:desired_samples]
    output = None
    with io.BytesIO() as fd:
        sf.write(fd, x, sr, "PCM_16", format="flac")
        output = fd.getvalue()
    return output, orig_duration


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def make_tf_example(record):
    labels = [lbl_map[lbl] for lbl in record['labels'].split(args.labels_delim)]
    processed_audio, orig_duration = load_and_pad_data(record['files'])
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                # audio as FloatList feature is EXTREMELY SLOW to deserialize
                # thus just encoding as bytes
                "audio": _bytes_feature(processed_audio),
                "label": get_int64_feature(labels),
                "duration": get_int64_feature([orig_duration]),
            }
        )
    )
    return example


def process_segments(segment_idx):
    segment = segments[segment_idx]
    start = segment[0]
    end = segment[1]
    records = df.copy()[start:end]
    examples = []
    for idx in range(len(records)):
        block_examples = make_tf_example(records.iloc[idx])
        examples.append(block_examples)

    tfrecord_name = "file_%.5i-%.5i_bytes_compressed.tfrec" % (segment_idx, num_tfrecords)
    tfrecord_fld = os.path.join(f"{args.output_dir}", args.split_name)
    if not os.path.exists(tfrecord_fld):
        os.makedirs(tfrecord_fld)
    tfrecord_path = os.path.join(tfrecord_fld, tfrecord_name)
    with tf.io.TFRecordWriter(
            tfrecord_path,
            options=args.compression
    ) as writer:
        for example in examples:
            writer.write(example.SerializeToString())

    if args.multiproc_threads:
        print("Done {:05d}/{:05d}".format(segment_idx, num_tfrecords))


if __name__ == "__main__":
    if not args.multiproc_threads:
        for ix in tqdm.tqdm(range(num_tfrecords)):
            process_segments(ix)
    else:
        from multiprocessing import Pool

        pool = Pool(args.multiproc_threads)
        o = pool.map_async(process_segments, range(num_tfrecords))
        res = o.get()
        pool.close()
        pool.join()
