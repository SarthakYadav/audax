## Data Prep

Data preparation can be done using the script provided in [data_prep/make_tfrecords.py](make_tfrecords.py), which accepts 

1. `.csv` manifest with the following structure
    ```shell
    files,labels
    "path to wav/flac file","label_string"
    ```
   Wav/flac files should be of desired sample rate and channels. (For speechcommands, 16000 kHz, mono). The label string comprises of list of label(s) delimited by `delimiter`. For eg, for a multilabel dataset it could be `"up,down"` tags, the delimiter being `,`
2. `.json` label map which maps individual labels to integer values
    ```
    {
      "backward": 0, 
      "bed": 1, 
      "bird": 2,
       .....
    }
    ```
3. The following command can then be used to create the tfrecords
   ```shell
   python make_tfrecords.py --manifest_path train.csv --labels_map lbl_map.json --output_dir $BASE_OUTPUT_DIR --split_name train --multiproc_threads 6 --files_per_record 2048 --desired_duration 1. --clip_larger_files --compression "ZLIB"
   ```
   More options can be found in the script. The above command was used to generate speechcommands v2 tfrecords
4. Once this is done, final step is to make `.csv` files with the following structure
   ```
   files
   $BASE_DIR/$SPLIT_NAME/file_00000-00020_bytes_compressed.tfrec
   ```
   which are used in model configs such as [recipes/speechcommands/configs/resnet18.py](../speechcommands/configs/resnet18.py)
