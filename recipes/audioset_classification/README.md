# AudioSet Classification

## About the dataset

AudioSet availability varies a lot. The models provided in this recipe were trained on 
a dataset with the following statistics in each split.

| Split            | #Samples |
|------------------|----------|
| train balanced   | 18966    |
| train unbalanced | 1766954  |
| eval             | 17408    |

- Preliminary hyperparameter tuning (learning rate, weight decay, batch size) was done using a 10% random split, but when training the full model mAP was computed directly on evaluation set.
- All models were trained for 50 epochs with AdamW optimizer. Linear warmup + cosine annealing schedule was used. Early experiments with 100 epochs paired showed pretty close performance. 
- Models were trained on a `single TPU-v3` device (with 8 tpu cores), with a per-tpu-core batch size of 256, (unless stated otherwise).
- Training was done on random 5-second crops (5-second center crop for calculating validation metrics.)

## Configuration
- All configs used for training are provided in the [configs](configs) directory. Most of the configuration arguments are self-explanatory.
- `config.py` file can also be found in the linked pretrained checkpoint, for exact hyperparams of that run.
- For data preparation, view [recipes/data_prep](../data_prep).
- SpecAugment (frequency and time masking), Mixup and Dropout (before FC layer) are used for all models
- EfficientNet and ConvNeXT models are trained with Stochastic Depth/Drop Path
- No label balancing was used.

## Training and evaluation
To train a [`Classifier`](../../audax/models/classifier.py), say, the resnet18 model, the following steps were followed

```shell
python main.py --config configs/resnet_18.py --workdir /tmp/resnet18_8x256
```

```shell
# calculates mAP, mAUC, and dprime on the full 10-second clip
python main.py --config configs/resnet_18.py --workdir /tmp/resnet18_8x256 --mode eval
```

## Models and their performance

| Architecture    | features                                | training<br>samples/sec | mAP     | mAUC     | d'      | pretrained model                                                                     |
|-----------------|-----------------------------------------|-------------------------|---------|----------|---------|--------------------------------------------------------------------------------------|
| ResNet18        | melspectrograms<br>(64 nmels, 1024-fft) | ~7500                   | 0.3648  | 0.9693   | 2.6472  | https://drive.google.com/drive/folders/1-DnF_JKbby4QfBbr4GOPc7N-C3h-XGmE?usp=sharing |
| EfficientNet-b0 | melspectrograms<br>(64 nmels, 1024-fft) | ~7250                   | 0.3832  | 0.9718   | 2.7002  | https://drive.google.com/drive/folders/1SpVedYbC06xB3xw415SfcIZ2hDtjYoOx?usp=sharing |
| ConvNeXT-Tiny   | melspectrograms<br>(64 nmels, 1024-fft) | ~4450                   | 0.4016  | 0.9733   | 2.7317  | https://drive.google.com/drive/folders/1LI3z_expAkaVSdq-CWOvkWfoyP4rTGoN?usp=sharing |

## Up next
- More ResNet, EfficientNet and ConvNeXT configurations
- Models trained on Spectrograms