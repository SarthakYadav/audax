# SpeechCommands v2 Classification

## Configuration
- All models were trained for 50 epochs with AdamW optimizer. Linear warmup + cosine annealing schedule was used. 
- Models were trained on a `single TPU-v3` device on a single core with a batch size of 1024, (unless stated otherwise).
- All configs used for training are provided in the [configs](configs) directory. Most of the configuration arguments are self-explanatory.
- `config.py` file can also be found in the linked pretrained checkpoint, for exact hyperparams of that run.
- For data preparation, view [recipes/data_prep](../data_prep).
- SpecAugment (frequency and time masking) and Dropout (before FC layer) are used for all models
- EfficientNet and ConvNeXT models are trained with Stochastic Depth/Drop Path
- No label balancing was used.

## Training and evaluation
To train a [`Classifier`](../../audax/models/classifier.py), say, the resnet18 model, the following steps were followed

```shell
python main.py --config configs/resnet_18.py --workdir /tmp/resnet18_8x256
```

```shell
export EVAL_MANIFEST_PATH="path_to_test.csv"
python main.py --config configs/resnet_18.py --workdir /tmp/resnet18_8x256 --mode eval --eval_manifest_override $EVAL_MANIFEST_PATH --eval_steps_override 11005
```

## Models and their performance

| Architecture    | features                                | Accuracy| pretrained model                                                                     |
|-----------------|-----------------------------------------|---------|--------------------------------------------------------------------------------------|
| ResNet18        | melspectrograms<br>(64 nmels, 1024-fft) | 0.9562  | https://drive.google.com/drive/folders/1jyykLnZIIA4KRe-Qn53qxdtuA2Mhiuxr?usp=sharing |
| EfficientNet-b0 | melspectrograms<br>(64 nmels, 1024-fft) | 0.9344  | https://drive.google.com/drive/folders/1h4PayVEiA-oxNwML8E0g-V4b9vrAH0Or?usp=sharing |

## Up next
- More ResNet, EfficientNet and ConvNeXT configurations
- Models trained on Spectrograms
