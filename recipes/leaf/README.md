# LEAF and SincNet

- [About](#about)
- [Configuration](#configuration)
- [Training and evaluation](#training-and-evaluation)
- [Models and their performance](#models-and-their-performance)
  - [AudioSet](#audioset)
    - [This repo](#this-repo)
    - [Literature](#literature)
- [References](#references)

## About
This recipes covers implementation of raw waveform frontends, LEAF [1] and SincNet [2].   
All experiments are done on AudioSet, and more datasets will be added soon.

## Configuration
- All models were trained for 50 epochs with AdamW optimizer. Linear warmup + cosine annealing schedule was used. Early experiments with 100 epochs paired showed pretty close performance. 
- Models were trained on a single TPU-v3 device (with 8 tpu cores), with a per-tpu-core batch size of 128, (unless stated otherwise).
- Training was done on random 5-second crops (5-second center crop for calculating validation metrics.)
- All configs used for training are provided in the [configs](configs) directory. Most of the configuration arguments are self-explanatory.
- `config.py` file can also be found in the linked pretrained checkpoint, for exact hyperparams of that run.
- For data preparation, view [recipes/data_prep](../data_prep).
- SpecAugment (frequency and time masking), and Dropout (before FC layer) are used for all models. 
- All configs have an additional flag `config.in_model_spec_aug = True` which governs that the SpecAugment happens after frontend features are extracted.
- EfficientNet and ConvNeXT models are trained with Stochastic Depth/Drop Path
- No label balancing was used.

## Training and evaluation
The following commands trains a [`Classifier`](../../audax/models/classifier.py), which has in-built `frontend` support, the following steps were followed

```shell
python main.py --config configs/efficientnetb0_audioset_leaf.py --workdir /tmp/efficientnetb0_8x128
```

```shell
# calculates performance metrics
python main.py --config configs/efficientnetb0_audioset_leaf.py --workdir /tmp/efficientnetb0_8x128 --mode eval
```

## Models and their performance
### AudioSet

#### This repo:

| Architecture    | features | training<br>samples/sec | mAP          | mAUC          | d'           | pretrained model                                                                     |
|-----------------|----------|-------------------------|--------------|---------------|--------------|--------------------------------------------------------------------------------------|
| EfficientNet-B0 | LEAF     | ~2048                   | 0.3802±0.002 | 0.9700±0.0009 | 2.6530±0.019 | https://drive.google.com/drive/folders/1DOg1R3MlnUaLmwWO2-Mjp8nxVotbqIYo?usp=sharing |
| EfficientNet-B0 | SincNet  | ~4100                   | 0.3640±0.004 | 0.9671±0.0006 | 2.6024±0.013 | https://drive.google.com/drive/folders/1ahSgweHSzEs6jekWWVQhw_1b32zhqDsv?usp=sharing |

#### Literature

| Architecture        | features | mAP | mAUC       | d'       |
|---------------------|----------|-----|------------|----------|
| EfficientNet-B0 [1] | LEAF     | -   | 0.968±.001 | 2.63±.01 |
| EfficientNet-B0 [1] | SincNet  | -   | 0.961±.000 | 2.48±.00 |

## Up next
- Evaluation on more models and configurations
- Experiments on more datasets (SpeechCommands-v2, VoxCeleb, etc)

## References
[1] Zeghidour, H., Teboul, O., Quitry, F., and Tagliasacchi, M., LEAF: A Learnable Frontend for Audio Classification, In International Conference on Learning Representations, 2021.    
[2] Ravanelli, M. and Bengio, Y., 2018, December. Speaker recognition from raw waveform with sincnet. In 2018 IEEE Spoken Language Technology Workshop (SLT) (pp. 1021-1028). IEEE.    