# `COLA`: Contrastive learning of general-purpose audio representations

- [About](#about)
- [Configuration](#configuration)
- [Training and evaluation](#training-and-evaluation)
- [Models and their performance](#models-and-their-performance)
  - [AudioSet](#audioset)
    - [This repo](#this-repo)
    - [Literature](#literature)
- [References](#references)


## About
This recipes covers implementation of COLA [1], the recent approach for contrastive self-supervised pretraining of audio representations.   
Pretraining experiments are done on AudioSet, with linear-probe experiments on SpeechCommand-v2. More linear probe experiments are being conducted.

## Configuration
- All models were trained for 50 epochs with AdamW optimizer. Linear warmup + cosine annealing schedule was used. Early experiments with 100 epochs paired showed pretty close performance.
- It's pretty easy to overfit: 100 epochs yields lower performance.
- Models were trained on a single TPU-v3 device (with 8 tpu cores), with a per-tpu-core batch size of 256, (unless stated otherwise).
- Training was done on random 0.96-second crops (yes, 960 ms worth of audio)
- All configs used for training are provided in the [configs](configs) directory. Most of the configuration arguments are self-explanatory.
- `config.py` file can also be found in the linked pretrained checkpoint, for exact hyperparams of that run.
- For data preparation, view [recipes/data_prep](../data_prep).
- Unlike the official COLA paper, `SpecAugment (frequency and time masking)` is used **only on the positive sample**, which stopped overfitting on `ConvNeXT` models and provided better performance than applying spec augment to both `anchor` and `positive`.
- EfficientNet and ConvNeXT models are trained with Stochastic Depth/Drop Path
- No label balancing was used in linear-probe

## Training and evaluation
The following commands

1. first pretrain a [`COLA`](contrastive_model.py) model

```shell
python main.py --config configs/ssl/convnext_tiny.py --workdir /tmp/convnext_tiny_8x256_ssl
```

2. Then finetune (fc-only) configuration for SpeechCommands-v2
```shell
# calculates performance metrics
python main.py --config configs/supervised/speechcommandsv2/convnext_tiny_sc_ft.py --workdir /tmp/convnext_tiny_8x256_sc_ftonly_1x1024 --mode train
```
Taking a look at the above config file, you'll see additional arguments under `config.model`:
- `config.model.pretrained`: This points to the directory were ssl model is saved
- `config.model.pretrained_fc_only`: linear probe flag

3. Evaluation on the SpeechCommand-v2
```shell
python main.py --config configs/supervised/speechcommandsv2/convnext_tiny_sc_ft.py --workdir /tmp/convnext_tiny_8x256_sc_ftonly_1x1024 --mode eval \
       --eval_manifest_override <path to test csv> --eval_steps_override 11005
```

## Models and their performance

| Architecture   | features                                | SpecAugment   | linear probe<br>(SpeechCommands-v2) | Pretrained weights                                                                                                                                                                                                           |
|----------------|-----------------------------------------|---------------|-------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ConvNeXT-Tiny  | melspectrograms<br>(64 nmels, 1024-fft) | positive only | 0.710                               | [COLA weights](https://drive.google.com/drive/folders/1Pul5XcAv1OWIFYx00dtrgCWgQ_km-u5i?usp=sharing) <br/> [SpeechCommands-v2 weights](https://drive.google.com/drive/folders/1CdUpQCXcS4fQ4MsZwsekNe6hXR_LEjkD?usp=sharing) |
| ConvNeXT-Large | melspectrograms<br>(64 nmels, 1024-fft) | positive only | 0.7159                              | [COLA weights](https://drive.google.com/drive/folders/1632ExyoC9xK_EczdoP7haV01oFLMJpxf?usp=sharing) <br/>[SpeechCommands-v2 weights](https://drive.google.com/drive/folders/1mvGTccHl_ZSLNJUVzNwQ7Zqyhkb97dL_?usp=sharing)                |

## Up next
- Evaluation on more models and configurations, repeated runs.
- Experiments on more datasets (SpeechCommands-v2, VoxCeleb, etc)

## References
[1] Saeed, A., Grangier, D. and Zeghidour, N., 2021, June. Contrastive learning of general-purpose audio representations. In ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 3875-3879). IEEE. 
