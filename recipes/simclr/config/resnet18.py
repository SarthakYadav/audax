import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.model = ml_collections.ConfigDict()
    config.model.arch = "ResNet18"
    config.model.type = "contrastive"
    config.model.num_classes = None
    config.model.embedding_dim = 512
    config.model.temperature = 0.1

    config.opt = ml_collections.ConfigDict()
    config.opt.optimizer = "Adamw"
    config.opt.learning_rate = 1e-3
    config.opt.weight_decay = 1e-6
    config.opt.schedule = "cosine"
    config.opt.warmup_epochs = 5.0
    config.opt.momentum = 0.9

    config.log_every_steps = 100
    config.num_train_steps = -1
    config.steps_per_eval = -1

    config.audio_config = ml_collections.ConfigDict()
    config.audio_config.normalize_waveform = True
    config.audio_config.sample_rate = 16000
    config.audio_config.features = "log_mel"
    config.audio_config.min_duration = .96       # min duration in seconds
    # config.audio_config.tr_feature_size = 501
    # config.audio_config.val_feature_size = 501
    config.audio_config.n_fft = 1024
    config.audio_config.win_len = 400
    config.audio_config.hop_len = 160
    config.audio_config.n_mels = 64
    config.audio_config.fmin = 60
    config.audio_config.fmax = 7800

    config.data = ml_collections.ConfigDict()
    config.data.tr_manifest = "/home/sarthak/my_disk/Datasets/audioset_tfrecords_v4_PCM16/train.csv"
    config.data.eval_manifest = "/home/sarthak/my_disk/Datasets/audioset_tfrecords_v4_PCM16/eval.csv"
    config.data.tr_samples = 1785920
    config.data.eval_samples = 17408
    config.data.compression = "ZLIB"
    config.data.reader = "tfio"
    config.data.cacheable = False
    config.data.jax_transforms = True
    config.data.dataset_name = "audioset"

    config.batch_size = 256*8
    config.half_precision = True
    config.input_shape = (100, 64, 1)
    config.num_epochs = 50

    config.wandb = ml_collections.ConfigDict()
    config.wandb.project = "audax-simclr"

    return config
