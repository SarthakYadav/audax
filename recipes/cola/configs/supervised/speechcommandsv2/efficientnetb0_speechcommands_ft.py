import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.model = ml_collections.ConfigDict()
    config.model.arch = "efficientnet_b0"
    config.model.type = "multiclass"
    config.model.num_classes = 35
    config.model.pretrained = "/home/sarthak/jax_exps/audax/cola/ssl/efficientnet-b0_1x1024_wd1e-7_100eps"
    config.model.pretrained_prefix = "checkpoint_"
    config.model.pretrained_fc_only = True

    config.opt = ml_collections.ConfigDict()
    config.opt.optimizer = "Adam"
    config.opt.learning_rate = 1e-3
    config.opt.weight_decay = 0.
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
    config.audio_config.min_duration = 1.       # min duration in seconds
    config.audio_config.tr_feature_size = 101
    config.audio_config.val_feature_size = 101
    config.audio_config.n_fft = 1024
    config.audio_config.win_len = 400
    config.audio_config.hop_len = 160
    config.audio_config.n_mels = 64
    config.audio_config.fmin = 60
    config.audio_config.fmax = 7800

    config.data = ml_collections.ConfigDict()
    config.data.tr_manifest = "/home/sarthak/my_disk/Datasets/speechcommands_tfrecords_v4_PCM16/train.csv"
    config.data.eval_manifest = "/home/sarthak/my_disk/Datasets/speechcommands_tfrecords_v4_PCM16/val.csv"
    config.data.tr_samples = 84864
    config.data.eval_samples = 9984
    config.data.compression = "ZLIB"
    config.data.reader = "tfio"
    config.data.cacheable = False
    config.data.jax_transforms = True
    config.data.dataset_name = "speechcommandsv2"

    config.batch_size = 1024
    config.half_precision = True
    config.input_shape = (101, 64, 1)
    config.num_epochs = 50
    config.device = 1

    config.wandb = ml_collections.ConfigDict()
    config.wandb.project = "cola_ft"

    return config
