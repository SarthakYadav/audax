import enum


@enum.unique
class TrainingMode(enum.Enum):
    """Look up for similarity measure in contrastive model."""
    MULTICLASS = "multiclass"
    MULTILABEL = "multilabel"
    CONTRASTIVE = "contrastive"
    COLA = "cola"


@enum.unique
class DataSplit(enum.Enum):
    """Look up for similarity measure in contrastive model."""
    TRAIN = "train"
    EVAL = "eval"
    VAL = "val"


@enum.unique
class Features(enum.Enum):
    """Look up for similarity measure in contrastive model."""
    RAW = "raw"
    SPECTROGRAM = "spectrogram"
    LOGMEL = "log_mel"
