# A bit more on feature extraction

Before moving forward, it's worth noting that feature extraction pipeline is _functional_ in nature, as opposed to nice callable objects as commonly found in torchaudio.
This is done to best comply with Jax's design principles of being functionally pure, and to act as a reference that you can compose your own transforms on.

It's also worth noting that `audax`, by default, adopts a `(..., channels)` format.

## Spectrograms
```python
import soundfile as sf
from audax.core import functional
import jax.numpy as jnp
from jax import jit
from functools import partial

x, sr = sf.read("misc_files/sample.wav")
NFFT = 512
WIN_LEN = 400
HOP_LEN = 160
SR = sr

# creates a spectrogram helper
window = jnp.hanning(WIN_LEN)
spec_func = partial(functional.spectrogram, pad=0, window=window, n_fft=NFFT,
                   hop_length=HOP_LEN, win_length=WIN_LEN, power=2.,
                   normalized=False, center=True, onesided=True)
jax_spec = spec_func(x)
# shape: (1, 101, 257)

jax_spec = jnp.clip(jax_spec, a_min=1e-8, a_max=1e8)
# log-scaling the spectrogram. 
# This is the audax spectrogram as found in main README.
jax_spec = jnp.log(jax_spec)

# you can also extract features on a batch of signals
xs = jnp.stack([x for _ in range(8)])   # a (8, 16000) array
batched_spec = spec_func(xs)    # output of shape (8, 101, 257)

# jit, vmap also work as expected. for vmap usage, check `audax.training_utils.train_supervised`
spec_func_jit = jit(spec_func)
batched_spec = spec_func_jit(xs)
```

## Melspectrograms
Melspectrograms can be create by chaining `audax.core.functional.spectrogram` followed by converting to melscale filterbanks
(`audax.core.functional.melscale_fbanks`) using `audax.core.functional.apply_melscale`

```python
import soundfile as sf
from audax.core import functional
import jax.numpy as jnp
from jax import jit
from functools import partial

x, sr = sf.read("misc_files/sample.wav")
NFFT = 512
WIN_LEN = 400
HOP_LEN = 160
SR = sr

# creates a spectrogram helper
window = jnp.hanning(WIN_LEN)
spec_func = partial(functional.spectrogram, pad=0, window=window, n_fft=NFFT,
                   hop_length=HOP_LEN, win_length=WIN_LEN, power=2.,
                   normalized=False, center=True, onesided=True)
fb = functional.melscale_fbanks(n_freqs=(NFFT//2)+1, n_mels=64,
                         sample_rate=SR, f_min=60., f_max=7800.)
mel_spec_func = partial(functional.apply_melscale, melscale_filterbank=fb)
jax_spec = spec_func(x)
mel_spec = mel_spec_func(jax_spec)    # output of shape (1, 101, 64)

```

Helper functions for both spectrogram and melspectrogram extraction can be found in `audax.feature_helper`
