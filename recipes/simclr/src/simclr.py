from typing import Callable, Optional, Any
import flax.linen as nn
import jax.numpy as jnp


Dtype = Any


class SimCLR(nn.Module):
    model_cls: Callable
    frontend_cls: Optional[Callable] = None
    embedding_dim: int = 512
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs, train: bool = True):
        """
        Inputs must have even numbered batch size
        with top N/2 indices being the anchors
        and bottom N/2 being corresponding positives
        """
        outputs = inputs
        if self.frontend_cls is not None:
            outputs = self.frontend_cls(dtype=jnp.float32, name="frontend")(outputs)
            outputs = outputs[Ellipsis, jnp.newaxis]
            outputs = outputs.astype(self.dtype)
        encoder = self.model_cls(num_classes=None, dtype=self.dtype, name='encoder')
        fc = nn.Dense(self.embedding_dim, use_bias=False, name='embedding_fc')
        encoded = encoder(outputs, train=train)
        embedding = fc(encoded)
        return embedding
