"""
COLA implementation from

Saeed, A., Grangier, D. and Zeghidour, N., 2021, June.
"Contrastive learning of general-purpose audio representations".
In ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)
(pp. 3875-3879). IEEE.

Based on the official implementation (https://github.com/google-research/google-research/tree/master/cola)

Written for audax by / Copyright 2022, Sarthak Yadav
"""
from typing import Callable, Optional, Any
import flax.linen as nn
from audax.models.layers.similarity_layers import BilinearProduct, DotProduct
from audax.models.utils import SimilarityMeasure
import jax.numpy as jnp


Dtype = Any


class COLA(nn.Module):
    model_cls: Callable
    frontend_cls: Optional[Callable] = None
    similarity_measure: SimilarityMeasure = SimilarityMeasure.BILINEAR
    embedding_dim: int = 512
    temperature: Optional[float] = 0.2
    spec_aug: Optional[Callable] = None
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
            if self.spec_aug is not None and train:
                spec_aug_rng = self.make_rng('spec_aug')
                anchor_outputs, positive_outputs = jnp.split(outputs, 2)
                # only applying SpecAugment to positives gave best results
                positive_outputs, _ = self.spec_aug(positive_outputs, spec_aug_rng)
                outputs = jnp.concatenate([anchor_outputs, positive_outputs])

        encoder = self.model_cls(num_classes=None, dtype=self.dtype, name='encoder')
        fc = nn.Dense(self.embedding_dim, use_bias=False, name='embedding_fc')

        encoded = encoder(outputs, train=train)
        embedding = fc(encoded)

        if self.similarity_measure == SimilarityMeasure.BILINEAR:
            embedding = nn.LayerNorm(name="bilinear_layernorm")(embedding)
            embedding = nn.tanh(embedding)

        # embeddings are split here
        anchor_embeddings, positive_embeddings = jnp.split(embedding, 2)
        if self.similarity_measure == SimilarityMeasure.BILINEAR:
            similarity_layer = BilinearProduct(self.embedding_dim, dtype=self.dtype, name='similarity_layer')
        else:
            similarity_layer = DotProduct(name='similarity_layer')

        similarities = similarity_layer(anchor_embeddings,
                                        positive_embeddings)
        if self.similarity_measure == SimilarityMeasure.DOT:
            similarities = similarities / self.temperature
        return similarities
