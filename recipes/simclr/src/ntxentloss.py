import jax.numpy as jnp
from jax import lax
from audax.models.utils import l2_normalize


def nt_xentloss(temperature=0.1, dtype=jnp.float32, precision=lax.Precision.HIGH):

    def func(embedding, labels):
        # labels are ignored
        embedding = l2_normalize(embedding, axis=-1)
        logits = jnp.divide(jnp.matmul(embedding, embedding.T, precision=precision), temperature)
        mask = jnp.eye(embedding.shape[0]//2, dtype=dtype)
        mask = jnp.tile(mask, (2, 2))
        logits_mask = (jnp.ones_like(mask) - jnp.eye(embedding.shape[0]))
        mask = mask * logits_mask

        exp_logits_den = jnp.log(jnp.sum(jnp.exp(logits) * logits_mask, axis=1, keepdims=True) + 1e-10)
        exp_logits_pos = jnp.log(jnp.exp(logits) + 1e-10)

        log_prob = exp_logits_pos - exp_logits_den

        mean_log_prob = jnp.sum((mask * log_prob), axis=1) / jnp.sum(mask, axis=1)
        loss = -1 * mean_log_prob
        return loss.mean()

    return func
