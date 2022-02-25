"""
Modified training state with easy param freezing support and updating rngs over training iterations

Written for audax by / Copyright 2022, Sarthak Yadav
"""
from typing import Any, Callable, Dict
import jax
from flax import core
from flax import struct
import optax
import flax


class TrainState_v2(struct.PyTreeNode):
    """Simple train state for the common case with a single Optax optimizer.

      Synopsis::
          state = TrainState.create(
              apply_fn=model.apply,
              params=variables['params'],
              tx=tx)
          grad_fn = jax.grad(make_loss_fn(state.apply_fn))
          for batch in data:
            grads = grad_fn(state.params, batch)
            state = state.apply_gradients(grads=grads)

      Note that you can easily extend this dataclass by subclassing it for storing
      additional data (e.g. additional variable collections).

      For more exotic usecases (e.g. multiple optimizers) it's probably best to
      fork the class and modify it.

      Args:
        step: Counter starts at 0 and is incremented by every call to
          `.apply_gradients()`.
        apply_fn: Usually set to `model.apply()`. Kept in this dataclass for
          convenience to have a shorter params list for the `train_step()` function
          in your training loop.
        params: The parameters to be updated by `tx` and used by `apply_fn`.
        frozen_params:
        tx: An Optax gradient transformation.
        opt_state: The state for `tx`.
    """
    step: int
    batch_stats: Any
    dynamic_scale: flax.optim.DynamicScale
    params: core.FrozenDict[str, Any]
    frozen_params: core.FrozenDict[str, Any]
    aux_rng_keys: core.FrozenDict[str, Any]
    opt_state: optax.OptState
    apply_fn: Callable = struct.field(pytree_node=False)
    tx: optax.GradientTransformation = struct.field(pytree_node=False)

    def apply_gradients(self, *, grads, **kwargs):
        """Updates `step`, `params`, `opt_state` and `**kwargs` in return value.

        Note that internally this function calls `.tx.update()` followed by a call
        to `optax.apply_updates()` to update `params` and `opt_state`.

        Args:
          grads: Gradients that have the same pytree structure as `.params`.
          **kwargs: Additional dataclass attributes that should be `.replace()`-ed.

        Returns:
          An updated instance of `self` with `step` incremented by one, `params`
          and `opt_state` updated by applying `grads`, and additional attributes
          replaced as specified by `kwargs`.
        """
        updates, new_opt_state = self.tx.update(
            grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        rng_keys = self.update_rng_keys()

        return self.replace(
            step=self.step + 1,
            params=new_params,
            frozen_params=self.frozen_params,
            opt_state=new_opt_state,
            aux_rng_keys=rng_keys,
            **kwargs)

    def update_rng_keys(self):
        unfrozen = flax.core.unfreeze(self.aux_rng_keys)
        for k in self.aux_rng_keys.keys():
            unfrozen[k] = jax.random.split(unfrozen[k], 1)[0]
        return flax.core.freeze(unfrozen)

    @property
    def get_all_params(self):
        return {**self.params, **self.frozen_params}

    @classmethod
    def create(cls, *, apply_fn, params, frozen_params, tx, aux_rng_keys, **kwargs):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        opt_state = tx.init(params)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            frozen_params=frozen_params,
            tx=tx,
            opt_state=opt_state,
            aux_rng_keys=aux_rng_keys,
            **kwargs,
        )
