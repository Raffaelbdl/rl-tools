from typing import Callable
from functools import partial

try:
    import chex
    import haiku as hk
    import jax
    import jax.random as jrd
    import optax
except ImportError:
    print("The following packages are necessary to use the updates functions")
    print("- jax (install as explained on the GitHub repository)")
    print("- chex")
    print("- haiku")
    print("- optax")

Grads = chex.ArrayTree


@partial(jax.jit, static_argnums=(0))
def apply_updates(
    optimizer: optax.GradientTransformation,
    params: hk.Params,
    opt_state: optax.OptState,
    grads: Grads,
):
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state


@partial(jax.jit, static_argnums=(3, 4))
def update(
    params: hk.Params,
    key: jrd.PRNGKeyArray,
    batch: dict,
    loss_fn: Callable,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
):
    output, grads = jax.value_and_grad(loss_fn, has_aux=True)(params, key, batch)
    params, opt_state = apply_updates(optimizer, params, opt_state, grads)
    return params, opt_state, output
