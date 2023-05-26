from copy import copy
from typing import Optional, Type

import distrax as dx
from einops import rearrange
import haiku as hk
import jax
import jax.lax as jlax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrd
import numpy as np

Array = np.ndarray | jnp.ndarray
PRNGKey = jrd.PRNGKeyArray


class LinearEncoder(hk.Module):
    """Simple Linear encoder with ReLU activation"""

    def __init__(self, layers: list[int], name: str | None = None):
        super().__init__(name)
        self.layers = layers

    def __call__(self, x: Array) -> Array:
        for l in self.layers:
            x = jnn.relu(hk.Linear(l)(x))
        return x


class AtariNatureCNN(hk.Module):
    """Nature CNN taken from the PPO implementation"""

    def __init__(self, rearrange_pattern: str | None = None, name: str | None = None):
        super().__init__(name)

        self.w_init = hk.initializers.Orthogonal(np.sqrt(2))
        self.b_init = hk.initializers.Constant(0.0)

        self.rearrange_pattern = rearrange_pattern

    def __call__(self, x: Array) -> Array:
        if self.rearrange_pattern is not None:
            x = rearrange(x, self.rearrange_pattern)

        x = jnn.relu(hk.Conv2D(32, 8, 4, w_init=self.w_init, b_init=self.b_init)(x))
        x = jnn.relu(hk.Conv2D(64, 4, 2, w_init=self.w_init, b_init=self.b_init)(x))
        x = jnn.relu(hk.Conv2D(64, 3, 1, w_init=self.w_init, b_init=self.b_init)(x))

        return x
