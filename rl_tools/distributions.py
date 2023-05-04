import distrax as dx
import jax.numpy as jnp


def normal_and_tanh_sample_and_log_prob(key, normal_dist: dx.Normal):
    assert isinstance(
        normal_dist, dx.Normal
    ), "normal_dist is not a dx.Normal distribution"

    sample, log_prob = normal_dist.sample_and_log_prob(seed=key)

    tanh_sample = jnp.tanh(sample)
    tanh_log_prob = log_prob - jnp.log(1.0 - jnp.square(tanh_sample) + 1e-6)

    return tanh_sample, tanh_log_prob
