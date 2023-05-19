import haiku as hk
import jax.random as jrd
import jax.nn as nn
import jax.numpy as jnp

key1, key2 = jrd.split(jrd.PRNGKey(0))
dummy_x = jnp.zeros((1, 1))


@hk.transform
def example_1(x):
    x = hk.Linear(1, name="yolo_1")(x)
    x = nn.relu(x)
    return hk.Linear(1, name="yolo_2")(x)


params_1 = example_1.init(key1, dummy_x)


class WithClass(hk.Module):
    def __call__(self, x):
        x = hk.Linear(1, name="yolo_1")(x)
        x = nn.relu(x)
        return hk.Linear(1, name="yolo_2")(x)


@hk.transform
def example_2(x):
    module = WithClass(name="superyolo")
    return module(x)


params_2 = example_1.init(key2, dummy_x)

params_de_mon_agent = hk.data_structures.merge(params_1, params_2)
print(params_de_mon_agent.keys())

# print(params_1.keys())
# print(params_2.keys())
