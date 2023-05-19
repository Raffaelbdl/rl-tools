from abc import abstractmethod

import jax.random as jrd


class Agent:
    def __init__(self, key: jrd.PRNGKeyArray) -> None:
        self.key = key

    @abstractmethod
    def get_action(self, observations):
        raise NotImplementedError("get_action method must be implemented")

    @abstractmethod
    def improve(self, logs: dict):
        raise NotImplementedError("improve method must be implemented")

    def _next_rng_key(self):
        self.key, _key = jrd.split(self.key)
        return _key
