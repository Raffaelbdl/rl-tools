from abc import abstractmethod
from typing import Dict, List

from gymnasium.spaces import Space
import numpy as np


class Buffer:
    def __init__(self, keys: List[str]) -> None:
        self.keys = keys
        self.buffer = None

    @abstractmethod
    def reset(self):
        raise NotImplementedError("reset method must be implemented")

    @abstractmethod
    def add(self, *args):
        raise NotImplementedError("add method must be implemented")

    @abstractmethod
    def __len__(self):
        raise NotImplementedError("__len__ method must be implemented")


class SimpleOnPolicyBuffer(Buffer):
    def __init__(self) -> None:
        keys = [
            "observations",
            "actions",
            "log_probs",
            "rewards",
            "dones",
            "next_observations",
            "values",
            "next_values",
        ]
        super().__init__(keys)

    def reset(self):
        self.buffer = {key: [] for key in self.keys}

    def add(
        self,
        observation,
        action,
        log_prob,
        reward,
        done,
        next_observation,
        values=None,
        next_values=None,
    ):
        self.buffer["observations"].append(observation)
        self.buffer["actions"].append(action)
        self.buffer["log_probs"].append(log_prob)
        self.buffer["rewards"].append(reward)
        self.buffer["dones"].append(done)
        self.buffer["next_observations"].append(next_observation)
        self.buffer["values"].append(values)
        self.buffer["next_values"].append(next_values)

    def __len__(self):
        return len(self.buffer["rewards"])

    @staticmethod
    def get_batch_from_data(data: dict, idx):
        batch = {}
        for key in data.keys():
            batch[key] = data[key][idx]
        return batch

    @staticmethod
    def get_time_batch_from_data(data: dict, idx):
        batch = {}
        for key in data.keys():
            batch[key] = data[key][:, idx]
        return batch

    @staticmethod
    def get_batch_from_data_list(data: list, idx):
        batch = []
        for d in data:
            batch.append(SimpleOnPolicyBuffer.get_batch_from_data(d, idx))
        return batch

    @staticmethod
    def get_time_batch_from_data_list(data: list, idx):
        batch = []
        for d in data:
            batch.append(SimpleOnPolicyBuffer.get_time_batch_from_data(d, idx))
        return batch



class ReplayBuffer:
    """Buffer for off-policy algorithms"""

    def __init__(
        self,
        seed: int,
        buffer_size: int,
        obs_space: Space,
        action_space: Space,
    ) -> None:
        self.rng = np.random.default_rng(seed=seed)
        self.buffer_size = max(buffer_size, 1)

        self.obs_shape = obs_space.shape
        self.action_shape = action_space.shape

        self.observations = np.zeros(
            (self.buffer_size,) + self.obs_shape, obs_space.dtype
        )
        self.actions = np.zeros(
            (self.buffer_size,) + self.action_shape, action_space.dtype
        )
        self.rewards = np.zeros((self.buffer_size,), np.float32)
        self.dones = np.zeros((self.buffer_size,), np.float32)

        self.pos = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        next_obs: np.ndarray,
    ) -> None:
        obs = obs.reshape(self.obs_shape)
        action = action.reshape(self.action_shape)
        reward = reward
        done = done
        next_obs = obs.reshape(self.obs_shape)

        self.observations[self.pos] = obs.copy()
        self.actions[self.pos] = action.copy()
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.observations[(self.pos + 1) % self.buffer_size] = next_obs.copy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int):
        if self.full:
            inds = self.rng.integers(1, self.buffer_size, size=batch_size)
            inds = (inds + self.pos) % self.buffer_size
        else:
            inds = self.rng.integers(0, self.pos, size=batch_size)
        return self._get_samples(inds)

    def _get_samples(self, inds: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            "observations": self.observations[inds],
            "actions": self.actions[inds],
            "rewards": self.rewards[inds],
            "dones": self.dones[inds],
            "next_observations": self.observations[(inds + 1) % self.buffer_size],
        }
