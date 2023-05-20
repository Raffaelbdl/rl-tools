from gymnasium.spaces import Space
import numpy as np


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
        next_obs = next_obs.reshape(self.obs_shape)

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

    def _get_samples(self, inds: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "observations": self.observations[inds],
            "actions": self.actions[inds],
            "rewards": self.rewards[inds],
            "dones": self.dones[inds],
            "next_observations": self.observations[(inds + 1) % self.buffer_size],
        }


class OnPolicyBuffer:
    """Buffer for off-policy algorithms"""

    def __init__(self) -> None:
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.next_observations = []
        self.values = []
        self.next_values = []

    def add(self, obs, action, log_prob, rwd, done, next_obs, value, next_value):
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(rwd)
        self.dones.append(done)
        self.next_observations.append(next_obs)
        self.values.append(value)
        self.next_values.append(next_value)

    def sample(self):
        return {
            "observations": np.array(self.observations),
            "actions": np.array(self.actions),
            "log_probs": np.array(self.log_probs),
            "rewards": np.array(self.rewards),
            "dones": np.array(self.dones),
            "next_observations": np.array(self.next_observations),
            "values": np.array(self.values),
            "next_values": np.array(self.next_values),
        }


def get_batch(buffer_dict: dict, inds: np.ndarray) -> dict:
    batch = {}
    for k, v in buffer_dict.items():
        batch[k] = v[inds]
    return batch


def get_time_batch(buffer_dict: dict, inds: np.ndarray) -> dict:
    batch = {}
    for k, v in buffer_dict.items():
        batch[k] = v[:, inds]
    return batch


def get_batch_from_list(list_buffer_dict: list[dict], inds: np.ndarray) -> list[dict]:
    batch = []
    for b in list_buffer_dict:
        batch.append(get_batch(b, inds))
    return batch


def get_time_batch_from_list(
    list_buffer_dict: list[dict], inds: np.ndarray
) -> list[dict]:
    batch = []
    for b in list_buffer_dict:
        batch.append(get_time_batch(b, inds))
    return batch
