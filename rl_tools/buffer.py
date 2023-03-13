from abc import abstractmethod
from typing import List


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
        ]
        super().__init__(keys)

    def reset(self):
        self.buffer = {
            "observations": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "dones": [],
            "next_observations": [],
        }

    def add(self, observation, action, log_prob, reward, done, next_observation):
        self.buffer["observations"].append(observation)
        self.buffer["actions"].append(action)
        self.buffer["log_probs"].append(log_prob)
        self.buffer["rewards"].append(reward)
        self.buffer["dones"].append(done)
        self.buffer["next_observations"].append(next_observation)

    def __len__(self):
        return len(self.buffer["rewards"])

    @staticmethod
    def get_batch_from_data(data, idx):
        keys = [
            "observations",
            "actions",
            "log_probs",
            "rewards",
            "dones",
            "next_observations",
            "advantages",
            "returns",
            "values",
        ]

        batch = {}
        for key in keys:
            batch[key] = data[key][idx]
        return batch
