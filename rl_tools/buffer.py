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
