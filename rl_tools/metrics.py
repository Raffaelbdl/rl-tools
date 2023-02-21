from collections import deque, UserDict


class Metrics(UserDict):
    def __init__(self, *user_keys):
        super().__init__()
        self.user_keys = user_keys

    def add(self, key, to_add):
        self.__getitem__(key).append(to_add)

    def reset(self):
        for k in self.user_keys:
            self.__setitem__(k, deque())
