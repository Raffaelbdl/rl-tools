import os
import pickle


class Saver:
    def __init__(self, name: str) -> None:
        self.path = os.path.join("results", name)
        os.makedirs(self.path)

    def save(self, name: str, params):
        with open(os.path.join(self.path, name), "wb") as f:
            pickle.dump(params, f)
        f.close()

    def load(self, name: str):
        with open(os.path.join(self.path, name), "rb") as f:
            params = pickle.load(f)
        f.close()
        return params
