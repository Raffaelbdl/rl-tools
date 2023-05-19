import os
import pickle


class Saver:
    def __init__(self, name: str, override_save: bool = False) -> None:
        self.path = os.path.join("results", name)
        os.makedirs(self.path, exist_ok=override_save)

    def save(self, name: str, params):
        with open(os.path.join(self.path, name), "wb") as f:
            pickle.dump(params, f)
        f.close()

    def load(self, name: str):
        with open(os.path.join(self.path, name), "rb") as f:
            params = pickle.load(f)
        f.close()
        return params


class Pickler(Saver):
    def __init__(self, name: str, override_save: bool = False) -> None:
        self.path = os.path.join("results", name)
        os.makedirs(self.path, exist_ok=override_save)

        self.params = None

    def update(self, to_pickle):
        self.params = to_pickle

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.save("save_at_last", self.params)
