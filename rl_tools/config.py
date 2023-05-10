import yaml
from yaml.loader import SafeLoader


def get_config(config_path: str, **kwargs):
    with open(config_path) as f:
        config: dict = yaml.load(f, SafeLoader)

    for k, v in kwargs.items():
        config[k] = v

    return config
