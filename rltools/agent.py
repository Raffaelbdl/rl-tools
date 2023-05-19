from abc import abstractmethod, abstractproperty

from rltools.buffer import Buffer


class Agent:
    def __init__(self) -> None:
        self.buffer: Buffer = None

    @abstractmethod
    def get_action(self, observations):
        raise NotImplementedError("get_action method must be implemented")

    @abstractmethod
    def get_value(self, observations):
        raise NotImplementedError("get_value method must be implemented")

    @abstractmethod
    def improve(self, logs: dict):
        raise NotImplementedError("improve method must be implemented")

    @abstractproperty
    def improve_condition(self):
        raise NotImplementedError("improve_condition property must be implemented")
