import abc


class CheckpointInterface(abc.ABC):

    @abc.abstractmethod
    def save_checkpoint(self):
        pass

    @abc.abstractmethod
    def load_checkpoint(self):
        pass
