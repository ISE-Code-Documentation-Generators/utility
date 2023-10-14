import abc

from torch import nn
from torch import optim



class CheckpointInterface(abc.ABC):

    @abc.abstractmethod
    def save_checkpoint(self):
        pass

    @abc.abstractmethod
    def load_checkpoint(self):
        pass



def get_model_checkpoint(model: nn.Module, optimizer: optim.Optimizer, filename: str) -> CheckpointInterface:
    from ise_cdg_utility.checkpoint.model_checkpoint import ModelCheckpointHandler
    return ModelCheckpointHandler(model, optimizer, filename)
