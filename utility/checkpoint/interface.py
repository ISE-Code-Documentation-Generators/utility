import abc

from torch import nn
from torch import optim

from utility.checkpoint import CheckpointInterface



class CheckpointInterface(abc.ABC):

    @abc.abstractmethod
    def save_checkpoint(self):
        pass

    @abc.abstractmethod
    def load_checkpoint(self):
        pass



def get_model_checkpoint(model: nn.Module, optimizer: optim.Optimizer, filename: str) -> CheckpointInterface:
    from utility.checkpoint.model_checkpoint import ModelCheckpointHandler
    return ModelCheckpointHandler(model, optimizer, filename)
