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



def get_model_checkpoint(workspace: str, model: nn.Module, optimizer: optim.Optimizer, *args, **kwargs) -> CheckpointInterface:
    from ise_cdg_utility.checkpoint.model_checkpoint import ModelCheckpointHandler, InOutModelCheckpointHandler
    if workspace == 'kaggle':
        return InOutModelCheckpointHandler(model, optimizer, *args, **kwargs)
    return ModelCheckpointHandler(model, optimizer, *args, **kwargs)
