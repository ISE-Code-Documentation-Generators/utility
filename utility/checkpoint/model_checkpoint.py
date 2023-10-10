import dataclasses
import typing

import torch
from torch import nn
from torch import optim

from utility.checkpoint.interface import CheckpointInterface



class ModelCheckpointHandler(CheckpointInterface):

    @dataclasses.dataclass
    class ModelCheckpoint:
        model: typing.Any
        optimizer: typing.Any

        @property
        def context(self):
            return dataclasses.asdict(self)

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, filename: str):
        self.model = model
        self.optimizer = optimizer
        self.filename = filename

    def get_checkpoint(self):
        return self.ModelCheckpoint(model=self.model.state_dict(), optimizer=self.optimizer.state_dict())

    def save_checkpoint(self):
        print(f'Saving Checkpoint in {self.filename}...')
        torch.save(self.get_checkpoint().context, self.filename)

    def load_checkpoint(self):
        print(f'Loading Checkpoint from {self.filename}...')
        checkpoint = self.ModelCheckpoint(**torch.load(self.filename))
        self.model.load_state_dict(checkpoint.model)
        self.optimizer.load_state_dict(checkpoint.optimizer)



def get_model_checkpoint(model: nn.Module, optimizer: optim.Optimizer, filename: str) -> CheckpointInterface:
    return ModelCheckpointHandler(model, optimizer, filename)
