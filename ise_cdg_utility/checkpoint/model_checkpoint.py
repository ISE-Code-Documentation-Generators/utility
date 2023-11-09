from abc import ABC
import dataclasses
import typing

import torch
from torch import nn
from torch import optim

from ise_cdg_utility.checkpoint import CheckpointInterface



class BaseModelCheckpointHandler(CheckpointInterface, ABC):
    @dataclasses.dataclass
    class ModelCheckpoint:
        model: typing.Any
        optimizer: typing.Any

        @property
        def context(self):
            return dataclasses.asdict(self)
    

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer,) -> None:
        self.model = model
        self.optimizer = optimizer

    def get_checkpoint(self):
        return self.ModelCheckpoint(model=self.model.state_dict(), optimizer=self.optimizer.state_dict())

    def _save_checkpoint(self, filename: str):
        print(f'Saving Checkpoint in {filename}...')
        torch.save(self.get_checkpoint().context, filename)

    def _load_checkpoint(self, filename: str):
        print(f'Loading Checkpoint from {filename}...')
        checkpoint = self.ModelCheckpoint(**torch.load(filename))
        self.model.load_state_dict(checkpoint.model)
        self.optimizer.load_state_dict(checkpoint.optimizer)


class ModelCheckpointHandler(BaseModelCheckpointHandler):

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, filename: str):
        super().__init__(model, optimizer)
        self.filename = filename

    def save_checkpoint(self):
        self._save_checkpoint(self.filename)

    def load_checkpoint(self):
        self._load_checkpoint(self.filename)


class InOutModelCheckpointHandler(BaseModelCheckpointHandler):

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, in_filename: str, out_filename: str):
        super().__init__(model, optimizer)
        self.in_filename = in_filename
        self.out_filename = out_filename

    def save_checkpoint(self):
        self._save_checkpoint(self.in_filename)

    def load_checkpoint(self):
        self._load_checkpoint(self.out_filename)
