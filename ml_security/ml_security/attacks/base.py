from abc import ABC, abstractmethod

import torch


class Attack(ABC):
    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abstractmethod
    def attack(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader):
        raise NotImplementedError

    @classmethod
    def __str__(cls):
        return cls.__name__
