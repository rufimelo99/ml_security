from abc import ABC, abstractmethod
from typing import List, Union

import torch
import torch.utils


class Attack(ABC):
    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abstractmethod
    def attack(self, **kwargs):
        raise NotImplementedError

    @classmethod
    def __str__(cls):
        return cls.__name__


class AdversarialAttack(Attack):
    def __init__(self, alias: str):
        self.alias = alias

    def attack(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader):
        """
        Generates adversarial examples.

        Args:
            model (torch.nn.Module): The model to attack.
            dataloader (torch.utils.data.DataLoader): The dataloader of the dataset.

        Returns:
            torch.Tensor: The adversarial examples.
        """
        raise NotImplementedError

    @staticmethod
    def evaluate(
        adv_examples: Union[List[torch.Tensor], torch.utils.data.DataLoader],
    ):
        """
        Evaluates the model on adversarial examples.

        Args:
            model (torch.nn.Module): The model to evaluate.
            adv_examples (list): The adversarial examples.

        Returns:
            float: The accuracy on adversarial examples.
        """
        assert isinstance(adv_examples, list) or isinstance(
            adv_examples, torch.utils.data.DataLoader
        ), "adv_examples must be a list or a DataLoader."
        assert len(adv_examples[0]) == 3, "adv_examples must be a list of tuples."
        correct = 0
        total = len(adv_examples)
        for original_target, pred, _ in adv_examples:
            if original_target == pred:
                correct += 1
        return correct / total


class InferenceAttack(Attack):
    def __init__(self, alias: str):
        self.alias = alias

    def attack(self, attack_model: torch.nn.Module, **kwargs):
        """
        Performs the Membership Inference Attack on the model.

        Args:
            attack_model (torch.nn.Module): The attack model.
            kwargs: Additional arguments.

        Returns:
            torch.nn.Module: The attack model.
        """
        raise NotImplementedError

    def evaluate(self, attack_model: torch.nn.Module, **kwargs):
        """
        Evaluates the model with the adversarial examples.

        Args:
            attack_model (torch.nn.Module): The attack model.
            kwargs: Additional arguments.

        Returns:
            float: The accuracy of the model.
        """
        raise NotImplementedError
