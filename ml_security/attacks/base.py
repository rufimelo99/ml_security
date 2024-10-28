from abc import ABC, abstractmethod
from typing import List, Union

import torch
import torch.utils


class Attack(ABC):
    """
    Abstract base class for implementing various adversarial attack methods.

    This class defines the interface for adversarial attacks, specifying
    the required methods that any derived attack classes must implement.
    It provides a foundation for creating and managing different attack
    strategies against machine learning models.

    Attributes
    ----------
    None
    """

    @abstractmethod
    def __init__(self):
        """
        Initializes the Attack instance.

        This method should be overridden by subclasses to initialize any
        parameters specific to the attack strategy being implemented.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError

    @abstractmethod
    def attack(self, attack_model: torch.nn.Module, **kwargs):
        """
        Executes the adversarial attack on the specified model.

        This method should be overridden by subclasses to define how the
        attack is performed on a given model. The implementation may
        involve manipulating input data to generate adversarial examples.

        Parameters
        ----------
        attack_model : torch.nn.Module
            The model to be attacked, which will process input data and produce outputs.
        **kwargs : additional keyword arguments
            Additional parameters that may be required for the specific attack implementation.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError

    @classmethod
    def __str__(cls):
        """
        Returns a string representation of the Attack class.

        This method provides the class name as a string when the class is printed.

        Returns
        -------
        str
            The name of the class.
        """
        return cls.__name__


class AdversarialAttack(Attack):
    """
    Base class for implementing adversarial attacks on machine learning models.

    This class serves as a foundation for specific adversarial attack implementations.
    It provides common attributes and methods for managing the generation and evaluation
    of adversarial examples, while enforcing the structure through an abstract base class.

    Attributes
    ----------
    alias : str
        A string identifier for the type of adversarial attack.
    """

    def __init__(self, alias: str):
        """
        Initializes the AdversarialAttack instance.

        Parameters
        ----------
        alias : str
            A string identifier that represents the specific attack type
            being implemented.
        """
        self.alias = alias

    def attack(
        self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, **kwargs
    ):
        """
        Generates adversarial examples from the input data.

        This method must be overridden by subclasses to define how adversarial examples
        are created from the input data provided by the dataloader.

        Parameters
        ----------
        model : torch.nn.Module
            The model to attack, which processes the input data to produce predictions.
        dataloader : torch.utils.data.DataLoader
            The DataLoader that supplies batches of input data for generating adversarial examples.
        **kwargs : additional keyword arguments
            Additional parameters that may be required for the specific attack implementation.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.

        Returns
        -------
        torch.Tensor
            The generated adversarial examples.
        torch.Tensor
            The original examples (if needed for comparison).
        """
        raise NotImplementedError

    @staticmethod
    def evaluate(
        adv_examples: Union[List[torch.Tensor], torch.utils.data.DataLoader],
    ) -> float:
        """
        Evaluates the model's performance on adversarial examples.

        This method calculates the accuracy of the model when provided with
        adversarial examples. It compares the predicted labels against the
        true labels to determine the proportion of correct predictions.

        Parameters
        ----------
        adv_examples : Union[List[torch.Tensor], torch.utils.data.DataLoader]
            A list or DataLoader containing adversarial examples,
            where each entry is expected to be a tuple containing
            the original target label, the predicted label, and
            the input data.

        Returns
        -------
        float
            The accuracy of the model on the adversarial examples,
            represented as a decimal between 0 and 1.

        Raises
        ------
        AssertionError
            If `adv_examples` is not a list or a DataLoader, or if
            the entries do not contain the expected number of elements.
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
    """
    Abstract base class for implementing inference attacks on machine learning models.

    This class defines the interface for various inference attack strategies, such as
    membership inference attacks. It provides a foundation for implementing specific
    attacks that evaluate model behavior based on input data and corresponding predictions.

    Attributes
    ----------
    alias : str
        A string identifier for the type of inference attack.
    """

    def __init__(self, alias: str):
        """
        Initializes the InferenceAttack instance.

        Parameters
        ----------
        alias : str
            A string identifier that represents the specific inference attack type
            being implemented.
        """
        self.alias = alias

    def attack(self, attack_model: torch.nn.Module, **kwargs):
        """
        Performs the inference attack on the specified model.

        This method must be overridden by subclasses to define how the attack
        is executed, typically involving training an attack model using the
        provided parameters.

        Parameters
        ----------
        attack_model : torch.nn.Module
            The attack model that will be trained or evaluated against the
            target model.
        **kwargs : additional keyword arguments
            Additional parameters that may be required for the specific attack implementation.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.

        Returns
        -------
        torch.nn.Module
            The trained attack model after executing the attack.
        """
        raise NotImplementedError

    def evaluate(self, attack_model: torch.nn.Module, **kwargs):
        """
        Evaluates the attack model using adversarial examples.

        This method must be overridden by subclasses to define how to evaluate
        the attack model, typically by comparing its predictions against known labels.

        Parameters
        ----------
        attack_model : torch.nn.Module
            The attack model to be evaluated against the adversarial examples.
        **kwargs : additional keyword arguments
            Additional parameters that may be required for the specific evaluation implementation.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.

        Returns
        -------
        float
            The accuracy of the attack model based on its performance on the adversarial examples.
        """
        raise NotImplementedError
