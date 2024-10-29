from typing import Callable, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ml_security.attacks.base import InferenceAttack
from ml_security.logger import logger


@torch.no_grad()
def get_confidence_scores(
    model, data_loader: DataLoader, device: torch.device
) -> np.ndarray:
    """
    Computes the confidence scores for a given model across the dataset provided by the data loader,
    indicating the model's certainty for its predictions.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model to use for generating predictions. Typically a classifier.
    data_loader : torch.utils.data.DataLoader
        DataLoader that supplies batches of input data for evaluation.
    device : torch.device
        The device (CPU or GPU) on which the model and computations will be executed.

    Returns
    -------
    np.ndarray
        An array of confidence scores corresponding to the model's predictions on the input data.
    """
    model.eval()
    confidence_scores = []
    for data, target in tqdm(data_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        confidence_scores.append(F.softmax(output, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(confidence_scores)


class ExampleAttackModel(nn.Module):
    """
    A simple feedforward neural network model for demonstrating adversarial attack techniques.

    This model consists of one hidden layer with 64 units and a linear output layer,
    using ReLU activation for the hidden layer and sigmoid activation for the output.

    Parameters
    ----------
    input_dim : int, optional
        The dimension of the input features. Default is 1.
    """

    def __init__(self, input_dim: int = 1):
        super(ExampleAttackModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)  # First fully connected layer
        self.fc2 = nn.Linear(64, 1)  # Output layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor, expected to be of shape (N, input_dim),
                              where N is the batch size.

        Returns:
            torch.Tensor: The output tensor after applying the forward pass.
                          The output will have values in the range [0, 1].
        """
        if x.dim() == 1:
            x = x.unsqueeze(1)  # Adjust dimensions if input is 1D
        x = F.relu(self.fc1(x))  # Apply ReLU activation
        x = torch.sigmoid(self.fc2(x))  # Apply sigmoid activation for output
        return x


class MembershipInferenceAttack(InferenceAttack):
    """
    A class implementing the Membership Inference Attack, which determines
    whether a specific instance was part of the training dataset of a machine learning model.

    This attack utilizes a model to infer membership based on the confidence scores
    of predictions made by the target model.
    """

    def __init__(
        self,
        train_loader: DataLoader,
        holdout_loader: DataLoader,
        model: nn.Module,
        device: torch.device,
        get_confidence_scores_fn: Optional[
            Callable[[torch.nn.Module, DataLoader, torch.device], np.ndarray]
        ] = None,
        batch_size: int = 64,
    ) -> None:
        """
        Initializes the Membership Inference Attack.

        Parameters
        ----------
        train_loader : DataLoader
            The DataLoader containing the training dataset used for the model.
        holdout_loader : DataLoader
            The DataLoader containing the holdout dataset (potentially containing unseen examples).
        model : nn.Module
            The neural network model to attack.
        device : torch.device
            The device (CPU or GPU) on which the attack computations are performed.
        get_confidence_scores_fn : Optional[Callable], optional
            A callable function to retrieve confidence scores from the model.
            If None, a default implementation will be used.
        batch_size : int, optional
            The number of samples per batch during the attack. Default is 64.
        """
        super().__init__(alias="MembershipInferenceAttack")
        self.device = device
        self.model = model
        self.attack_model = ExampleAttackModel()
        if not get_confidence_scores_fn:
            get_confidence_scores_fn = get_confidence_scores
            logger.info("Using default get_confidence_scores function")
        else:
            logger.warning(
                "Using custom get_confidence_scores function. Make sure it matches the attacker model input."
            )
        self.get_confidence_scores_fn = get_confidence_scores_fn
        self.batch_size = batch_size
        self.attack_loader, self.attack_labels = self.create_attack_dataloader(
            train_loader=train_loader,
            holdout_loader=holdout_loader,
            model=model,
            device=device,
            get_confidence_scores=get_confidence_scores_fn,
            batch_size=batch_size,
        )

    @classmethod
    def create_attack_dataloader(
        cls,
        train_loader: DataLoader,
        holdout_loader: DataLoader,
        model: nn.Module,
        device: torch.device,
        get_confidence_scores: Callable[
            [torch.nn.Module, DataLoader, torch.device], np.ndarray
        ] = get_confidence_scores,
        batch_size: int = 64,
    ) -> Union[DataLoader, np.ndarray]:
        """
        Creates a DataLoader for the attack model, combining the training and holdout datasets
        to facilitate membership inference.

        Parameters
        ----------
        train_loader : DataLoader
            The DataLoader for the training data, containing examples that the model has seen.
        holdout_loader : DataLoader
            The DataLoader for the holdout data, containing examples that the model has not seen.
        model : nn.Module
            The neural network model to use for generating confidence scores.
        device : torch.device
            The device (CPU or GPU) on which the computations will be performed.
        get_confidence_scores : Callable, optional
            A function to obtain confidence scores from the model.
            Defaults to the provided function.
        batch_size : int, optional
            The number of samples per batch for the DataLoader. Default is 64.

        Returns
        -------
        Union[DataLoader, np.ndarray]
            A DataLoader for the attack model and the corresponding labels for the attack model.
        """
        # Gets confidence scores for both train and holdout sets
        train_confidence_scores = get_confidence_scores(model, train_loader, device)
        holdout_confidence_scores = get_confidence_scores(model, holdout_loader, device)

        # Label the samples: 1 for training data, 0 for holdout data
        train_labels = np.ones(len(train_confidence_scores))
        holdout_labels = np.zeros(len(holdout_confidence_scores))

        # Creates the dataset for the attack model.
        attack_data = np.concatenate(
            (train_confidence_scores, holdout_confidence_scores), axis=0
        )
        attack_labels = np.concatenate((train_labels, holdout_labels), axis=0)

        # Prepares data for the attack model.
        attack_dataset = TensorDataset(
            torch.Tensor(attack_data), torch.Tensor(attack_labels)
        )
        attack_loader = DataLoader(attack_dataset, batch_size=batch_size, shuffle=True)

        return attack_loader, attack_labels

    def attack(
        self,
        attack_model: nn.Module,
        epochs: int = 10,
        lr: float = 0.01,
        **kwargs,
    ) -> nn.Module:
        """
        Performs the Membership Inference Attack by training an attack model on the provided dataset.

        This method optimizes the attack model's parameters through a specified number of epochs
        using a given learning rate to effectively infer membership.

        Parameters
        ----------
        attack_model : nn.Module
            The neural network model used for performing the membership inference attack.
        epochs : int, optional
            The number of training epochs for the attack model. Default is 10.
        lr : float, optional
            The learning rate for optimizing the attack model's parameters. Default is 0.01.
        **kwargs : keyword arguments
            Additional parameters for further customization of the attack training process.

        Returns
        -------
        nn.Module
            The trained attack model after completion of the training process.
        """
        # Initialize the attack model.
        attack_model = attack_model.to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(attack_model.parameters(), lr=lr)

        # Trains the attack model.
        attack_model.train()

        for epoch in tqdm(range(epochs), desc="Training attack model"):
            for data, target in tqdm(self.attack_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = attack_model(data)
                loss = criterion(output, target.unsqueeze(1))
                loss.backward()
                optimizer.step()

            if epoch % 10 == 0:
                logger.info("Finished epoch", epoch=epoch, loss=loss.item())

        return attack_model

    def evaluate(
        self,
        attack_model: nn.Module,
    ) -> float:
        """
        Evaluates the performance of the attack model on the provided dataset.

        This method calculates the accuracy of the attack model by comparing its predictions
        against the true labels for the dataset.

        Parameters
        ----------
        attack_model : nn.Module
            The neural network model that performs the membership inference attack and whose performance will be evaluated.

        Returns
        -------
        float
            The accuracy of the attack model, represented as a decimal between 0 and 1,
            indicating the proportion of correct predictions.
        """
        attack_model = attack_model.to(self.device)
        attack_model.eval()

        attack_predictions = []
        with torch.no_grad():
            for data, target in tqdm(self.attack_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = attack_model(data)
                attack_predictions.append(output.cpu().numpy())

        attack_predictions = np.concatenate(attack_predictions)

        # Calculate the accuracy of the attack model.
        attack_accuracy = np.mean((attack_predictions > 0.5) == self.attack_labels)
        logger.info("Attack stats", accuracy=attack_accuracy)

        return attack_accuracy
