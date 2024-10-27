from typing import Callable, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ml_security.attacks.base import Attack
from ml_security.logger import logger


@torch.no_grad()
def get_confidence_scores(
    model, data_loader: DataLoader, device: torch.device
) -> np.ndarray:
    """
    Get the confidence scores for the given model and data loader.

    Args:
        model (torch.nn.Module): The model to use. A classifier in this scenario.
        data_loader (torch.utils.data.DataLoader): The data loader to use.
        device (torch.device): The device to use.

    Returns:
        np.ndarray: The confidence scores.
    """
    model.eval()
    confidence_scores = []
    for data, target in tqdm(data_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        confidence_scores.append(F.softmax(output, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(confidence_scores)


class ExampleAttackModel(nn.Module):
    def __init__(self, input_dim: int = 1):
        super(ExampleAttackModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


class MembershipInferenceAttack(Attack):
    """
    Membership Inference Attack.
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

        Args:
            model (nn.Module): The model to attack.
            device (torch.device): The device to use for the attack.
            attack_model (Optional[nn.Module]): The attack model to use.
        """
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
        Create the DataLoader for the attack model.

        Args:
            train_loader (DataLoader): The DataLoader for the training data.
            holdout_loader (DataLoader): The DataLoader for the holdout data.
            model (nn.Module): The model to use.
            device (torch.device): The device to use.

        Returns:
            DataLoader: The DataLoader for the attack model.
            np.ndarray: The labels for the attack model.
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
    ) -> nn.Module:
        """
        Performs the Membership Inference Attack on the model.

        Args:
            attack_loader (DataLoader): The DataLoader for the attack model.
            epochs (int): The number of epochs to train the attack model.
            lr (float): The learning rate for the attack model.

        Returns:
            nn.Module: The trained attack model.
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
        Evaluates the attack model.

        Args:
            attack_loader (DataLoader): The DataLoader for the attack model.
            attack_labels (np.ndarray): The labels for the attack model.

        Returns:
            float: The accuracy of the attack model.
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
