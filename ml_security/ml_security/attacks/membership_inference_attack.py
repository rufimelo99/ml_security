from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


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
    confidence_scores = []
    with torch.no_grad():
        for data, target in tqdm(data_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            confidence_scores.append(F.softmax(output, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(confidence_scores)


def create_attack_dataloader(
    train_loader: DataLoader,
    holdout_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
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
    attack_loader = DataLoader(attack_dataset, batch_size=4, shuffle=True)

    return attack_loader, attack_labels
