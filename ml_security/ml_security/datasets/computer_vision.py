"""
    Contains utility functions for loading Computer Vision datasets.
"""

from enum import Enum
from typing import List, Optional

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ml_security.logger import logger


class DatasetType(str, Enum):
    MNIST = "MNIST"
    CIFAR10 = "CIFAR10"
    FASHION_MNIST = "FASHION_MNIST"
    STL10 = "STL10"
    FGVC_AIIRCRAFT = "FGVC_AIIRCRAFT"


NUM_CLASSES = {
    DatasetType.MNIST: 10,
    DatasetType.CIFAR10: 10,
    DatasetType.FASHION_MNIST: 10,
    DatasetType.STL10: 10,
    DatasetType.FGVC_AIIRCRAFT: 100,
}

DEFAULT_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


def create_dataloader(
    dataset: DatasetType = DatasetType.MNIST,
    download: bool = True,
    root: str = "./data",
    batch_size: int = 64,
    shuffle: bool = True,
    train: bool = True,
    transformation: transforms.Compose = DEFAULT_TRANSFORM,
    max_samples: Optional[int] = None,
    split_ratio: Optional[List[float]] = None,
) -> DataLoader:
    """
    Create a DataLoader for the specified dataset.

    Args:
        dataset (Dataset): The dataset to load.
        transform (torchvision.transforms.Compose, optional): A list of transformations to apply to the data. Defaults to None.
        download (bool, optional): Whether to download the dataset. Defaults to True.
        root (str, optional): The directory to save the dataset. Defaults to './data'.
        batch_size (int, optional): The batch size. Defaults to 64.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        transformation (transforms.Compose, optional): The transformation to apply to the data. Defaults to DEFAULT_TRANSFORM.
        max_samples (int, optional): The maximum number of samples to load. Defaults to 1000.
        split_ratio (List[float], optional): The ratio to split the dataset into training and testing sets. Defaults to None.

    Returns:
        torch.utils.data.DataLoader: A DataLoader for the specified dataset.
    """
    if split_ratio:
        assert len(split_ratio) == 2, "Split ratio must be a list of 2 floats."
        assert sum(split_ratio) == 100, "Split ratio must sum to 100."
        logger.warning("``split_ratio`` is set. Will be returning two DataLoaders.")

    if dataset == DatasetType.MNIST:
        dataset = datasets.MNIST(
            root=root, train=train, download=download, transform=transformation
        )
    elif dataset == DatasetType.CIFAR10:
        dataset = datasets.CIFAR10(
            root=root, train=train, download=download, transform=transformation
        )
    elif dataset == DatasetType.FASHION_MNIST:
        dataset = datasets.FashionMNIST(
            root=root, train=train, download=download, transform=transformation
        )
    elif dataset == DatasetType.STL10:
        dataset = datasets.STL10(
            root=root, train=train, download=download, transform=transformation
        )
    elif dataset == DatasetType.FGVC_AIIRCRAFT:
        dataset = datasets.FGVCAircraft(
            root=root, download=download, transform=transformation
        )
    else:
        logger.error("Unsupported dataset.", dataset=dataset)
        raise ValueError(f"Unsupported dataset: {dataset}")

    if max_samples:
        dataset = torch.utils.data.Subset(dataset, range(max_samples))

    if split_ratio:
        train_dataset, test_dataset = train_test_split(
            dataset, test_size=split_ratio[1], random_state=42
        )
        return DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle
        ), DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
