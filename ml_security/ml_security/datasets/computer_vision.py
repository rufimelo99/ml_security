"""
    Contains utility functions for loading Computer Vision datasets.
"""

from enum import Enum

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
    Returns:
        torch.utils.data.DataLoader: A DataLoader for the specified dataset.
    """
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

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
