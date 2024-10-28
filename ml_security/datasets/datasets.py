from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ml_security.datasets.tabular_UCI import from_uciml_to_dataset
from ml_security.logger import logger


class DatasetOrigin(str, Enum):
    UCI = "UCI"
    TORCHVISION = "TORCHVISION"


class DatasetType(str, Enum):
    MNIST = "MNIST"
    CIFAR10 = "CIFAR10"
    FASHION_MNIST = "FASHION_MNIST"
    STL10 = "STL10"
    FGVC_AIIRCRAFT = "FGVC_AIIRCRAFT"
    IRIS = "IRIS"
    WINE = "WINE"
    BREAST_CANCER = "BREAST_CANCER"
    HEART_DISEASE = "HEART_DISEASE"
    BANK_MARKETING = "BANK_MARKETING"


@dataclass
class DatasetInfo:
    origin: DatasetOrigin
    num_classes: int
    input_features: Optional[int]
    uci_id: Optional[int]  # Some datasets may not have a UCI ID


DATASET_REGISTRY = {
    DatasetType.MNIST: DatasetInfo(
        origin=DatasetOrigin.TORCHVISION,
        num_classes=10,
        input_features=28 * 28,
        uci_id=None,
    ),
    DatasetType.CIFAR10: DatasetInfo(
        origin=DatasetOrigin.TORCHVISION,
        num_classes=10,
        input_features=28 * 28,
        uci_id=None,
    ),
    DatasetType.FASHION_MNIST: DatasetInfo(
        origin=DatasetOrigin.TORCHVISION,
        num_classes=10,
        input_features=28 * 28,
        uci_id=None,
    ),
    DatasetType.STL10: DatasetInfo(
        origin=DatasetOrigin.TORCHVISION,
        num_classes=10,
        input_features=28 * 28,
        uci_id=None,
    ),
    DatasetType.FGVC_AIIRCRAFT: DatasetInfo(
        origin=DatasetOrigin.TORCHVISION,
        num_classes=100,
        input_features=28 * 28,
        uci_id=None,
    ),
    DatasetType.IRIS: DatasetInfo(
        origin=DatasetOrigin.UCI, num_classes=3, input_features=4, uci_id=53
    ),
    DatasetType.WINE: DatasetInfo(
        origin=DatasetOrigin.UCI, num_classes=3, input_features=13, uci_id=109
    ),
    DatasetType.BREAST_CANCER: DatasetInfo(
        origin=DatasetOrigin.UCI, num_classes=2, input_features=30, uci_id=17
    ),
    DatasetType.HEART_DISEASE: DatasetInfo(
        origin=DatasetOrigin.UCI, num_classes=2, input_features=13, uci_id=45
    ),
    DatasetType.BANK_MARKETING: DatasetInfo(
        origin=DatasetOrigin.UCI, num_classes=2, input_features=16, uci_id=222
    ),
}

DEFAULT_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

DEFAULT_TRANSFORM_3CH = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ]
)


def create_cv_dataloader(
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
    Create a DataLoader for the specified dataset, suitable for cross-validation.

    This function sets up a DataLoader that can be used for training or testing a model
    on a specified dataset. It supports downloading datasets, applying transformations,
    and splitting datasets for cross-validation.

    Parameters
    ----------
    dataset : DatasetType, optional
        The dataset to load. Defaults to DatasetType.MNIST.
    download : bool, optional
        Whether to download the dataset if it is not found locally. Defaults to True.
    root : str, optional
        The directory where the dataset will be stored or loaded from. Defaults to './data'.
    batch_size : int, optional
        The number of samples to load per batch. Defaults to 64.
    shuffle : bool, optional
        Whether to shuffle the data after every epoch. Defaults to True.
    train : bool, optional
        If True, loads the training data; if False, loads the testing data. Defaults to True.
    transformation : transforms.Compose, optional
        The transformation to apply to the dataset. Defaults to DEFAULT_TRANSFORM.
    max_samples : int, optional
        The maximum number of samples to load from the dataset. If None, loads the entire dataset. Defaults to None.
    split_ratio : List[float], optional
        The ratio to split the dataset into training and testing sets. If None, no splitting is performed.
        Defaults to None.

    Returns
    -------
    DataLoader
        A DataLoader object for the specified dataset, ready for use in model training or evaluation.

    Raises
    ------
    FileNotFoundError
        If the dataset cannot be found and download is set to False.
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


def create_tabular_dataloader(
    dataset: DatasetType = DatasetType.IRIS,
    batch_size: int = 64,
    shuffle: bool = True,
    train: bool = True,
    max_samples: Optional[int] = None,
    split_ratio: Optional[List[float]] = None,
) -> DataLoader:
    """
    Create a DataLoader for the specified tabular dataset.

    This function sets up a DataLoader specifically for tabular datasets, enabling
    training and evaluation workflows. It allows for batch processing, shuffling, and
    splitting the dataset into training and test subsets.

    Parameters
    ----------
    dataset : DatasetType, optional
        The dataset to load. Defaults to DatasetType.IRIS.
    batch_size : int, optional
        The number of samples to load per batch. Defaults to 64.
    shuffle : bool, optional
        Whether to shuffle the data after every epoch. Defaults to True.
    train : bool, optional
        If True, loads the training data; if False, loads the testing data. Defaults to True.
    max_samples : int, optional
        The maximum number of samples to load from the dataset. If None, loads the entire dataset. Defaults to None.
    split_ratio : List[float], optional
        The ratio to split the dataset into training and testing sets. If None, no splitting is performed.
        The ratios must sum to 1. Defaults to None.

    Returns
    -------
    DataLoader
        A DataLoader object for the specified dataset, ready for use in model training or evaluation.

    """
    assert dataset in DatasetType, "Invalid dataset type."

    dataset_info = DATASET_REGISTRY[dataset]

    if dataset_info.origin != DatasetOrigin.UCI:
        logger.error("Tabular datasets outside UCI are not supported.", dataset=dataset)
        raise NotImplementedError("Tabular datasets outside UCI are not supported.")

    if split_ratio:
        split_ratio = split_ratio[0]
        assert 0 < split_ratio < 1, "Split ratio must be a float between 0 and 1."
    else:
        logger.warning("``split_ratio`` not provided. Defaulting to 0.8.")
        split_ratio = 0.8

    ucimlid = dataset_info.uci_id
    assert ucimlid is not None, "UCI ID not provided for the dataset"

    train_dataset, test_dataset = from_uciml_to_dataset(ucimlid, split_ratio)

    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    logger.warning("Returning two DataLoaders from an UCI dataset.")
    return dataloader, test_dataloader


def create_dataloader(
    dataset: DatasetType = DatasetType.MNIST,
    download: bool = True,
    root: str = "./data",
    batch_size: int = 64,
    shuffle: bool = True,
    train: Optional[bool] = True,
    transformation: Optional[transforms.Compose] = DEFAULT_TRANSFORM,
    max_samples: Optional[int] = None,
    split_ratio: Optional[List[float]] = None,
) -> DataLoader:
    """
    Create a DataLoader for the specified dataset.

    This function initializes a DataLoader that handles loading and preprocessing of the specified dataset.
    It provides options for downloading, shuffling, and splitting the dataset for training and testing.

    Parameters
    ----------
    dataset : DatasetType, optional
        The dataset to load. Defaults to DatasetType.MNIST.
    download : bool, optional
        Whether to download the dataset if it is not already available. Defaults to True.
    root : str, optional
        The directory where the dataset will be saved. Defaults to './data'.
    batch_size : int, optional
        The number of samples to load per batch. Defaults to 64.
    shuffle : bool, optional
        Whether to shuffle the data at every epoch. Defaults to True.
    train : bool, optional
        If True, loads the training data; if False, loads the testing data. Defaults to True.
    transformation : Optional[transforms.Compose], optional
        A series of transformations to apply to the data. Defaults to DEFAULT_TRANSFORM.
    max_samples : int, optional
        The maximum number of samples to load from the dataset. If None, loads the entire dataset. Defaults to None.
    split_ratio : List[float], optional
        A list of ratios for splitting the dataset into training and testing sets.
        If None, no splitting is performed. The ratios must sum to 1. Defaults to None.

    Returns
    -------
    DataLoader
        A DataLoader object configured for the specified dataset, ready for use in model training or evaluation.

    Raises
    ------
    ValueError
        If the split_ratio does not sum to 1 or if invalid ratios are provided.
    """

    assert dataset in DatasetType, "Invalid dataset type."

    dataset_info = DATASET_REGISTRY[dataset]
    if dataset_info.origin == DatasetOrigin.TORCHVISION:
        return create_cv_dataloader(
            dataset=dataset,
            download=download,
            root=root,
            batch_size=batch_size,
            shuffle=shuffle,
            train=train,
            transformation=transformation,
            max_samples=max_samples,
            split_ratio=split_ratio,
        )
    elif dataset_info.origin == DatasetOrigin.UCI:
        return create_tabular_dataloader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            train=train,
            max_samples=max_samples,
            split_ratio=split_ratio,
        )
    else:
        logger.error("Unsupported dataset origin.", dataset=dataset)
        raise ValueError(f"Unsupported dataset origin: {dataset_info.origin}")
