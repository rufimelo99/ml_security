from abc import ABC
from enum import Enum
from typing import Optional

import torch

from ml_security.logger import logger


class DistanceMetricType(Enum):
    """
    Enumeration for distance metric types.

    Attributes
    ----------
    L1 : str
        Represents the L1 distance metric.
    L2 : str
        Represents the L2 distance metric.
    LINF : str
        Represents the Linf (maximum) distance metric.
    """

    L1 = "l1"
    L2 = "l2"
    LINF = "linf"


class DistanceMetric(ABC):
    """
    Abstract base class for distance metrics.

    This class provides a static method to compute the distance between two tensors.
    Derived classes must implement the distance method.

    Methods
    -------
    distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor
        Abstract method to compute the distance between two tensors.

    __call__(x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor
        Computes the distance by calling the distance method.
        If the second tensor is not provided, it defaults to a zero tensor of the same shape as the first tensor.
    """

    @staticmethod
    def distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass

    @classmethod
    def __call__(
        cls, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if y is None:
            y = torch.zeros_like(x)
        return cls.distance(x, y)


class L1Distance(DistanceMetric):
    """
    L1 Distance Metric.

    This class implements the L1 distance metric, which computes the sum of absolute differences
    between the elements of two tensors.

    Methods
    -------
    distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor
        Computes the L1 distance between two tensors.
    """

    @staticmethod
    def distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.abs(x - y), dim=1)


class L2Distance(DistanceMetric):
    """
    L2 Distance Metric.

    This class implements the L2 distance metric, which computes the Euclidean distance between
    two tensors.

    Methods
    -------
    distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor
        Computes the L2 distance between two tensors.
    """

    @staticmethod
    def distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.norm(x - y, p=2, dim=1)


class LinfDistance(DistanceMetric):
    """
    Linf Distance Metric.

    This class implements the Linf distance metric, which computes the maximum absolute difference
    between the elements of two tensors.

    Methods
    -------
    distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor
        Computes the Linf distance between two tensors.
    """

    @staticmethod
    def distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.max(torch.abs(x - y), dim=1).values


def get_distance_metric(distance_metric: DistanceMetricType) -> DistanceMetric:
    """
    Retrieve the appropriate distance metric class based on the provided type.

    Args
    ----
    distance_metric : DistanceMetricType
        The type of distance metric to retrieve.

    Returns
    -------
    DistanceMetric
        An instance of the specified distance metric class.

    Raises
    ------
    ValueError
        If an invalid distance metric type is provided.
    """
    if distance_metric == DistanceMetricType.L1:
        return L1Distance()
    elif distance_metric == DistanceMetricType.L2:
        return L2Distance()
    elif distance_metric == DistanceMetricType.LINF:
        return LinfDistance()
    else:
        logger.error("Invalid distance metric", distance_metric=distance_metric)
        raise ValueError(f"Invalid distance metric: {distance_metric}")
