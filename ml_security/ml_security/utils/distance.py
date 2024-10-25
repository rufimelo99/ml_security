from abc import ABC
from enum import Enum
from typing import Optional

import torch

from ml_security.logger import logger


class DistanceMetricType(Enum):
    L1 = "l1"
    L2 = "l2"
    LINF = "linf"


class DistanceMetric(ABC):
    @classmethod
    def distance(
        cls, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        pass

    @classmethod
    def __call__(
        cls, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if y is None:
            y = torch.zeros_like(x)
        return cls.distance(x, y)


class L1Distance(DistanceMetric):
    @classmethod
    def distance(cls, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.abs(x - y), dim=1)


class L2Distance(DistanceMetric):
    @classmethod
    def distance(cls, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.norm(x - y, p=2, dim=1)


class LinfDistance(DistanceMetric):
    @classmethod
    def distance(cls, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.max(torch.abs(x - y), dim=1).values


def get_distance_metric(distance_metric: DistanceMetricType) -> DistanceMetric:
    if distance_metric == DistanceMetricType.L1:
        return L1Distance()
    elif distance_metric == DistanceMetricType.L2:
        return L2Distance()
    elif distance_metric == DistanceMetricType.LINF:
        return LinfDistance()
    else:
        logger.error("Invalid distance metric", distance_metric=distance_metric)
        raise ValueError(f"Invalid distance metric: {distance_metric}")
