import os

import numpy as np
import torch

from ml_security.logger import logger


def set_seed(seed: int = 42):
    """Sets the seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(allow_mps: bool = True) -> torch.device:
    """Returns the best available device: CUDA (NVIDIA GPU), MPS (Apple Silicon GPU), or CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA (GPU)", device_name=torch.cuda.get_device_name(0))
    elif torch.backends.mps.is_available() and allow_mps:
        device = torch.device("mps")
        logger.info("Using MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    return device
