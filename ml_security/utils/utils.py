import os

import numpy as np
import torch

from ml_security.logger import logger


def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility across various libraries.

    This function sets the seed for PyTorch, NumPy, and the Python hash function
    to ensure that results can be reproduced across multiple runs of the code.
    It is particularly useful in experiments involving stochastic processes.

    Args:
        seed (int, optional): The seed value to set. Default is 42.

    Notes:
        - Setting the seed does not guarantee full reproducibility in every scenario,
          especially in multi-threaded environments or when using non-deterministic algorithms.
        - If using CUDA, ensure that you also set `torch.backends.cudnn.deterministic = True`
          for more reproducible results in models that rely on cuDNN.
        - Use a specific seed value to control randomness and ensure consistent results during experimentation.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(allow_mps: bool = True) -> torch.device:
    """
    Returns the best available device for tensor operations.

    This function checks for available devices in the following order:
    1. CUDA (NVIDIA GPU)
    2. MPS (Apple Silicon GPU) if allowed
    3. CPU

    The selected device is logged for informational purposes.

    Args:
        allow_mps (bool, optional): If True, allow the use of Apple Silicon GPU (MPS).
            Default is True.

    Returns:
        torch.device: The device selected for computations.

    Notes:
        - This function utilizes PyTorch's built-in capabilities to detect available devices.
        - If neither CUDA nor MPS is available, the function will default to the CPU.
        - The function logs the name of the device being used, which can help in debugging
          and ensuring that the expected hardware is being utilized.
    """
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
