import os

import numpy as np
import torch


def set_seed(seed: int = 42):
    """Sets the seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device():
    """Returns the best available device: CUDA (NVIDIA GPU), MPS (Apple Silicon GPU), or CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA (GPU): {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    return device
