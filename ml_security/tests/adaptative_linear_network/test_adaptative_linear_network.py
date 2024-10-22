import numpy as np
import pytest
import torch

from ml_security.adaptative_network.network import HybridLinearKAN


def test_forward():
    input_size = 28 * 28
    output_size = 100
    batch_size = 64
    input_data = np.random.rand(batch_size, input_size)
    input_data = torch.tensor(input_data, dtype=torch.float32)
    model = HybridLinearKAN(input_size, output_size)
    output = model(input_data)
    assert output.shape == (batch_size, output_size)
