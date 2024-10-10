"""
Fast Gradient Sign Attack (FGSM)
The Fast Gradient Sign Attack (FGSM), introduced by Goodfellow et al. in Explaining and Harnessing Adversarial Examples, is one of the earliest and most widely-used adversarial attacks. This method is both powerful and intuitive, exploiting how neural networks learn by using gradients.

Neural networks typically adjust their internal weights to minimize a loss function during training, based on gradients computed via backpropagation. FGSM, however, takes the opposite approach: instead of altering the model’s weights, it perturbs the input data to maximize the loss.

In practice, FGSM computes the gradient of the loss with respect to the input data and adjusts the input in the direction that increases the loss. By making minimal but strategic changes to the input, the attack can significantly degrade the model’s performance, effectively "fooling" it into making incorrect predictions.

This technique is widely used to test the robustness of models and study their vulnerabilities to adversarial examples. 

"""

from typing import List, Union

import torch

from ml_security.logger import logger


def fgsm_attack(
    image: torch.Tensor, epsilon: float, data_grad: torch.Tensor
) -> torch.Tensor:
    """
    Fast Gradient Sign Method.

    Args:
        image (torch.Tensor): The original input image.
        epsilon (float): The epsilon value to use for the attack. Corresponds to the magnitude of the perturbation.
        data_grad (torch.Tensor): The gradient of the loss with respect to the input.

    Returns:
        torch.Tensor: The perturbed image.
    """
    # Collects the element-wise sign of the data gradient.
    data_grad_sign = data_grad.sign()

    # Creates the perturbed image by adjusting each pixel of the input image.
    perturbed_image = image + epsilon * data_grad_sign

    # Adds clipping to maintain [0,1] range.
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    return perturbed_image


def denorm(
    batch: Union[torch.Tensor, List[torch.Tensor]],
    mean: Union[torch.Tensor, List[float]],
    std: Union[torch.Tensor, List[float]],
) -> torch.Tensor:
    """
    Convert a batch of tensors to their original scale.

    Args:
        batch (Union[torch.Tensor, List[torch.Tensor]]): Batch of normalized tensors.
        mean (Union[torch.Tensor, List[float]]): Mean used for normalization.
        std (Union[torch.Tensor, List[float]]): Standard deviation used for normalization.

    Returns:
        torch.Tensor: batch of tensors without normalization applied to them.
    """
    if isinstance(mean, list):
        logger.error("Mean is a list. Should be a tensor.")
    if isinstance(std, list):
        logger.error("Standard deviation is a list. Should be a tensor.")

    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)
