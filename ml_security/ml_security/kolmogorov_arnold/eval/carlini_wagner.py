# Implement carlini wagner attack
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ml_security.datasets.datasets import (
    DATASET_REGISTRY,
    DEFAULT_TRANSFORM_3CH,
    DatasetType,
    create_dataloader,
)
from ml_security.kolmogorov_arnold.eval.utils import CNN, CNNKAN
from ml_security.logger import logger

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def carlini_wagner_attack(
    model, images, labels, target_labels=None, c=1e-4, lr=0.01, num_steps=1000
):
    """
    Perform the Carlini-Wagner attack on a batch of images.

    Parameters:
    - model: The neural network model.
    - images: Original images (input batch).
    - labels: True labels of the original images.
    - target_labels: (Optional) Target labels for a targeted attack. If None, the attack is untargeted.
    - c: Confidence parameter to balance the trade-off between perturbation size and attack success.
    - lr: Learning rate for the optimizer.
    - num_steps: Number of iterations for optimization.

    Returns:
    - adv_images: Adversarial examples generated from the original images.
    """
    images = images.to(DEVICE)
    labels = labels.to(DEVICE)
    if target_labels is not None:
        target_labels = target_labels.to(DEVICE)
    ori_images = images.clone().detach()

    lower_bounds = torch.zeros_like(images).to(DEVICE)
    upper_bounds = torch.ones_like(images).to(DEVICE)

    # Initialize perturbation delta as a variable with gradient enabled
    delta = torch.zeros_like(images, requires_grad=True).to(DEVICE)

    # Optimizer for perturbation delta
    optimizer = torch.optim.Adam([delta], lr=lr)

    for step in range(num_steps):
        # Generate adversarial image
        adv_images = torch.clamp(ori_images + delta, 0, 1)

        # Forward pass for predictions
        outputs = model(adv_images)

        # Compute the loss
        if target_labels:  # Targeted attack
            # Targeted loss: Maximize the logit for the target label
            targeted_loss = F.cross_entropy(outputs, target_labels)
            f_loss = -targeted_loss
        else:  # Untargeted attack
            # Untargeted loss: Minimize the logit for the true label
            true_loss = F.cross_entropy(outputs, labels)
            f_loss = true_loss

        # Minimize perturbation size with L2 norm and add f_loss
        l2_loss = torch.norm(delta.view(delta.size(0), -1), p=2, dim=1).mean()
        loss = c * f_loss + l2_loss

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Project delta to be within bounds if necessary
        delta.data = torch.clamp(
            delta, lower_bounds - ori_images, upper_bounds - ori_images
        )

        if step % 100 == 0:
            print(
                f"Step [{step}/{num_steps}], Loss: {loss.item():.4f}, f_loss: {f_loss.item():.4f}, L2: {l2_loss.item():.4f}"
            )

    # Generate final adversarial examples
    adv_images = torch.clamp(ori_images + delta, 0, 1).detach()
    return adv_images
