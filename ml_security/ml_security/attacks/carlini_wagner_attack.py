from typing import Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

from ml_security.logger import logger
from ml_security.utils.distance import DistanceMetricType, get_distance_metric


def carlini_wagner_attack(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    distance_metric: DistanceMetricType = DistanceMetricType.L2,
    target_labels: Optional[torch.Tensor] = None,
    c: float = 1e-4,
    lr: float = 0.01,
    num_steps: int = 1000,
):
    """
    Perform the Carlini-Wagner attack on a batch of images.

    Parameters:
    - model: The neural network model.
    - images: Original images (input batch).
    - labels: True labels of the original images.
    - device: Device to run the attack on.
    - target_labels: (Optional) Target labels for a targeted attack. If None, the attack is untargeted.
    - c: Confidence parameter to balance the trade-off between perturbation size and attack success.
    - lr: Learning rate for the optimizer.
    - num_steps: Number of iterations for optimization.

    Returns:
    - adv_images: Adversarial examples generated from the original images.
    """
    distance_metric = get_distance_metric(distance_metric)

    images = images.to(device)
    labels = labels.to(device)
    if target_labels is not None:
        target_labels = target_labels.to(device)
    ori_images = images.clone().detach()

    lower_bounds = torch.zeros_like(images).to(device)
    upper_bounds = torch.ones_like(images).to(device)

    # Initializes perturbation delta as a variable with gradient enabled.
    delta = torch.zeros_like(images, requires_grad=True).to(device)

    optimizer = torch.optim.Adam([delta], lr=lr)

    for _ in range(num_steps):
        # Generates thr adversarial image and clamp it to be within [0, 1].
        adv_images = torch.clamp(ori_images + delta, 0, 1)

        # Predicts the class of the adversarial image.
        outputs = model(adv_images)

        # Computes the loss
        if target_labels:
            # If target_labels is provided, the attack is targeted.
            # It maximizes the logit for the target label.
            targeted_loss = F.cross_entropy(outputs, target_labels)
            f_loss = -targeted_loss
        else:
            # If target_labels is not provided, the attack is untargeted.
            # It minimizes the logit for the true label.
            true_loss = F.cross_entropy(outputs, labels)
            f_loss = true_loss

        # Minimises perturbation size with L2 norm and add f_loss.
        # l2_loss = torch.norm(delta.view(delta.size(0), -1), p=2, dim=1).mean()
        l2_loss = distance_metric(delta.view(delta.size(0), -1)).mean()

        loss = c * f_loss + l2_loss

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Projects delta to be within bounds if necessary.
        delta.data = torch.clamp(
            delta, lower_bounds - ori_images, upper_bounds - ori_images
        )
    # Generates final adversarial examples, clamped to be within [0, 1].
    adv_images = torch.clamp(ori_images + delta, 0, 1).detach()
    return adv_images


def test_cw_attack(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    c: float,
    lr: float,
    num_steps: int,
    device: torch.device,
    batch_size: int,
):
    correct = 0
    adv_examples = []

    for data, target in tqdm(test_loader):
        data, target = data.to(device), target.to(device)

        # Generates adversarial example.
        perturbed_data = carlini_wagner_attack(
            model, data, target, device, c=c, lr=lr, num_steps=num_steps
        )

        # Re-classify the perturbed image.
        output = model(perturbed_data)

        # Gets the index of the max log-probability.
        final_pred = output.max(1, keepdim=True)[1]
        correct += final_pred.eq(target.view_as(final_pred)).sum().item()

        # Gets the adversarial examples when it is misclassified.
        adv_idxs = final_pred.ne(target.view_as(final_pred)).view(-1)
        for i in range(len(adv_idxs)):
            if not adv_idxs[i]:
                adv_ex = perturbed_data[i].squeeze().detach().cpu().numpy()
                adv_examples.append((target[i], final_pred[i].item(), adv_ex))

    final_acc = correct / (float(len(test_loader)) * batch_size)
    logger.info("Final Accuracy", final_acc=final_acc)
    return final_acc, adv_examples
