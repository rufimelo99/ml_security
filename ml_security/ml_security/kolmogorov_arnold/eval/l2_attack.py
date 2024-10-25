import argparse
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
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
from ml_security.utils.utils import get_device, set_seed

set_seed(42)
DEVICE = get_device()
BATCH_SIZE = 64


def l2_pgd_attack(model, images, labels, epsilon, alpha, iters):
    images = images.to(DEVICE)
    labels = labels.to(DEVICE)
    ori_images = images.data

    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)

        # Calculate loss
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()

        # Generate perturbations with the gradient
        grad = images.grad.data

        # Normalize the gradient for L2 attack
        grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1).view(-1, 1, 1, 1)
        grad = grad / (grad_norm + 1e-8)  # Avoid division by zero

        # Update the image with small steps
        adv_images = images + alpha * grad

        # Clip the perturbation to stay within epsilon L2 norm
        perturbation = adv_images - ori_images
        perturbation_norm = torch.norm(
            perturbation.view(perturbation.size(0), -1), dim=1
        ).view(-1, 1, 1, 1)
        perturbation = perturbation * torch.min(
            torch.ones_like(perturbation_norm), epsilon / perturbation_norm
        )

        # Update adversarial image
        images = ori_images + perturbation
        images = torch.clamp(images, -1, 1)  # Keep image in valid range
        images = images.detach()  # Detach the t
    return images


def test_l2_attack(model, test_loader, epsilon, alpha, iters):
    correct = 0
    adv_examples = []

    for data, target in tqdm(test_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)

        # Generate adversarial example
        perturbed_data = l2_pgd_attack(model, data, target, epsilon, alpha, iters)

        # Re-classify the perturbed image
        output = model(perturbed_data)
        final_pred = output.max(1, keepdim=True)[
            1
        ]  # Get the index of the max log-probability
        correct += final_pred.eq(target.view_as(final_pred)).sum().item()

        # get the adversarial examples when it is misclassified
        adv_idxs = final_pred.ne(target.view_as(final_pred)).view(-1)
        for i in range(len(adv_idxs)):
            if not adv_idxs[i]:
                adv_ex = perturbed_data[i].squeeze().detach().cpu().numpy()
                adv_examples.append((target[i], final_pred[i].item(), adv_ex))

    final_acc = correct / (float(len(test_loader)) * BATCH_SIZE)
    logger.info("Final Accuracy", final_acc=final_acc)
    return final_acc, adv_examples


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


def test_cw_attack(model, test_loader, c, lr, num_steps):
    correct = 0
    adv_examples = []

    for data, target in tqdm(test_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)

        # Generate adversarial example
        perturbed_data = carlini_wagner_attack(
            model, data, target, c=c, lr=lr, num_steps=num_steps
        )

        # Re-classify the perturbed image
        output = model(perturbed_data)
        final_pred = output.max(1, keepdim=True)[
            1
        ]  # Get the index of the max log-probability
        correct += final_pred.eq(target.view_as(final_pred)).sum().item()

        # get the adversarial examples when it is misclassified
        adv_idxs = final_pred.ne(target.view_as(final_pred)).view(-1)
        for i in range(len(adv_idxs)):
            if not adv_idxs[i]:
                adv_ex = perturbed_data[i].squeeze().detach().cpu().numpy()
                adv_examples.append((target[i], final_pred[i].item(), adv_ex))

    final_acc = correct / (float(len(test_loader)) * BATCH_SIZE)
    logger.info("Final Accuracy", final_acc=final_acc)
    return final_acc, adv_examples


@torch.no_grad()
def test_original(model, test_loader):
    correct = 0
    for data, target in tqdm(test_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        output = model(data)
        final_pred = output.max(1, keepdim=True)[
            1
        ]  # Get the index of the max log-probability
        correct += final_pred.eq(target.view_as(final_pred)).sum().item()

    final_acc = correct / (float(len(test_loader)) * BATCH_SIZE)
    return final_acc


def get_str_parameters(epsilon, alpha, iters):
    return f"eps_{epsilon}_alpha_{alpha}_iters_{iters}"


def save_adv_examples(adv_examples, directory, max_examples=math.inf):
    if not os.path.exists(directory):
        os.makedirs(directory)
    for i, (target, pred, adv_ex) in enumerate(adv_examples):
        adv_ex = np.transpose(adv_ex, (1, 2, 0))
        adv_ex = (adv_ex + 1) / 2

        # Save the adversarial example
        plt.imsave(f"{directory}/adv_{i}_true_{target}_pred_{pred}.png", adv_ex)

        if i >= max_examples:
            break


def save_results(
    accuracy,
    epsilon,
    alpha,
    iters,
    directory,
):
    results = {
        "accuracy": accuracy,
        "epsilon": epsilon,
        "alpha": alpha,
        "iters": iters,
    }
    with open(directory + "results.json", "a") as f:
        json.dump(results, f)
        f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epsilon", type=float, default=1.0, help="Maximum L2 perturbation"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.01, help="Step size for each iteration"
    )
    parser.add_argument("--iters", type=int, default=40, help="Number of iterations")
    parser.add_argument(
        "--dataset",
        type=str,
        default="CIFAR10",
        choices=[
            "CIFAR10",
        ],
    )
    parser.add_argument(
        "--max_sample", type=int, default=20, help="Number of adv samples to save"
    )
    args = parser.parse_args()

    dataset = DatasetType[args.dataset]
    dataset_info = DATASET_REGISTRY[dataset]

    if dataset_info.origin == "TORCHVISION":
        valloader = create_dataloader(
            dataset=dataset,
            batch_size=64,
            train=False,
            transformation=DEFAULT_TRANSFORM_3CH,
        )
    else:
        raise ValueError("Unknown dataset origin or Unsupported for this experiment.")

    epsilon = args.epsilon
    alpha = args.alpha
    iters = args.iters
    max_examples = args.max_sample

    ##############
    logger.info("Evaluating Classic CNN")
    model = CNN()
    model.load_state_dict(
        torch.load("ml_security/kolmogorov_arnold/eval/cnn/CIFAR10/classic_cnn.pth")
    )
    model.to(DEVICE)

    original_acc = test_original(model, valloader)
    logger.info("Original Accuracy", original_acc=original_acc)

    final_acc, adv_examples = test_l2_attack(model, valloader, epsilon, alpha, iters)
    directory = f"adv_examples/{get_str_parameters(epsilon, alpha, iters)}"
    save_adv_examples(adv_examples, directory, max_examples=max_examples)
    save_results(final_acc, epsilon, alpha, iters, directory)

    ##############
    logger.info("Evaluating KAN CNN")

    model = CNNKAN()
    model.load_state_dict(
        torch.load("ml_security/kolmogorov_arnold/eval/cnn/CIFAR10/kan_cnn.pth")
    )
    model.to(DEVICE)

    original_acc = test_original(model, valloader)
    logger.info("Original Accuracy", original_acc=original_acc)

    final_acc, adv_examples = test_l2_attack(model, valloader, epsilon, alpha, iters)
    directory = f"adv_examples_kan/{get_str_parameters(epsilon, alpha, iters)}"
    save_adv_examples(adv_examples, directory, max_examples=max_examples)
    save_results(final_acc, epsilon, alpha, iters, directory)

    ##############
    logger.info("Evaluating Classic CNN with Carlini-Wagner Attack")
    model = CNN()
    model.load_state_dict(
        torch.load("ml_security/kolmogorov_arnold/eval/cnn/CIFAR10/classic_cnn.pth")
    )
    model.to(DEVICE)

    original_acc = test_original(model, valloader)
    logger.info("Original Accuracy", original_acc=original_acc)

    final_acc, adv_examples = test_cw_attack(
        model, valloader, c=1e-4, lr=0.01, num_steps=1000
    )
    directory = f"adv_examples_cw/eps_1.0_alpha_0.01_iters_1000"
    save_adv_examples(adv_examples, directory, max_examples=max_examples)
    save_results(final_acc, epsilon, alpha, iters, directory)

    ##############

    logger.info("Evaluating KAN CNN with Carlini-Wagner Attack")
    model = CNNKAN()
    model.load_state_dict(
        torch.load("ml_security/kolmogorov_arnold/eval/cnn/CIFAR10/kan_cnn.pth")
    )
    model.to(DEVICE)

    original_acc = test_original(model, valloader)
    logger.info("Original Accuracy", original_acc=original_acc)

    final_acc, adv_examples = test_cw_attack(
        model, valloader, c=1e-4, lr=0.01, num_steps=1000
    )
    directory = f"adv_examples_kan_cw/eps_1.0_alpha_0.01_iters_1000"
    save_adv_examples(adv_examples, directory, max_examples=max_examples)
    save_results(final_acc, epsilon, alpha, iters, directory)

    logger.info("Finished Evaluation")
