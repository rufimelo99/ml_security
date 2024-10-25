import argparse
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

from ml_security.adaptative_network.eval.utils import (
    CIFARCNN,
    CIFARCNNKAN,
    PreActBlock,
    PreActResNet,
    PreActResNetwithKAN,
)
from ml_security.attacks.membership_inference_attack import create_attack_dataloader
from ml_security.datasets.datasets import (
    DATASET_REGISTRY,
    DatasetType,
    create_dataloader,
)
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
        # Check if the adversarial image was classified correctly
        if final_pred.item() == target.item():
            correct += 1

        # if we create a successful adversarial example, save it
        if final_pred.item() != target.item():
            adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
            adv_examples.append((target, final_pred.item(), adv_ex))

    final_acc = correct / float(len(test_loader))
    print(f"Test Accuracy = {final_acc * 100:.2f}%")
    return final_acc, adv_examples


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
            "MNIST",
            "CIFAR10",
            "FASHION_MNIST",
        ],
    )
    args = parser.parse_args()

    dataset = DatasetType[args.dataset]
    dataset_info = DATASET_REGISTRY[dataset]

    if dataset_info.origin == "TORCHVISION":
        valloader = create_dataloader(dataset=dataset, batch_size=64, train=False)
    else:
        raise ValueError("Unknown dataset origin or Unsupported for this experiment.")

    epsilon = args.epsilon
    alpha = args.alpha
    iters = args.iters

    model = CIFARCNN()
    model.load_state_dict(
        torch.load("ml_security/adaptative_network/eval/cnn/CIFAR10/classic_cnn.pth")
    )
    model.to(DEVICE)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    final_acc, adv_examples = test_l2_attack(model, valloader, epsilon, alpha, iters)
    save_adv_examples(adv_examples, "adv_examples", max_examples=5)
    save_results(final_acc, epsilon, alpha, iters, "adv_examples/")

    model = CIFARCNNKAN()
    model.load_state_dict(torch.load("cnn/CIFAR10/kan_cnn.pth"))
    model.to(DEVICE)

    final_acc, adv_examples = test_l2_attack(model, valloader, epsilon, alpha, iters)
    save_adv_examples(adv_examples, "adv_examples_kan", max_examples=5)
    save_results(final_acc, epsilon, alpha, iters, "adv_examples_kan/")
