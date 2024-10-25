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