import argparse
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from ml_security.attacks.carlini_wagner_attack import CarliniWagnerAttack
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
    logger.info("Evaluating Classic CNN with Carlini-Wagner Attack")
    model = CNN()
    model.load_state_dict(
        torch.load("ml_security/kolmogorov_arnold/eval/cnn/CIFAR10/classic_cnn.pth")
    )
    model.to(DEVICE)

    original_acc = test_original(model, valloader)
    logger.info("Original Accuracy", original_acc=original_acc)

    cw_attack = CarliniWagnerAttack(DEVICE, c=1e-4, lr=0.01, num_steps=2)
    adv_examples = cw_attack.attack(model, valloader)
    final_acc = cw_attack.evaluate(adv_examples)

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

    adv_examples = cw_attack.attack(model, valloader)
    final_acc = cw_attack.evaluate(adv_examples)

    directory = f"adv_examples_kan_cw/eps_1.0_alpha_0.01_iters_1000"
    save_adv_examples(adv_examples, directory, max_examples=max_examples)
    save_results(final_acc, epsilon, alpha, iters, directory)

    logger.info("Finished Evaluation")
