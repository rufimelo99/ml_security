import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim

from ml_security.adaptative_network.eval.utils import (
    HybridNet,
    LinearNet,
    classic_training,
    plot_results,
    save_results,
)
from ml_security.datasets.datasets import (
    DATASET_REGISTRY,
    DatasetType,
    create_dataloader,
)
from ml_security.logger import logger
from ml_security.utils import get_device, set_seed

set_seed(42)
DEVICE = get_device()

HIDDEN_SIZE = 100
LR = 1e-3


##############################################


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="MNIST",
        choices=[
            "MNIST",
            "CIFAR10",
            "FASHION_MNIST",
            "IRIS",
            "WINE",
            "BREAST_CANCER",
            "HEART_DISEASE",
            "BANK_MARKETING",
        ],
    )
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    dataset = DatasetType[args.dataset]
    dataset_info = DATASET_REGISTRY[dataset]

    epochs = args.epochs

    if dataset_info.origin == "UCI":
        trainloader, valloader = create_dataloader(
            dataset=dataset, batch_size=64, train=True
        )
    elif dataset_info.origin == "TORCHVISION":
        trainloader = create_dataloader(dataset=dataset, batch_size=64, train=True)
        valloader = create_dataloader(dataset=dataset, batch_size=64, train=False)
    else:
        raise ValueError("Unknown dataset origin.")

    linear_model = LinearNet(
        num_classes=dataset_info.num_classes,
        input_size=dataset_info.input_features,
        hidden_size=HIDDEN_SIZE,
    )
    linear_model.to(DEVICE)
    optimizer = optim.AdamW(linear_model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    criterion = nn.CrossEntropyLoss()

    (
        linear_model,
        linear_train_losses,
        linear_train_accuracies,
        linear_val_losses,
        linear_val_accuracies,
    ) = classic_training(
        trainloader,
        valloader,
        linear_model,
        optimizer,
        criterion,
        scheduler,
        epochs,
        DEVICE,
        input_features=dataset_info.input_features,
    )

    hybrid_model = HybridNet(
        num_classes=dataset_info.num_classes,
        hidden_size=HIDDEN_SIZE,
        input_size=dataset_info.input_features,
    )
    hybrid_model.to(DEVICE)
    optimizer = optim.AdamW(hybrid_model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    criterion = nn.CrossEntropyLoss()

    (
        hybrid_model,
        hybrid_train_losses,
        hybrid_train_accuracies,
        hybrid_val_losses,
        hybrid_val_accuracies,
    ) = classic_training(
        trainloader,
        valloader,
        hybrid_model,
        optimizer,
        criterion,
        scheduler,
        epochs,
        DEVICE,
        input_features=dataset_info.input_features,
    )

    # Save models
    directory = "ml_security/adaptative_network/eval/"

    # Create repo for experience dataset
    dataset_dir = directory + args.dataset + "/"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    torch.save(linear_model.state_dict(), dataset_dir + "linear.pth")
    torch.save(hybrid_model.state_dict(), dataset_dir + "hybrid.pth")

    # Plot results
    plot_results(
        linear_train_losses,
        linear_train_accuracies,
        linear_val_losses,
        linear_val_accuracies,
        hybrid_train_losses,
        hybrid_train_accuracies,
        hybrid_val_losses,
        hybrid_val_accuracies,
        dataset_dir,
    )

    save_results(
        linear_train_losses,
        linear_train_accuracies,
        linear_val_losses,
        linear_val_accuracies,
        hybrid_train_losses,
        hybrid_train_accuracies,
        hybrid_val_losses,
        hybrid_val_accuracies,
        dataset_dir,
    )

    logger.info("Results saved.", dataset_dir=dataset_dir)
