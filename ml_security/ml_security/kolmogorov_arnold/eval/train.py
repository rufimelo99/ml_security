import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm

from ml_security.kolmogorov_arnold.eval.utils import (
    CNN,
    CNNKAN,
    plot_results,
    save_results,
)
from ml_security.datasets.datasets import (
    DATASET_REGISTRY,
    DatasetType,
    create_dataloader,
)
from ml_security.logger import logger
from ml_security.utils.utils import get_device, set_seed

set_seed(42)
DEVICE = get_device()

HIDDEN_SIZE = 100
LR = 1e-3


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    total_loss /= len(train_loader.dataset)
    correct /= len(train_loader.dataset)
    return total_loss, correct


def evaluate(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.CrossEntropyLoss()(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    correct /= len(test_loader.dataset)
    return test_loss, correct


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
        ],
    )
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    dataset = DatasetType[args.dataset]
    dataset_info = DATASET_REGISTRY[dataset]

    epochs = args.epochs

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )
    if dataset_info.origin == "TORCHVISION":
        trainloader = create_dataloader(
            dataset=dataset, batch_size=64, train=True, transformation=transform
        )
        valloader = create_dataloader(
            dataset=dataset, batch_size=64, train=False, transformation=transform
        )
    else:
        raise ValueError("Unknown dataset origin.")

    classic_cnn = CNN()
    classic_cnn.to(DEVICE)
    optimizer = optim.AdamW(classic_cnn.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    criterion = nn.CrossEntropyLoss()

    classic_cnn_train_losses = []
    classic_cnn_val_losses = []
    classic_cnn_train_accuracies = []
    classic_cnn_val_accuracies = []

    for epoch in tqdm(range(1, epochs + 1)):
        loss, acc = train(classic_cnn, DEVICE, trainloader, optimizer, epoch)
        classic_cnn_train_losses.append(loss)
        classic_cnn_train_accuracies.append(acc)
        logger.info(f"Epoch {epoch} - Train Loss: {loss} - Train Acc: {acc}")
        loss, acc = evaluate(classic_cnn, DEVICE, valloader)
        classic_cnn_val_losses.append(loss)
        classic_cnn_val_accuracies.append(acc)

    kan_cnn = CNNKAN()
    kan_cnn.to(DEVICE)
    optimizer = optim.AdamW(kan_cnn.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    criterion = nn.CrossEntropyLoss()

    kan_cnn_train_losses = []
    kan_cnn_val_losses = []
    kan_cnn_train_accuracies = []
    kan_cnn_val_accuracies = []

    for epoch in tqdm(range(1, epochs + 1)):
        loss, acc = train(kan_cnn, DEVICE, trainloader, optimizer, epoch)
        kan_cnn_train_losses.append(loss)
        kan_cnn_train_accuracies.append(acc)
        loss, acc = evaluate(kan_cnn, DEVICE, valloader)
        kan_cnn_val_losses.append(loss)
        kan_cnn_val_accuracies.append(acc)

    # Save models
    directory = "ml_security/kolmogorov_arnold/eval/cnn/"

    # Create repo for experience dataset
    dataset_dir = directory + args.dataset + "/"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    plot_results(
        classic_cnn_train_losses,
        classic_cnn_train_accuracies,
        classic_cnn_val_losses,
        classic_cnn_val_accuracies,
        kan_cnn_train_losses,
        kan_cnn_train_accuracies,
        kan_cnn_val_losses,
        kan_cnn_val_accuracies,
        dataset_dir + "results.png",
    )

    save_results(
        classic_cnn_train_losses,
        classic_cnn_train_accuracies,
        classic_cnn_val_losses,
        classic_cnn_val_accuracies,
        kan_cnn_train_losses,
        kan_cnn_train_accuracies,
        kan_cnn_val_losses,
        kan_cnn_val_accuracies,
        dataset_dir + "results.csv",
    )

    torch.save(classic_cnn.state_dict(), dataset_dir + "classic_cnn.pth")
    torch.save(kan_cnn.state_dict(), dataset_dir + "kan_cnn.pth")
