import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim

from ml_security.adaptative_network.eval.utils import (
    HybridNet,
    CNNKAN,
    CNN,
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

    from torchvision import datasets, transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  
    ])
    if dataset_info.origin == "TORCHVISION":
        trainloader = create_dataloader(dataset=dataset, batch_size=64, train=True, transformation=transform)
        valloader = create_dataloader(dataset=dataset, batch_size=64, train=False, transformation=transform)
    else:
        raise ValueError("Unknown dataset origin.")


    classic_cnn = CNN(    )
    classic_cnn.to(DEVICE)
    optimizer = optim.AdamW(classic_cnn.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    criterion = nn.CrossEntropyLoss()


    def train(model, device, train_loader, optimizer, epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

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
        print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')


    for epoch in range(1, epochs + 1):
        train(classic_cnn, DEVICE, trainloader, optimizer, epoch)
        evaluate(classic_cnn, DEVICE, valloader)

    # Save models
    directory = "ml_security/adaptative_network/eval/cnn/"

    # Create repo for experience dataset
    dataset_dir = directory + args.dataset + "/"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)