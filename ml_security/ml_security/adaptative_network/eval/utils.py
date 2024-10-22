import json

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ml_security.adaptative_network.network import HybridLinearKAN
from ml_security.logger import logger


def classic_training(
    trainloader,
    valloader,
    model,
    optimizer,
    criterion,
    scheduler,
    epochs,
    device,
    input_features,
):
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    model.to(device)
    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss = 0
        total_accuracy = 0
        with tqdm(trainloader) as pbar:
            for i, (images, labels) in enumerate(pbar):
                if images.shape[1] == 3:
                    # convert to grayscale
                    images = images.mean(dim=1, keepdim=True)
                images = images.view(-1, input_features).to(device)
                optimizer.zero_grad()
                output = model(images)
                softmax = torch.nn.Softmax(dim=1)
                output = softmax(output)
                loss = criterion(output, labels.to(device))
                loss.backward()
                optimizer.step()
                accuracy = (output.argmax(dim=1) == labels.to(device)).float().mean()
                pbar.set_postfix(
                    loss=loss.item(),
                    accuracy=accuracy.item(),
                    lr=optimizer.param_groups[0]["lr"],
                )
                total_loss += loss.item()
                total_accuracy += accuracy.item()

        train_losses.append(total_loss / len(trainloader))
        train_accuracies.append(total_accuracy / len(trainloader))

        model.eval()
        val_loss = 0
        val_accuracy = 0
        with torch.no_grad():
            for images, labels in valloader:
                if images.shape[1] == 3:
                    # convert to grayscale
                    images = images.mean(dim=1, keepdim=True)
                images = images.view(-1, input_features).to(device)
                output = model(images)
                val_loss += criterion(output, labels.to(device)).item()
                val_accuracy += (
                    (output.argmax(dim=1) == labels.to(device)).float().mean().item()
                )

        val_loss /= len(valloader)
        val_accuracy /= len(valloader)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        scheduler.step()

    logger.info(
        "Finished epoch.",
        epoch=epoch + 1,
        val_loss=val_loss,
        val_accuracy=val_accuracy,
    )

    return model, train_losses, train_accuracies, val_losses, val_accuracies


def plot_results(
    linear_train_losses,
    linear_train_accuracies,
    linear_val_losses,
    linear_val_accuracies,
    hybrid_train_losses,
    hybrid_train_accuracies,
    hybrid_val_losses,
    hybrid_val_accuracies,
    directory,
):
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    # Define colors for each model
    colors = {"KAN": "blue", "Linear": "green", "Hybrid": "red"}

    # Plot losses
    ax[0].plot(
        linear_train_losses,
        label="Linear train loss",
        color=colors["Linear"],
        linestyle="-",
    )
    ax[0].plot(
        linear_val_losses,
        label="Linear val loss",
        color=colors["Linear"],
        linestyle="--",
    )
    ax[0].plot(
        hybrid_train_losses,
        label="Hybrid train loss",
        color=colors["Hybrid"],
        linestyle="-",
    )
    ax[0].plot(
        hybrid_val_losses,
        label="Hybrid val loss",
        color=colors["Hybrid"],
        linestyle="--",
    )
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend()

    # Plot accuracies
    ax[1].plot(
        linear_train_accuracies,
        label="Linear train accuracy",
        color=colors["Linear"],
        linestyle="-",
    )
    ax[1].plot(
        linear_val_accuracies,
        label="Linear val accuracy",
        color=colors["Linear"],
        linestyle="--",
    )
    ax[1].plot(
        hybrid_train_accuracies,
        label="Hybrid train accuracy",
        color=colors["Hybrid"],
        linestyle="-",
    )
    ax[1].plot(
        hybrid_val_accuracies,
        label="Hybrid val accuracy",
        color=colors["Hybrid"],
        linestyle="--",
    )
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()

    fig.savefig(directory + "results.png")


def save_results(
    linear_train_losses,
    linear_train_accuracies,
    linear_val_losses,
    linear_val_accuracies,
    hybrid_train_losses,
    hybrid_train_accuracies,
    hybrid_val_losses,
    hybrid_val_accuracies,
    directory,
):
    results = {
        "linear_train_losses": linear_train_losses,
        "linear_train_accuracies": linear_train_accuracies,
        "linear_val_losses": linear_val_losses,
        "linear_val_accuracies": linear_val_accuracies,
        "hybrid_train_losses": hybrid_train_losses,
        "hybrid_train_accuracies": hybrid_train_accuracies,
        "hybrid_val_losses": hybrid_val_losses,
        "hybrid_val_accuracies": hybrid_val_accuracies,
    }
    with open(directory + "results.json", "w") as f:
        json.dump(results, f)


class HybridNet(torch.nn.Module):
    def __init__(
        self,
        num_classes=10,
        input_size=28 * 28,
        hidden_size=100,
    ):
        super(HybridNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.liner_kan1 = HybridLinearKAN(self.input_size, self.hidden_size)
        self.linear_1 = HybridLinearKAN(self.hidden_size, self.hidden_size)
        self.liner_kan2 = HybridLinearKAN(self.hidden_size, num_classes)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.liner_kan1(x)
        x = self.linear_1(x)
        x = self.liner_kan2(x)
        return x


class LinearNet(torch.nn.Module):
    def __init__(
        self,
        num_classes=10,
        input_size=28 * 28,
        hidden_size=100,
    ):
        super(LinearNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.linear2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.linear3 = torch.nn.Linear(self.hidden_size, num_classes)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x


import torch.nn as nn


class CNNKAN(nn.Module):
    def __init__(self):
        super(CNNKAN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.kan1 = HybridLinearKAN(64 * 8 * 8, 256)
        self.kan2 = HybridLinearKAN(256, 10)

    def forward(self, x):
        x = F.selu(self.conv1(x))
        x = self.pool1(x)
        x = F.selu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.kan1(x)
        x = self.kan2(x)
        return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)  # Final output layer

    def forward(self, x):
        # Convolutional layers
        x = F.selu(self.conv1(x))
        x = self.pool1(x)
        x = F.selu(self.conv2(x))
        x = self.pool2(x)

        # Flattening the layer for the fully connected layer
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.selu(self.fc1(x))
        x = self.fc2(x)

        return x
