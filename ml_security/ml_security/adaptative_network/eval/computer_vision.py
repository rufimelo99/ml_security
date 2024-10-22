import argparse
import json
import math
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from ml_security.adaptative_network.network import HybridLinearKAN
from ml_security.datasets.computer_vision import (
    NUM_CLASSES,
    DatasetType,
    create_dataloader,
)
from ml_security.logger import logger
from ml_security.utils import get_device, set_seed

set_seed(42)
DEVICE = get_device()

INPUT_SIZE = 28 * 28
HIDDEN_SIZE = 100
LR = 1e-3


class HybridNet(torch.nn.Module):
    def __init__(self, num_classes=NUM_CLASSES[DatasetType.MNIST]):
        super(HybridNet, self).__init__()
        self.liner_kan1 = HybridLinearKAN(INPUT_SIZE, HIDDEN_SIZE)
        self.linear_1 = HybridLinearKAN(HIDDEN_SIZE, HIDDEN_SIZE)
        self.liner_kan2 = HybridLinearKAN(HIDDEN_SIZE, num_classes)

    def forward(self, x):
        x = x.view(-1, INPUT_SIZE)
        x = self.liner_kan1(x)
        x = F.relu(x)
        x = self.linear_1(x)
        x = F.relu(x)
        x = self.liner_kan2(x)
        return x


class LinearNet(torch.nn.Module):
    def __init__(self, num_classes=NUM_CLASSES[DatasetType.MNIST]):
        super(LinearNet, self).__init__()
        self.linear1 = torch.nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.linear2 = torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.linear3 = torch.nn.Linear(HIDDEN_SIZE, num_classes)

    def forward(self, x):
        x = x.view(-1, INPUT_SIZE)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x


""" Retrieved from https://github.com/Blealtan/efficient-kan/blob/master/examples/mnist.py"""


class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(
            self.base_weight, a=math.sqrt(5) * self.scale_base
        )
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(
                    self.spline_scaler, a=math.sqrt(5) * self.scale_spline
                )

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output

        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )


##############################################


def classic_training(
    trainloader,
    valloader,
    model,
    optimizer,
    criterion,
    scheduler,
    epochs,
    device,
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
                images = images.view(-1, 28 * 28).to(device)
                optimizer.zero_grad()
                output = model(images)
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
                images = images.view(-1, 28 * 28).to(device)
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
    train_losses,
    train_accuracies,
    val_losses,
    val_accuracies,
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
    ax[0].plot(train_losses, label="KAN train loss", color=colors["KAN"], linestyle="-")
    ax[0].plot(val_losses, label="KAN val loss", color=colors["KAN"], linestyle="--")
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
        train_accuracies, label="KAN train accuracy", color=colors["KAN"], linestyle="-"
    )
    ax[1].plot(
        val_accuracies, label="KAN val accuracy", color=colors["KAN"], linestyle="--"
    )
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
    train_losses,
    train_accuracies,
    val_losses,
    val_accuracies,
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
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="MNIST", choices=["MNIST", "CIFAR10", "FASHION_MNIST"]
    )
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    dataset = DatasetType[args.dataset]
    epochs = args.epochs

    trainloader = create_dataloader(dataset=dataset, batch_size=64, train=True)
    valloader = create_dataloader(dataset=dataset, batch_size=64, train=False)

    # Define model
    model = KAN([INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES[dataset]])
    model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    criterion = nn.CrossEntropyLoss()

    (model, train_losses, train_accuracies, val_losses, val_accuracies) = (
        classic_training(
            trainloader,
            valloader,
            model,
            optimizer,
            criterion,
            scheduler,
            epochs,
            DEVICE,
        )
    )

    linear_model = LinearNet(num_classes=NUM_CLASSES[dataset])
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
    )

    hybrid_model = HybridNet(num_classes=NUM_CLASSES[dataset])
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
    )

    # Save models
    directory = "ml_security/adaptative_network/eval/"

    # Create repo for experience dataset
    dataset_dir = directory + args.dataset + "/"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    torch.save(model.state_dict(), dataset_dir + "kan.pth")
    torch.save(linear_model.state_dict(), dataset_dir + "linear.pth")
    torch.save(hybrid_model.state_dict(), dataset_dir + "hybrid.pth")

    # Plot results
    plot_results(
        train_losses,
        train_accuracies,
        val_losses,
        val_accuracies,
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
        train_losses,
        train_accuracies,
        val_losses,
        val_accuracies,
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
