import json
import math

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ml_security.exploration.kolmogorov_arnold.network import HybridLinearKAN
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


class OriginalKANLinear(torch.nn.Module):
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
        super(OriginalKANLinear, self).__init__()
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


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Define the layers
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, padding=1
        )  # Conv2D(32, (3, 3), padding='same')
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1
        )  # Conv2D(32, (3, 3), padding='same')
        self.pool1 = nn.MaxPool2d(
            kernel_size=2, stride=2
        )  # MaxPooling2D(pool_size=(2, 2))

        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )  # Conv2D(64, (3, 3), padding='same')
        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1
        )  # Conv2D(64, (3, 3), padding='same')
        self.pool2 = nn.MaxPool2d(
            kernel_size=2, stride=2
        )  # MaxPooling2D(pool_size=(2, 2))

        self.conv5 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )  # Conv2D(128, (3, 3), padding='same')
        self.conv6 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, padding=1
        )  # Conv2D(128, (3, 3), padding='same')
        self.pool3 = nn.MaxPool2d(
            kernel_size=2, stride=2
        )  # MaxPooling2D(pool_size=(2, 2))

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 1024)  # Dense(1024)
        self.fc2 = nn.Linear(1024, 512)  # Dense(512)
        self.fc3 = nn.Linear(512, 10)  # Dense(10)

        # Dropout layer
        self.dropout = nn.Dropout(0.5)

        # L2 regularization for Dense layers will be added during the optimizer setup

    def forward(self, x):
        # Pass through the convolutional layers
        x = F.relu(self.conv1(x))  # Conv2D(32, (3, 3)) -> Activation('relu')
        x = F.relu(self.conv2(x))  # Conv2D(32, (3, 3)) -> Activation('relu')
        x = self.pool1(x)  # MaxPooling2D(pool_size=(2, 2))

        x = F.relu(self.conv3(x))  # Conv2D(64, (3, 3)) -> Activation('relu')
        x = F.relu(self.conv4(x))  # Conv2D(64, (3, 3)) -> Activation('relu')
        x = self.pool2(x)  # MaxPooling2D(pool_size=(2, 2))

        x = F.relu(self.conv5(x))  # Conv2D(128, (3, 3)) -> Activation('relu')
        x = F.relu(self.conv6(x))  # Conv2D(128, (3, 3)) -> Activation('relu')
        x = self.pool3(x)  # MaxPooling2D(pool_size=(2, 2))

        # Flatten the output
        x = x.view(-1, 128 * 4 * 4)  # Flatten()

        # Pass through the fully connected layers
        x = self.dropout(
            F.relu(self.fc1(x))
        )  # Dense(1024) -> Dropout -> Activation('relu')
        x = self.dropout(
            F.relu(self.fc2(x))
        )  # Dense(512) -> Dropout -> Activation('relu')

        # Final output layer (no activation since it's used with CrossEntropyLoss)
        x = self.fc3(x)  # Dense(10)

        return x


class CNNKAN(nn.Module):
    def __init__(self):
        super(CNNKAN, self).__init__()

        # Define the layers
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, padding=1
        )  # Conv2D(32, (3, 3), padding='same')
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1
        )  # Conv2D(32, (3, 3), padding='same')
        self.pool1 = nn.MaxPool2d(
            kernel_size=2, stride=2
        )  # MaxPooling2D(pool_size=(2, 2))

        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )  # Conv2D(64, (3, 3), padding='same')
        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1
        )  # Conv2D(64, (3, 3), padding='same')
        self.pool2 = nn.MaxPool2d(
            kernel_size=2, stride=2
        )  # MaxPooling2D(pool_size=(2, 2))

        self.conv5 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )  # Conv2D(128, (3, 3), padding='same')
        self.conv6 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, padding=1
        )  # Conv2D(128, (3, 3), padding='same')
        self.pool3 = nn.MaxPool2d(
            kernel_size=2, stride=2
        )  # MaxPooling2D(pool_size=(2, 2))

        # Fully connected layers
        self.kan_linear1 = OriginalKANLinear(128 * 4 * 4, 1024)
        self.kan_linear2 = OriginalKANLinear(1024, 512)
        self.kan_linear3 = OriginalKANLinear(512, 10)  # Dense(10)

        # Dropout layer
        self.dropout = nn.Dropout(0.5)

        # L2 regularization for Dense layers will be added during the optimizer setup

    def forward(self, x):
        # Pass through the convolutional layers
        x = F.relu(self.conv1(x))  # Conv2D(32, (3, 3)) -> Activation('relu')
        x = F.relu(self.conv2(x))  # Conv2D(32, (3, 3)) -> Activation('relu')
        x = self.pool1(x)  # MaxPooling2D(pool_size=(2, 2))

        x = F.relu(self.conv3(x))  # Conv2D(64, (3, 3)) -> Activation('relu')
        x = F.relu(self.conv4(x))  # Conv2D(64, (3, 3)) -> Activation('relu')
        x = self.pool2(x)  # MaxPooling2D(pool_size=(2, 2))

        x = F.relu(self.conv5(x))  # Conv2D(128, (3, 3)) -> Activation('relu')
        x = F.relu(self.conv6(x))  # Conv2D(128, (3, 3)) -> Activation('relu')
        x = self.pool3(x)  # MaxPooling2D(pool_size=(2, 2))

        # Flatten the output
        x = x.view(-1, 128 * 4 * 4)  # Flatten()

        # Pass through the fully connected layers
        x = self.dropout(
            self.kan_linear1(x)
        )  # Dense(1024) -> Dropout -> Activation('relu')
        x = self.dropout(
            self.kan_linear2(x)
        )  # Dense(512) -> Dropout -> Activation('relu')

        # Final output layer (no activation since it's used with CrossEntropyLoss)
        x = self.kan_linear3(x)  # Dense(10)

        return x
