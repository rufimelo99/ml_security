import argparse

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


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.liner_kan1 = HybridLinearKAN(28 * 28, 100)
        self.linear_1 = torch.nn.Linear(100, 100)
        self.liner_kan2 = HybridLinearKAN(100, NUM_CLASSES[DatasetType.MNIST])

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.liner_kan1(x)
        x = F.relu(x)
        x = self.linear_1(x)
        x = F.relu(x)
        x = self.liner_kan2(x)
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    trainloader = create_dataloader(
        dataset=DatasetType.MNIST, batch_size=64, train=True
    )
    valloader = create_dataloader(dataset=DatasetType.MNIST, batch_size=64, train=False)

    model = Net()

    model.to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    criterion = nn.CrossEntropyLoss()
    for epoch in tqdm(range(args.epochs)):
        model.train()
        with tqdm(trainloader) as pbar:
            for i, (images, labels) in enumerate(pbar):
                images = images.view(-1, 28 * 28).to(DEVICE)
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels.to(DEVICE))
                loss.backward()
                optimizer.step()
                accuracy = (output.argmax(dim=1) == labels.to(DEVICE)).float().mean()
                pbar.set_postfix(
                    loss=loss.item(),
                    accuracy=accuracy.item(),
                    lr=optimizer.param_groups[0]["lr"],
                )

        model.eval()
        val_loss = 0
        val_accuracy = 0
        with torch.no_grad():
            for images, labels in valloader:
                images = images.view(-1, 28 * 28).to(DEVICE)
                output = model(images)
                val_loss += criterion(output, labels.to(DEVICE)).item()
                val_accuracy += (
                    (output.argmax(dim=1) == labels.to(DEVICE)).float().mean().item()
                )
        val_loss /= len(valloader)
        val_accuracy /= len(valloader)

        scheduler.step()

        logger.info(
            "Finished epoch.",
            epoch=epoch + 1,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
        )
