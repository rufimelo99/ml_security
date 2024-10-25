import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm

from ml_security.attacks.membership_inference_attack import create_attack_dataloader
from ml_security.datasets.datasets import DatasetType, create_dataloader
from ml_security.examples.mnist.model import Net
from ml_security.logger import logger
from ml_security.utils.utils import get_device, set_seed

# Sets random seed for reproducibility.
set_seed(42)
MODEL_PATH = "ml_security/examples/mnist/mnist_cnn.pt"


# Defines a simple attack model.
class AttackModel(nn.Module):
    def __init__(self):
        super(AttackModel, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


if __name__ == "__main__":
    # Defines what device we are using.
    device = get_device()
    logger.info("Initializing the network", device=device)

    # Initialize the network.
    model = Net().to(device)

    # Loads the pretrained model.
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device, weights_only=True)
    )

    # Sets the model in evaluation mode. In this case this is for the Dropout layers
    model.eval()

    transformation = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_loader, holdout_loader = create_dataloader(
        dataset=DatasetType.MNIST,
        download=True,
        root="../data",
        batch_size=1,
        shuffle=True,
        train=False,
        transformation=transformation,
        max_samples=100,
        split_ratio=[80, 20],
    )

    attack_loader, attack_labels = create_attack_dataloader(
        train_loader=train_loader,
        holdout_loader=holdout_loader,
        model=model,
        device=device,
    )

    # Initialize the attack model.
    attack_model = AttackModel().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(attack_model.parameters(), lr=0.001)

    # Trains the attack model.
    attack_model.train()

    for epoch in range(100):
        for data, target in tqdm(attack_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = attack_model(data)
            loss = criterion(output, target.unsqueeze(1))
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            logger.info("Finished epoch", epoch=epoch, loss=loss.item())

    attack_model.eval()

    attack_predictions = []
    with torch.no_grad():
        for data, target in tqdm(attack_loader):
            data, target = data.to(device), target.to(device)
            output = attack_model(data)
            attack_predictions.append(output.cpu().numpy())

    attack_predictions = np.concatenate(attack_predictions)

    # Calculate the accuracy of the attack model.
    attack_accuracy = np.mean((attack_predictions > 0.5) == attack_labels)
    logger.info("Attack stats", accuracy=attack_accuracy)
