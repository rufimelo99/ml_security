import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from tqdm import tqdm

from ml_security.datasets.computer_vision import DatasetType, create_dataloader
from ml_security.examples.mnist.model import Net
from ml_security.logger import logger
from ml_security.utils import get_device, set_seed

# Sets random seed for reproducibility.
set_seed(42)


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


# Function to collect model predictions (confidence scores).
def get_confidence_scores(model, data_loader):
    confidence_scores = []
    with torch.no_grad():
        for data, target in tqdm(data_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            confidence_scores.append(F.softmax(output, dim=1)[:, 1].cpu().numpy())
    return np.concatenate(confidence_scores)


if __name__ == "__main__":
    pretrained_model = "examples/mnist/mnist_cnn.pt"

    # Defines what device we are using.
    device = get_device()
    logger.info("Initializing the network", device=device)

    # Initialize the network.
    model = Net().to(device)

    # Loads the pretrained model.
    model.load_state_dict(
        torch.load(pretrained_model, map_location=device, weights_only=True)
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

    # Gets confidence scores for both train and holdout sets
    train_confidence_scores = get_confidence_scores(model, train_loader)
    holdout_confidence_scores = get_confidence_scores(model, holdout_loader)

    # Label the samples: 1 for training data, 0 for holdout data
    train_labels = np.ones(len(train_confidence_scores))
    holdout_labels = np.zeros(len(holdout_confidence_scores))

    # Creates the dataset for the attack model.
    attack_data = np.concatenate(
        (train_confidence_scores, holdout_confidence_scores), axis=0
    )
    attack_labels = np.concatenate((train_labels, holdout_labels), axis=0)

    # Prepares data for the attack model.
    attack_dataset = TensorDataset(
        torch.Tensor(attack_data), torch.Tensor(attack_labels)
    )
    attack_loader = DataLoader(attack_dataset, batch_size=4, shuffle=True)

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
