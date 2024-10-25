import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml_security.adaptative_network.eval.utils import (
    CNN,
    CNNKAN,
)
from ml_security.attacks.membership_inference_attack import create_attack_dataloader
from ml_security.datasets.datasets import (
    DATASET_REGISTRY,
    DatasetType,
    create_dataloader,
    DEFAULT_TRANSFORM_3CH,
)
from ml_security.logger import logger
from ml_security.utils.utils import get_device, set_seed

set_seed(42)
DEVICE = get_device()
BATCH_SIZE = 64


model_path = "ml_security/adaptative_network/eval/cnn/CIFAR10/classic_cnn.pth"
model = CNN()
model.load_state_dict(torch.load(model_path))
model.to(DEVICE)
model.eval()

dataset = "CIFAR10"

dataset = DatasetType[dataset]
dataset_info = DATASET_REGISTRY[dataset]

if dataset_info.origin == "TORCHVISION":
    trainloader = create_dataloader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        train=True,
        transformation=DEFAULT_TRANSFORM_3CH,
        max_samples=10000,
    )
    valloader = create_dataloader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        train=False,
        transformation=DEFAULT_TRANSFORM_3CH,
        max_samples=10000,
    )
else:
    raise ValueError("Unknown dataset origin.")


@torch.no_grad()
def get_confidence_scores(
    model, data_loader: DataLoader, device: torch.device
) -> np.ndarray:
    """
    Get the confidence scores for the given model and data loader.

    Args:
        model (torch.nn.Module): The model to use. A classifier in this scenario.
        data_loader (torch.utils.data.DataLoader): The data loader to use.
        device (torch.device): The device to use.

    Returns:
        np.ndarray: The confidence scores.
    """
    model.eval()
    confidence_scores = []
    for batch_data, batch_target in tqdm(data_loader):
        batch_data, batch_target = batch_data.to(device), batch_target.to(device)
        output = model(batch_data)
        confidence_scores.append(F.softmax(output, dim=1).cpu().numpy())

    # Returns the confidence scores -> Shape: (n_samples, n_classes)
    return np.concatenate(confidence_scores)


attack_loader, attack_labels = create_attack_dataloader(
    train_loader=trainloader,
    holdout_loader=valloader,
    model=model,
    device=DEVICE,
    get_confidence_scores=get_confidence_scores,
)


# Defines a simple attack model.
class AttackModel(nn.Module):
    def __init__(self):
        super(AttackModel, self).__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x.squeeze(1)


attack_model = AttackModel().to(DEVICE)
criterion = nn.BCELoss()
optimizer = optim.Adam(attack_model.parameters(), lr=0.001)

# Trains the attack model.
attack_model.train()

for epoch in range(100):
    for batch in tqdm(attack_loader):
        data, target = batch
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = attack_model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

attack_model.eval()

attack_predictions = []
with torch.no_grad():
    for batch in tqdm(attack_loader):
        data, target = batch
        data, target = data.to(DEVICE), target.to(DEVICE)
        output = attack_model(data)
        attack_predictions.append(output.cpu().numpy())


# Check accuracy
attack_predictions = np.concatenate(attack_predictions)
attack_predictions = np.round(attack_predictions)
accuracy = np.mean(attack_predictions == attack_labels)

logger.info(f"Attack accuracy: {accuracy}")
