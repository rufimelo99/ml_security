import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml_security.attacks.membership_inference_attack import MembershipInferenceAttack
from ml_security.datasets.datasets import (
    DATASET_REGISTRY,
    DEFAULT_TRANSFORM_3CH,
    DatasetType,
    create_dataloader,
)
from ml_security.exploration.kolmogorov_arnold.eval.utils import CNN, CNNKAN
from ml_security.logger import logger
from ml_security.utils.utils import get_device, set_seed

set_seed(42)
DEVICE = get_device()
BATCH_SIZE = 64


model_path = (
    "ml_security/exploration/kolmogorov_arnold/eval/cnn/CIFAR10/classic_cnn.pth"
)
model = CNN()
model.load_state_dict(torch.load(model_path))
model.to(DEVICE)
model.eval()

dataset = "CIFAR10"

dataset = DatasetType[dataset]
dataset_info = DATASET_REGISTRY[dataset]
MAX_SAMPLES = 10

if dataset_info.origin == "TORCHVISION":
    trainloader = create_dataloader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        train=True,
        transformation=DEFAULT_TRANSFORM_3CH,
        max_samples=MAX_SAMPLES,
    )
    valloader = create_dataloader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        train=False,
        transformation=DEFAULT_TRANSFORM_3CH,
        max_samples=MAX_SAMPLES,
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
        confidences = F.softmax(output, dim=1).unsqueeze(1)
        confidence_scores.append(confidences.cpu().numpy())

    # Returns the confidence scores -> Shape: (n_samples, n_classes)
    return np.concatenate(confidence_scores)


mia = MembershipInferenceAttack(
    model=model,
    train_loader=trainloader,
    holdout_loader=valloader,
    device=DEVICE,
    get_confidence_scores_fn=get_confidence_scores,
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

attack_model = mia.attack(attack_model, epochs=1)

accuracy = mia.evaluate(attack_model)

logger.info(f"Attack accuracy: {accuracy}")
