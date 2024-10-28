import numpy as np
import torch
import torch.nn.functional as F
import torch.utils
from torchvision import transforms
from tqdm import tqdm

from ml_security.attacks.membership_inference_attack import (
    ExampleAttackModel,
    MembershipInferenceAttack,
)
from ml_security.datasets.datasets import DatasetType, create_dataloader
from ml_security.examples.mnist.model import Net
from ml_security.utils.logger import logger
from ml_security.utils.utils import get_device, set_seed

# Sets random seed for reproducibility.
set_seed(42)
MODEL_PATH = "ml_security/examples/mnist/mnist_cnn.pt"


@torch.no_grad()
def get_confidence_scores(
    model, data_loader: torch.utils.data.DataLoader, device: torch.device
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

    mia = MembershipInferenceAttack(
        train_loader=train_loader,
        holdout_loader=holdout_loader,
        model=model,
        device=device,
        get_confidence_scores_fn=get_confidence_scores,
    )
    attack_model = ExampleAttackModel(input_dim=10).to(device)
    attack_model = mia.attack(
        attack_model=attack_model,
        epochs=1,
        lr=0.01,
    )

    # Defines the attack model.
    acc = mia.evaluate(attack_model)
    logger.info(f"Attack model accuracy: {acc}")
