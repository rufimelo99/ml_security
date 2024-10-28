"""
This module contains the tests for the MNIST example.
"""

# pylint: disable=duplicate-code

import torch
from torchvision import transforms

from ml_security.attacks.membership_inference_attack import MembershipInferenceAttack
from ml_security.datasets.datasets import DatasetType, create_dataloader
from ml_security.examples.mnist.membership_inference_attack import MODEL_PATH, Net


def test_can_load_model():
    model = Net()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    assert model is not None


def test_dataloader_creation():
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
    model = Net()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))

    mia = MembershipInferenceAttack(
        train_loader=train_loader,
        holdout_loader=holdout_loader,
        model=model,
        device="cpu",
    )

    attack_loader, attack_labels = mia.create_attack_dataloader(
        train_loader=train_loader,
        holdout_loader=holdout_loader,
        model=model,
        device="cpu",
    )
    assert attack_loader is not None
    assert attack_labels is not None
