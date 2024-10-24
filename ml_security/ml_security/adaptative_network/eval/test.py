import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
# I have a cnn model that I trained on the CIFAR10 dataset.
# I want to see how robust it is to adversarial attacks.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from ml_security.adaptative_network.eval.utils import (
    PreActBlock,
    PreActResNet,
    PreActResNetwithKAN,
    CIFARCNNKAN,
    CIFARCNN,
)
from ml_security.attacks.membership_inference_attack import create_attack_dataloader
from ml_security.datasets.datasets import (
    DATASET_REGISTRY,
    DatasetType,
    create_dataloader,
)
from ml_security.logger import logger
from ml_security.utils.utils import get_device, set_seed

# Set the seed
set_seed(42)
DEVICE = get_device()
BATCH_SIZE = 64
# Step 1: Load CIFAR10 Dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

test_set = torch.utils.data.Subset(test_set, range(100))

test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)

# Step 2: Load Pretrained Model (e.g., ResNet18)
model = CIFARCNN()
model.load_state_dict(torch.load("ml_security/adaptative_network/eval/cnn/CIFAR10/classic_cnn.pth"))
model.to(DEVICE)


# Step 3: Define L2 Attack (PGD)
def l2_pgd_attack(model, images, labels, epsilon, alpha, iters):
    images = images.to(DEVICE)
    labels = labels.to(DEVICE)
    ori_images = images.data

    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)
        
        # Calculate loss
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()

        # Generate perturbations with the gradient
        grad = images.grad.data

        # Normalize the gradient for L2 attack
        grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1).view(-1, 1, 1, 1)
        grad = grad / (grad_norm + 1e-8)  # Avoid division by zero

        # Update the image with small steps
        adv_images = images + alpha * grad

        # Clip the perturbation to stay within epsilon L2 norm
        perturbation = adv_images - ori_images
        perturbation_norm = torch.norm(perturbation.view(perturbation.size(0), -1), dim=1).view(-1, 1, 1, 1)
        perturbation = perturbation * torch.min(torch.ones_like(perturbation_norm), epsilon / perturbation_norm)
        
        # Update adversarial image
        images = ori_images + perturbation
        images = torch.clamp(images, -1, 1)  # Keep image in valid range
        images = images.detach()  # Detach the t
    return images

# Step 4: Test the L2 PGD Attack
def test_l2_attack(model, test_loader, epsilon, alpha, iters):
    correct = 0
    adv_examples = []
    
    for data, target in tqdm(test_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        
        # Generate adversarial example
        perturbed_data = l2_pgd_attack(model, data, target, epsilon, alpha, iters)
        
        # Re-classify the perturbed image
        output = model(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1]  # Get the index of the max log-probability
        # Check if the adversarial image was classified correctly
        if final_pred.item() == target.item():
            correct += 1

        # if we create a successful adversarial example, save it
        if final_pred.item() != target.item():
            adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
            adv_examples.append((target, final_pred.item(), adv_ex))

    final_acc = correct / float(len(test_loader))
    print(f"Test Accuracy = {final_acc * 100:.2f}%")
    return final_acc, adv_examples


# Step 5: Set attack parameters and run the attack
epsilon = 1.0  # Maximum L2 perturbation
alpha = 0.01   # Step size for each iteration
iters = 40     # Number of iterations

final_acc, adv_examples = test_l2_attack(model, test_loader, epsilon, alpha, iters)
