import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

from ml_security.datasets.computer_vision import DatasetType, create_dataloader
from ml_security.examples.mnist.model import Net
from ml_security.logger import logger
from ml_security.utils.utils import get_device, set_seed

# Sets random seed for reproducibility.
set_seed(42)


def fgsm_attack(
    image: torch.Tensor, epsilon: float, data_grad: torch.Tensor
) -> torch.Tensor:
    """
    Fast Gradient Sign Method.

    Args:
        image (torch.Tensor): The original input image.
        epsilon (float): The epsilon value to use for the attack. Corresponds to the magnitude of the perturbation.
        data_grad (torch.Tensor): The gradient of the loss with respect to the input.

    Returns:
        torch.Tensor: The perturbed image.
    """
    # Collects the element-wise sign of the data gradient.
    data_grad_sign = data_grad.sign()

    # Creates the perturbed image by adjusting each pixel of the input image.
    perturbed_image = image + epsilon * data_grad_sign

    # Adds clipping to maintain [0,1] range.
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    return perturbed_image


def denorm(batch, mean=[0.1307], std=[0.3081]):
    """
    Convert a batch of tensors to their original scale.

    Args:
        batch (torch.Tensor): Batch of normalized tensors.
        mean (torch.Tensor or list): Mean used for normalization.
        std (torch.Tensor or list): Standard deviation used for normalization.

    Returns:
        torch.Tensor: batch of tensors without normalization applied to them.
    """
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)

    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)


def test(model, device, test_loader, epsilon):
    correct = 0
    adv_examples = []

    for data, target in tqdm(test_loader):
        data, target = data.to(device), target.to(device)

        # Sets `requires_grad` attribute of tensor.
        data.requires_grad = True

        # Forward pass the data through the model.
        output = model(data)
        init_pred = output.max(1, keepdim=True)[
            1
        ]  # Gets the index of the max log-probability.

        # If the initial prediction is wrong, move on.
        if init_pred.item() != target.item():
            continue

        loss = F.nll_loss(output, target)
        model.zero_grad()
        loss.backward()

        # Collects ``datagrad``.
        data_grad = data.grad.data

        # Restores the data to its original scale.
        data_denorm = denorm(data)

        # Calls FGSM Attack.
        perturbed_data = fgsm_attack(data_denorm, epsilon, data_grad)

        # Reapply normalization.
        perturbed_data_normalized = transforms.Normalize((0.1307,), (0.3081,))(
            perturbed_data
        )

        # Re-classify the perturbed image.
        output = model(perturbed_data_normalized)

        final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() == target.item():
            correct += 1
            if epsilon == 0 and len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        else:
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

    # Calculates final accuracy for this epsilon.
    final_acc = correct / float(len(test_loader))
    logger.info("Finished testing", epsilon=epsilon, accuracy=final_acc)

    # Returns the accuracy and an adversarial example
    return final_acc, adv_examples


def plot_examples(examples):
    cnt = 0
    plt.figure(figsize=(8, 10))
    for i in range(len(epsilons)):
        for j in range(len(examples[i])):
            cnt += 1
            plt.subplot(len(epsilons), len(examples[0]), cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel(f"Eps: {epsilons[i]}", fontsize=14)
            orig, adv, ex = examples[i][j]
            plt.title(f"{orig} -> {adv}")
            plt.imshow(ex, cmap="gray")
    plt.tight_layout()
    plt.show()


# Plot the accuracy vs epsilon graph
def plot_accuracy(epsilons, accuracies):
    plt.figure(figsize=(5, 5))
    plt.plot(epsilons, accuracies, "*-")
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(0, 0.35, step=0.05))
    plt.title("Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.show()


if __name__ == "__main__":
    epsilons = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    pretrained_model = "ml_security/examples/mnist/mnist_cnn.pt"
    transformation = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # MNIST Test dataset and dataloader declaration
    test_loader = create_dataloader(
        dataset=DatasetType.MNIST,
        download=True,
        root="../data",
        batch_size=1,
        shuffle=True,
        train=False,
        transformation=transformation,
    )

    # Define what device we are using
    device = get_device()
    logger.info("Initializing the network", device=device)

    # Initialize the network
    model = Net().to(device)

    # Load the pretrained model
    model.load_state_dict(
        torch.load(pretrained_model, map_location=device, weights_only=True)
    )

    # Set the model in evaluation mode. In this case this is for the Dropout layers
    model.eval()

    accuracies = []
    examples = []

    for eps in epsilons:
        acc, ex = test(model, device, test_loader, eps)
        accuracies.append(acc)
        examples.append(ex)

    plot_examples(examples)
    plot_accuracy(epsilons, accuracies)
