from dataclasses import dataclass
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from tqdm import tqdm

from ml_security.attacks.base import AdversarialAttack


@dataclass
class DenormalizingTransformation:
    mean: Union[torch.Tensor, List[float]]
    std: Union[torch.Tensor, List[float]]


class FastGradientSignAttack(AdversarialAttack):
    """
    Fast Gradient Sign Attack (FGSM).

    The Fast Gradient Sign Attack (FGSM), introduced by Goodfellow et al. in
    "Explaining and Harnessing Adversarial Examples," is one of the earliest
    and most widely-used adversarial attacks. This method is both powerful and
    intuitive, exploiting how neural networks learn by using gradients.

    Neural networks typically adjust their internal weights to minimize a loss
    function during training, based on gradients computed via backpropagation.
    FGSM, however, takes the opposite approach: instead of altering the model's
    weights, it perturbs the input data to maximize the loss.

    In practice, FGSM computes the gradient of the loss with respect to the input
    data and adjusts the input in the direction that increases the loss. By making
    minimal but strategic changes to the input, the attack can significantly degrade
    the model's performance, effectively "fooling" it into making incorrect predictions.
    """

    def __init__(
        self,
        epsilon: float,
        device: torch.device,
    ) -> None:
        """
        Initializes the Fast Gradient Sign Attack.

        Args:
            epsilon (float): The epsilon value to use for the attack.
            device (torch.device): The device to use for the attack.
        """
        super().__init__(alias="FastGradientSignAttack")
        self.device = device
        self.epsilon = epsilon

    def attack(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        denormlizing_transform: Optional[DenormalizingTransformation] = None,
    ) -> List:
        """
        Performs the Fast Gradient Sign Attack on the model.

        Args:
            model (torch.nn.Module): The model to attack.
            dataloader (torch.utils.data.DataLoader): The dataloader for the dataset.
            denormlizing_transform (Optional[DenormalizingTransformation]): The denormalizing transformation.

        Returns:
            List: The adversarial examples.
            List: All examples.
        """
        adv_examples = []
        all_examples = []

        for data, target in tqdm(dataloader):
            data, target = data.to(self.device), target.to(self.device)

            # Sets `requires_grad` attribute of tensor.
            data.requires_grad = True

            # Forward pass the data through the model.
            output = model(data)
            init_pred = output.max(1, keepdim=True)[1]

            # If the initial prediction is wrong, move on.
            if init_pred.item() != target.item():
                continue

            loss = F.nll_loss(output, target)
            model.zero_grad()
            loss.backward()

            # Collects ``datagrad``.
            data_grad = data.grad.data

            # Restores the data to its original scale.
            if denormlizing_transform:
                data = self._denorm(
                    data, denormlizing_transform.mean, denormlizing_transform.std
                )

            # Calls FGSM Attack.
            perturbed_data = self._fgsm_attack(data, self.epsilon, data_grad)

            # Re-classify the perturbed image.
            output = model(perturbed_data)

            final_pred = output.max(1, keepdim=True)[1]
            adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
            entry = (init_pred.item(), final_pred.item(), adv_ex)
            if final_pred.item() == target.item():
                adv_examples.append(entry)
            all_examples.append(entry)
        return adv_examples, all_examples

    def _fgsm_attack(
        self, image: torch.Tensor, epsilon: float, data_grad: torch.Tensor
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

    def _denorm(
        self,
        batch: Union[torch.Tensor, List[torch.Tensor]],
        mean: Union[torch.Tensor, List[float]],
        std: Union[torch.Tensor, List[float]],
    ) -> torch.Tensor:
        """
        Convert a batch of tensors to their original scale.

        Args:
            batch (Union[torch.Tensor, List[torch.Tensor]]): Batch of normalized tensors.
            mean (Union[torch.Tensor, List[float]]): Mean used for normalization.
            std (Union[torch.Tensor, List[float]]): Standard deviation used for normalization.

        Returns:
            torch.Tensor: batch of tensors without normalization applied to them.
        """
        if isinstance(mean, list):
            mean = torch.tensor(mean).to(self.device)
        if isinstance(std, list):
            std = torch.tensor(std).to(self.device)

        return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)
