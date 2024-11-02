from typing import Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

from ml_security.attacks.base import AdversarialAttack
from ml_security.logger import logger
from ml_security.utils.distance import DistanceMetricType, get_distance_metric

C_UPPER = 1e10


class CarliniWagnerAttack(AdversarialAttack):
    """
    Carlini-Wagner attack is an optimization-based attack that aims to generate adversarial examples
    that are misclassified by the model. The attack minimizes the perturbation size while maximizing
    the loss of the true class or minimizing the loss of the target class.

    Parameters
        ----------
        device: torch.device
            The device to run the attack on.
        distance_metric: DistanceMetricType
            The distance metric to use for the attack.
        c: float
            The weight for the loss term.
        lr: float
            The learning rate for the attack.
        num_iterations: int
            The number of optimization steps to perform.
        binary_search_steps: int
            Number of times to adjust constant with binary search (positive value).

    References
    -----------
    - Carlini, N., & Wagner, D. (2017). Towards evaluating the robustness of neural networks.
        In 2017 IEEE Symposium on Security and Privacy (SP) (pp. 39-57).
    """

    def __init__(
        self,
        device: torch.device,
        distance_metric: DistanceMetricType = DistanceMetricType.L2,
        c: float = 1e-4,
        lr: float = 0.01,
        num_iterations: int = 1000,
        binary_search_steps: int = 9,
    ):
        super().__init__(alias="CarliniWagnerAttack")
        self.distance_metric = distance_metric
        self.c = c
        self.lr = lr
        self.num_iterations = num_iterations
        self.binary_search_steps = binary_search_steps
        self.device = device
        logger.info(
            "Initialized Carlini-Wagner Attack",
            c=c,
            lr=lr,
            num_iterations=num_iterations,
            binary_search_steps=binary_search_steps,
        )

        if self.binary_search_steps <= 0:
            self.binary_search_steps = 1

    def attack(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader):
        """
        Generates adversarial examples by optimizing the input to misclassify in the target model.

        Parameters
        ----------
        model : torch.nn.Module
            The neural network model to attack.
        dataloader : torch.utils.data.DataLoader
            A dataloader providing batches of input data to generate adversarial examples.

        Returns
        -------
        torch.Tensor
            A tensor containing the adversarial examples created from the input dataset.
        """
        adv_examples = []
        all_examples = []
        correct = 0
        for data, target in tqdm(dataloader):
            data, target = data.to(self.device), target.to(self.device)

            # Generates adversarial example.
            perturbed_data = self._carlini_wagner_attack(
                model, data, target, self.device
            )

            # Re-classify the perturbed image.
            output = model(perturbed_data)

            # Gets the index of the max log-probability.
            final_pred = output.max(1, keepdim=True)[1]
            correct += final_pred.eq(target.view_as(final_pred)).sum().item()

            # Gets the adversarial examples when it is misclassified.
            adv_idxs = final_pred.ne(target.view_as(final_pred)).view(-1)
            for i in range(len(adv_idxs)):
                adv_ex = perturbed_data[i].squeeze().detach().cpu().numpy()
                entry = (target[i].item(), final_pred[i].item(), adv_ex)
                if adv_idxs[i]:
                    adv_examples.append(entry)

                all_examples.append(entry)

        final_acc = correct / (float(len(dataloader)) * dataloader.batch_size)
        logger.info("Final Accuracy", final_acc=final_acc)
        return adv_examples, all_examples

    def _carlini_wagner_attack(
        self,
        model: torch.nn.Module,
        images: torch.Tensor,
        labels: torch.Tensor,
        device: torch.device,
        distance_metric: DistanceMetricType = DistanceMetricType.L2,
        target_labels: Optional[torch.Tensor] = None,
    ):
        """
        Performs the Carlini-Wagner attack on a batch of images, creating adversarial examples
        with minimal perturbation to the original images based on the specified distance metric.

        Parameters
        ----------
        model : torch.nn.Module
            The neural network model to attack.
        images : torch.Tensor
            A batch of original images serving as the input for the attack.
        labels : torch.Tensor
            True class labels for the input images, used in untargeted attacks.
        device : torch.device
            Device (CPU or GPU) on which the attack computations are performed.
        distance_metric : DistanceMetricType, optional
            The distance metric (e.g., L2) guiding the attack's perturbation minimization.
        target_labels : Optional[torch.Tensor], optional
            Target class labels for a targeted attack. If None, the attack is untargeted.

        Returns
        -------
        torch.Tensor
            A tensor containing the adversarial examples generated from the original images.
        """
        distance_metric = get_distance_metric(self.distance_metric)

        images = images.to(device)
        labels = labels.to(device)
        if target_labels is not None:
            target_labels = target_labels.to(device)
        ori_images = images.clone().detach()

        lower_bounds = torch.zeros_like(images).to(device)
        upper_bounds = torch.ones_like(images).to(device)

        # Initializes perturbation delta as a variable with gradient enabled.
        delta = torch.zeros_like(images, requires_grad=True).to(device)

        optimizer = torch.optim.Adam([delta], lr=self.lr)

        lower_bound_c = torch.zeros(images.size(0)).to(device)
        upper_bound_c = torch.ones(images.size(0)).to(device) * C_UPPER
        current_c = torch.ones(images.size(0)).to(device) * self.c

        for s in range(self.binary_search_steps):

            # Clamps the perturbation to be within the bounds.
            delta.data = torch.clamp(
                delta, lower_bounds - ori_images, upper_bounds - ori_images
            )

            # Optimizes the perturbation.
            delta = self._optimize_perturbation(
                model,
                ori_images,
                delta,
                labels,
                current_c,
                optimizer,
                distance_metric,
                target_labels,
            )

            # Check if the attack is successful
            success = self._check_attack_success(
                model, ori_images + delta, labels, target_labels
            )

            # Update the binary search bounds and current_c
            lower_bound_c = torch.where(success, lower_bound_c, current_c)
            upper_bound_c = torch.where(success, current_c, upper_bound_c)
            current_c = (lower_bound_c + upper_bound_c) / 2

        # Generates final adversarial examples, clamped to be within [0, 1].
        adv_images = torch.clamp(ori_images + delta, 0, 1).detach()
        return adv_images

    def _optimize_perturbation(
        self,
        model: torch.nn.Module,
        ori_images: torch.Tensor,
        delta: torch.Tensor,
        labels: torch.Tensor,
        current_c: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        distance_metric: DistanceMetricType,
        target_labels: Optional[torch.Tensor] = None,
    ):
        """
        Optimizes the perturbation delta to minimize the loss function.

        Parameters
        ----------
        model : torch.nn.Module
            The neural network model to attack.
        ori_images : torch.Tensor
            The original images used as the input for the attack.
        delta : torch.Tensor
            The perturbation to optimize.
        labels : torch.Tensor
            True class labels for the input images, used in untargeted attacks.
        current_c : torch.Tensor
            The weight for the loss term.
        optimizer : torch.optim.Optimizer
            The optimizer used to update the perturbation.
        distance_metric : DistanceMetricType
            The distance metric guiding the attack's perturbation minimization.
        target_labels : Optional[torch.Tensor], optional
            Target class labels for a targeted attack. If None, the attack is untargeted.

        Returns
        -------
        torch.Tensor
            The optimized perturbation delta.
        """
        model.eval()
        for _ in range(self.num_iterations):
            # Generates the adversarial image and clamps it to be within [0, 1].
            adv_images = torch.clamp(ori_images + delta, 0, 1)

            # Predicts the class of the adversarial image.
            outputs = model(adv_images)

            # Computes the loss.
            if target_labels:
                # If target_labels is provided, the attack is targeted.
                # It maximizes the logit for the target label.
                targeted_loss = F.cross_entropy(outputs, target_labels)
                f_loss = targeted_loss
            else:
                # If target_labels is not provided, the attack is untargeted.
                # It minimizes the logit for the true label.
                true_loss = F.cross_entropy(outputs, labels)
                f_loss = -true_loss

            # Minimizes perturbation size with L2 norm and adds f_loss.
            l2_loss = distance_metric(delta.view(delta.size(0), -1)).mean()
            loss = self.c * f_loss + l2_loss
            # Backward pass and optimize.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Projects delta to be within bounds if necessary.
            delta.data = torch.clamp(
                delta, -current_c.view(-1, 1, 1, 1), current_c.view(-1, 1, 1, 1)
            )

        return delta

    def _check_attack_success(
        self, model, perturbed_images, original_labels, target_labels=None
    ):
        """
        Check if the attack is successful.

        Args:
            model (torch.nn.Module): The model to attack.
            perturbed_images (torch.Tensor): The perturbed images.
            original_labels (torch.Tensor): The original labels of the images.
            target_labels (torch.Tensor, optional): The target labels for a targeted attack.

        Returns:
            torch.Tensor: A boolean tensor indicating the success of the attack for each image.
        """
        model.eval()
        with torch.no_grad():
            outputs = model(perturbed_images)
            _, predicted = torch.max(outputs, 1)

        if target_labels is not None:
            # Targeted attack: check if the predictions match the target labels
            success = predicted.eq(target_labels)
        else:
            # Untargeted attack: check if the predictions do not match the original labels
            success = ~predicted.eq(original_labels)

        return success
