"""
Fast Gradient Sign Attack (FGSM)
The Fast Gradient Sign Attack (FGSM), introduced by Goodfellow et al. in Explaining and Harnessing Adversarial Examples, is one of the earliest and most widely-used adversarial attacks. This method is both powerful and intuitive, exploiting how neural networks learn by using gradients.

Neural networks typically adjust their internal weights to minimize a loss function during training, based on gradients computed via backpropagation. FGSM, however, takes the opposite approach: instead of altering the model’s weights, it perturbs the input data to maximize the loss.

In practice, FGSM computes the gradient of the loss with respect to the input data and adjusts the input in the direction that increases the loss. By making minimal but strategic changes to the input, the attack can significantly degrade the model’s performance, effectively "fooling" it into making incorrect predictions.

This technique is widely used to test the robustness of models and study their vulnerabilities to adversarial examples. 

"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
