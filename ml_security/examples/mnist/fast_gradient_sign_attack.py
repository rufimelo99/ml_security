import torch
from torchvision import transforms

from ml_security.attacks.fast_gradient_sign_attack import (
    DenormalizingTransformation,
    FastGradientSignAttack,
)
from ml_security.datasets.datasets import DatasetType, create_dataloader
from ml_security.examples.mnist.model import Net
from ml_security.utils.logger import logger
from ml_security.utils.utils import get_device, set_seed

# Sets random seed for reproducibility.
set_seed(42)


if __name__ == "__main__":
    epsilons = [0, 0.05]
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

    fgsm = FastGradientSignAttack(epsilon=0.3, device=device)

    for eps in epsilons:
        adv_examples = fgsm.attack(
            model,
            test_loader,
            denormlizing_transform=DenormalizingTransformation(
                mean=[0.130], std=[0.3081]
            ),
        )
        acc = fgsm.evaluate(adv_examples)
        accuracies.append(acc)
        examples.append(adv_examples)
        logger.info("Finished attack.", epsilon=eps, accuracy=acc)
