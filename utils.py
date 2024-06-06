import torch
import torchvision


def get_mnist_split():
    """
    Common function to get training, validation and test splits.
    Will reuse in training and tuning applications.
    :return: (train, valid, test)
    """
    # For now, just use MNIST dataset
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    (train_dataset, test_dataset) = (torchvision.datasets.MNIST(
        "MNIST",
        train=t,
        download=True,
        transform=transform,
    ) for t in (True, False))

    # Split the training dataset into training and validation in a reproducible way.
    [train_dataset, validation_dataset] = torch.utils.data.random_split(train_dataset, [0.7, 0.3],
                                                                        generator=torch.Generator().manual_seed(1234))

    return train_dataset, validation_dataset, test_dataset
