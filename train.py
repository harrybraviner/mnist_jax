import jax
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision

from typing import Tuple


def train(
        train_dataset: Dataset,
        validation_dataset: Dataset,
        # Random seeds
        training_shuffle_seed: int,
        # Hyperparameters
        num_epochs: int,
        batch_size: int,
):
    # Convert datasets into dataloaders
    train_shuffle_generator = torch.Generator().manual_seed(training_shuffle_seed)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   num_workers=1, shuffle=True, generator=train_shuffle_generator)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size,
                                                        num_workers=1, shuffle=False)


def main():
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

    train(
        train_dataset=train_dataset,
        validation_dataset=test_dataset,  # FIXME - switch to using a proper validation dataset
        training_shuffle_seed=123,
        num_epochs=3,
        batch_size=64,
    )


if __name__ == '__main__':
    main()
