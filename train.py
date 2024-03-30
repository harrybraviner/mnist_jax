import jax
import jax.numpy as jnp
from jaxtyping import Float, Array
import equinox as eqx
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision

from typing import Tuple, List


class CNN(eqx.Module):
    layers: List

    def __init__(self, key):
        # Architecture from Equinox tutorial. Single convolutional layer than 3-layer MLP. Parameterize later.
        key1, key2, key3, key4 = jax.random.split(key, 4)
        self.layers = [
            eqx.nn.Conv2d(1, 3, kernel_size=4, key=key1),
            eqx.nn.MaxPool2d(kernel_size=2),
            jax.nn.relu,
            jnp.ravel,
            eqx.nn.Linear(1728, 512, key=key2),
            jax.nn.sigmoid,  # Curious choice.
            eqx.nn.Linear(512, 64, key=key3),
            jax.nn.relu,
            eqx.nn.Linear(64, 10, key=key4),
            jax.nn.log_softmax,
        ]

    def __call__(self, x: Float[Array, "1 28 28"]) -> Float[Array, "10"]:
        for layer in self.layers:
            x = layer(x)
        return x


def train(
        train_dataset: Dataset,
        validation_dataset: Dataset,
        # Random seeds
        seed: int,
        # Hyperparameters
        num_epochs: int,
        batch_size: int,
):
    key_master = jax.random.PRNGKey(seed)
    key_training_shuffle_seed, key_model_init = jax.random.split(key_master, 2)

    # Convert datasets into dataloaders
    train_shuffle_generator = \
        torch.Generator().manual_seed(jax.random.choice(key_training_shuffle_seed, 10000, shape=()).item())
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                   num_workers=1, shuffle=True, generator=train_shuffle_generator)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size,
                                                        num_workers=1, shuffle=False)

    model = CNN(key_model_init)


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
        seed=123,
        num_epochs=3,
        batch_size=64,
    )


if __name__ == '__main__':
    main()
