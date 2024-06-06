import jax
import jax.numpy as jnp
from jaxtyping import Float, Int, Array, PyTree
import equinox as eqx
import optax
import torch
from torch.utils.data import Dataset, DataLoader

from time import time
from typing import Tuple, List

import utils


class CNN(eqx.Module):
    layers: List

    def __init__(
            self,
            key,
            num_conv_channels: int = 3,
            hidden_layer_size: int = 512,
    ):
        n_x, n_y = 28, 28  # MNIST image sizes
        conv_kernel_size = 4
        pool_kernel_size = 2
        conv_output_dim = (n_x - conv_kernel_size - pool_kernel_size + 2) * \
                          (n_y - conv_kernel_size - pool_kernel_size + 2) * \
                          num_conv_channels

        # Architecture from Equinox tutorial. Single convolutional layer than 3-layer MLP. Parameterize later.
        key1, key2, key3, key4 = jax.random.split(key, 4)
        self.layers = [
            eqx.nn.Conv2d(1, num_conv_channels, kernel_size=conv_kernel_size, key=key1),
            eqx.nn.MaxPool2d(kernel_size=pool_kernel_size),
            jax.nn.relu,
            jnp.ravel,
            eqx.nn.Linear(conv_output_dim, hidden_layer_size, key=key2),
            jax.nn.sigmoid,  # Curious choice.
            eqx.nn.Linear(hidden_layer_size, hidden_layer_size, key=key3),
            jax.nn.relu,
            eqx.nn.Linear(hidden_layer_size, 10, key=key4),
            jax.nn.log_softmax,
        ]

    def __call__(self, x: Float[Array, "1 28 28"]) -> Float[Array, "10"]:
        for layer in self.layers:
            x = layer(x)
        return x


@eqx.filter_jit
def loss(
        model: CNN,
        x: Float[Array, "batch 1 28 28"],
        y: Int[Array, "batch"],
) -> Float[Array, ""]:
    pred_y = jax.vmap(model)(x)
    return cross_entropy(y, pred_y)


def cross_entropy(
        y: Int[Array, "batch"],
        pred_y: Float[Array, "batch 10"],
) -> Float[Array, ""]:
    pred_y = jnp.take_along_axis(pred_y, indices=jnp.expand_dims(y, axis=1), axis=1)
    return -jnp.mean(pred_y)


def accuracy(
        y: Int[Array, "batch"],
        pred_y: Float[Array, "batch 10"],
) -> Float[Array, ""]:
    pred_y = jnp.argmax(pred_y, axis=1)
    return jnp.mean(y == pred_y)


def evaluate(
        model: CNN,
        test_loader: torch.utils.data.DataLoader,
):
    avg_loss = 0.0
    avg_acc = 0.0
    n = 0
    for (x_batch, y_batch) in test_loader:
        x_batch = x_batch.numpy()
        y_batch = y_batch.numpy()
        n_batch = x_batch.shape[0]
        avg_loss += n_batch*loss(model, x_batch, y_batch)
        y_pred = jax.vmap(model)(x_batch)
        avg_acc += n_batch*accuracy(y_batch, y_pred)
        n += n_batch
    avg_loss /= n
    avg_acc /= n

    return avg_loss, avg_acc


def train(
        train_dataset: Dataset,
        validation_dataset: Dataset,
        # Random seeds
        seed: int,
        # Hyperparameters
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        model: CNN,
):
    key_training_shuffle_seed = jax.random.PRNGKey(seed)

    # Convert datasets into dataloaders
    train_shuffle_generator = \
        torch.Generator().manual_seed(jax.random.choice(key_training_shuffle_seed, 10000, shape=()).item())
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  num_workers=0, shuffle=True, generator=train_shuffle_generator)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size,
                                       num_workers=0, shuffle=False)

    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def take_step(
            model: CNN,
            opt_state: PyTree,
            x: Float[Array, "batch 1 28 28"],
            y: Int[Array, "batch"],
    ):
        loss_val, grads = eqx.filter_value_and_grad(loss)(model, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_val

    start_time_s = time()
    for epoch in range(1, num_epochs+1):
        for (x_batch, y_batch) in train_dataloader:
            model, opt_state, loss_val = take_step(model, opt_state, x_batch.numpy(), y_batch.numpy())

        test_loss, test_acc = evaluate(model, validation_dataloader)
        print(f'Epoch {epoch:02d}:\tTest loss: {test_loss:.06f}\tTest acc: {test_acc:.2%}')
    print(f'Took {time() - start_time_s:.2f}s')
    return test_acc


def main():
    train_dataset, validation_dataset, _ = utils.get_mnist_split()

    key_model_init = jax.random.PRNGKey(42)
    model = CNN(key_model_init)

    train(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        seed=123,
        num_epochs=3,
        batch_size=64,
        learning_rate=1e-2,
        model=model,
    )


if __name__ == '__main__':
    main()
