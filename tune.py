import jax.random
import optuna
import argparse
from os import path
import torch
from typing import Optional

import train
import utils


def objective(
        train_dataset: torch.utils.data.Dataset,
        validation_dataset: torch.utils.data.Dataset,
        trial: optuna.Trial,
) -> float:
    num_conv_channels = trial.suggest_categorical('num_conv_channels', [1, 2, 3, 4, 5, 6])
    hidden_layer_size = trial.suggest_categorical('hidden_layer_size', [64, 128, 256, 512])
    learning_rate = trial.suggest_categorical('learning_rate', [5e-4, 1e-3, 5e-3, 1e-2, 5e-2])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])

    key_model_init = jax.random.PRNGKey(42)
    model = train.CNN(
        key=key_model_init,
        num_conv_channels=num_conv_channels,
        hidden_layer_size=hidden_layer_size,
    )

    validation_acc = train.train(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        seed=123,
        num_epochs=10,
        batch_size=batch_size,
        learning_rate=learning_rate,
        model=model,
    )
    return validation_acc


def run_tuning(
        study_filename: str,
        n_trials: Optional[int],
):
    study = optuna.create_study(
        study_name='mnist_jax',
        storage=f'sqlite:///{path.abspath(study_filename)}',
        load_if_exists=True,
        direction='maximize',
    )

    train_dataset, validation_dataset, _ = utils.get_mnist_split()

    study.optimize(
        lambda trial: objective(
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            trial=trial
        ),
        n_trials=n_trials,
        gc_after_trial=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Runs tuning sweep over hyperparameters")
    parser.add_argument('--study_filename', type=str, default='./study.db')
    parser.add_argument('--n_trials', type=int, default=None)
    args = parser.parse_args()

    run_tuning(**vars(args))


if __name__ == '__main__':
    main()
