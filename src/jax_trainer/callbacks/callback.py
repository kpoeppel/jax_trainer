from dataclasses import dataclass
from typing import Any

from compoconf import ConfigInterface, RegistrableConfigInterface, register_interface

from jax_trainer.datasets import DatasetModule


@dataclass
class CallbackConfig(ConfigInterface):
    every_n_epochs: int = -1
    every_n_minutes: float = -1


@register_interface
class Callback(RegistrableConfigInterface):
    """Base class for callbacks."""

    def __init__(
        self, config: CallbackConfig, trainer: Any, data_module: DatasetModule | None = None
    ):
        """Base class for callbacks.

        Args:
            config: Configuration dictionary.
            trainer: Trainer object.
            data_module (optional): Data module object.
        """
        self.config = config
        self.trainer = trainer
        self.data_module = data_module
        self.every_n_epochs = config.every_n_epochs

    def on_training_start(self):
        """Called at the beginning of training."""
        pass

    def on_training_end(self):
        """Called at the end of training."""
        pass

    def on_training_epoch_start(self, epoch_idx):
        """Called at the beginning of each training epoch.

        Args:
            epoch_idx: Index of the current epoch.
        """
        if epoch_idx % self.every_n_epochs != 0:
            return
        self.on_filtered_training_epoch_start(epoch_idx)

    def on_filtered_training_epoch_start(self, epoch_idx):
        """Called at the beginning of each `every_n_epochs` training epoch. To
        be implemented by subclasses.

        Args:
            epoch_idx: Index of the current epoch.
        """
        pass

    def on_training_epoch_end(self, train_metrics, epoch_idx):
        """Called at the end of each training epoch.

        Args:
            train_metrics: Dictionary of training metrics of the current epoch.
            epoch_idx: Index of the current epoch.
        """
        if epoch_idx % self.every_n_epochs != 0:
            return
        self.on_filtered_training_epoch_end(train_metrics, epoch_idx)

    def on_filtered_training_epoch_end(self, train_metrics, epoch_idx):
        """Called at the end of each `every_n_epochs` training epoch. To be
        implemented by subclasses.

        Args:
            train_metrics: Dictionary of training metrics of the current epoch.
        """
        pass

    def on_validation_epoch_start(self, epoch_idx):
        """Called at the beginning of validation."""
        if epoch_idx % self.every_n_epochs != 0:
            return
        self.on_filtered_validation_epoch_start(epoch_idx)

    def on_filtered_validation_epoch_start(self, epoch_idx):
        """Called at the beginning of `every_n_epochs` validation. To be
        implemented by subclasses.

        Args:
            epoch_idx: Index of the current epoch.
        """
        pass

    def on_validation_epoch_end(self, eval_metrics, epoch_idx):
        """Called at the end of each validation epoch.

        Args:
            eval_metrics: Dictionary of evaluation metrics of the current epoch.
            epoch_idx: Index of the current epoch.
        """
        if epoch_idx % self.every_n_epochs != 0:
            return
        self.on_filtered_validation_epoch_end(eval_metrics, epoch_idx)

    def on_filtered_validation_epoch_end(self, eval_metrics, epoch_idx):
        """Called at the end of each `every_n_epochs` validation epoch. To be
        implemented by subclasses.

        Args:
            eval_metrics: Dictionary of evaluation metrics of the current epoch.
            epoch_idx: Index of the current epoch.
        """
        pass

    def on_test_epoch_start(self, epoch_idx):
        """Called at the beginning of testing.

        To be implemented by subclasses.
        """
        pass

    def on_test_epoch_end(self, test_metrics, epoch_idx):
        """Called at the end of each test epoch. To be implemented by
        subclasses.

        Args:
            test_metrics: Dictionary of test metrics of the current epoch.
            epoch_idx: Index of the current epoch.
        """
        pass

    def set_dataset(self, data_module: DatasetModule):
        """Sets the data module.

        Args:
            data_module: Data module object.
        """
        self.data_module = data_module

    def finalize(self, status: str | None = None):
        """Called at the end of the whole training process.

        To be implemented by subclasses.
        """
        pass
