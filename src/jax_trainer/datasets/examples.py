from dataclasses import dataclass, field
from typing import Any, Optional

import jax
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from compoconf import ConfigInterface, register
from torchvision.datasets import CIFAR10, MNIST

from .collate import build_batch_collate
from .data_struct import DatasetConfig, DatasetModule, SupervisedBatch
from .transforms import image_to_numpy, normalize_transform
from .utils import build_data_loaders


class LimitDataset(data.Dataset):
    def __init__(self, dataset: data.Dataset, limit: int):
        self.dataset = dataset
        self.limit = limit

    def __len__(self):
        return self.limit

    def __getitem__(self, idx: int):
        if idx >= self.limit:
            raise IndexError
        else:
            return self.dataset[idx]


@dataclass
class CIFAR10Config(DatasetConfig, ConfigInterface):
    """Configuration for CIFAR10 dataset."""

    _short_name: str = "cf10"
    data_dir: str | None = None
    normalize: bool = True
    val_size: int = 5000
    split_seed: int = 42
    limit_train_size: int | None = None
    local_batch_size: int = 128
    global_batch_size: int = 128
    resolution: tuple[int, int] = (32, 32)
    channels_first: int = False
    num_classes: int = 10
    num_channels: int = 3
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 4

    aux: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        assert self.data_dir is not None


@register
class CIFAR10Dataset(DatasetModule):
    config: CIFAR10Config
    """CIFAR10 dataset implementation."""

    def __init__(self, config: CIFAR10Config, mesh: Optional[jax.sharding.Mesh] = None):
        """Initialize CIFAR10 dataset.

        Args:
            config: Configuration for the dataset.
            mesh: Optional mesh for distributed training.
        """
        self.config = config
        transform = transforms.Compose(
            [
                image_to_numpy,
                normalize_transform(
                    mean=np.array([0.4914, 0.4822, 0.4465]), std=np.array([0.2023, 0.1994, 0.2010])
                )
                if config.normalize
                else transforms.Lambda(lambda x: x),
            ]
        )

        # Loading the training/validation set
        train_dataset = CIFAR10(
            root=config.data_dir, train=True, transform=transform, download=True
        )

        train_set, val_set = data.random_split(
            train_dataset,
            [50000 - config.val_size, config.val_size],
            generator=torch.Generator().manual_seed(config.split_seed),
        )

        limit_train_size = (
            config.limit_train_size if config.limit_train_size is not None else len(train_set)
        )
        train_set = LimitDataset(train_set, limit=limit_train_size)

        # Loading the test set
        test_set = CIFAR10(root=config.data_dir, train=False, transform=transform, download=True)

        train_loader, val_loader, test_loader = build_data_loaders(
            train_set,
            val_set,
            test_set,
            train=[True, False, False],
            collate_fn=build_batch_collate(SupervisedBatch),
            world_size=jax.process_count(),
            rank=jax.process_index(),
            mesh=mesh,
            config=config,
        )

        super().__init__(
            config=config,
            train=train_set,
            val=val_set,
            test=test_set,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
        )


@dataclass
class MNISTConfig(ConfigInterface):
    """Configuration for MNIST dataset."""

    data_dir: str | None = None
    local_batch_size: int = 128
    global_batch_size: int = 128
    num_workers: int = 4
    normalize: bool = True
    val_size: int = 5000
    split_seed: int = 42
    pin_memory: bool = True
    prefetch_factor: int = 4

    def __post_init__(self):
        assert self.data_dir is not None


@register
class MNISTDataset(DatasetModule):
    """MNIST dataset implementation."""

    config: MNISTConfig

    def __init__(self, config: MNISTConfig, mesh: Optional[jax.sharding.Mesh] = None):
        """Initialize MNIST dataset.

        Args:
            config: Configuration for the dataset.
            mesh: Optional mesh for distributed training.
        """
        self.config = config
        transform = transforms.Compose(
            [
                image_to_numpy,
                normalize_transform(mean=np.array([0.1307]), std=np.array([0.3081]))
                if config.normalize
                else transforms.Lambda(lambda x: x),
            ]
        )

        # Loading the training/validation set
        train_dataset = MNIST(root=config.data_dir, train=True, transform=transform, download=True)

        train_set, val_set = data.random_split(
            train_dataset,
            [60000 - config.val_size, config.val_size],
            generator=torch.Generator().manual_seed(config.split_seed),
        )

        # Loading the test set
        test_set = MNIST(root=config.data_dir, train=False, transform=transform, download=True)

        train_loader, val_loader, test_loader = build_data_loaders(
            train_set,
            val_set,
            test_set,
            train=[True, False, False],
            collate_fn=build_batch_collate(SupervisedBatch),
            world_size=jax.process_count(),
            rank=jax.process_index(),
            mesh=mesh,
            config=config,
        )

        super().__init__(
            config=config,
            train=train_set,
            val=val_set,
            test=test_set,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
        )


# Legacy functions for backward compatibility
# def build_cifar10_datasets(dataset_config: ConfigDict, mesh: Optional[jax.sharding.Mesh] = None):
#     """Builds CIFAR10 datasets.

#     Args:
#         dataset_config: Configuration for the dataset.
#         mesh: Optional mesh for distributed training.

#     Returns:
#         DatasetModule object.
#     """
#     # Convert ConfigDict to CIFAR10Config
#     config = CIFAR10Config(
#         data_dir=dataset_config.data_dir,
#         batch_size=dataset_config.get("batch_size", 128),
#         num_workers=dataset_config.get("num_workers", 4),
#         normalize=dataset_config.get("normalize", True),
#         val_size=dataset_config.get("val_size", 5000),
#         split_seed=dataset_config.get("split_seed", 42),
#         limit_train_size=dataset_config.get("limit_train_size", None),
#         pin_memory=dataset_config.get("pin_memory", True),
#         prefetch_factor=dataset_config.get("prefetch_factor", 4),
#     )

#     return CIFAR10Dataset(config, mesh)


# def build_mnist_datasets(dataset_config: ConfigDict, mesh: Optional[jax.sharding.Mesh] = None):
#     """Builds MNIST datasets.

#     Args:
#         dataset_config: Configuration for the dataset.
#         mesh: Optional mesh for distributed training.

#     Returns:
#         DatasetModule object.
#     """
#     # Convert ConfigDict to MNISTConfig
#     config = MNISTConfig(
#         data_dir=dataset_config.data_dir,
#         batch_size=dataset_config.get("batch_size", 128),
#         num_workers=dataset_config.get("num_workers", 4),
#         normalize=dataset_config.get("normalize", True),
#         val_size=dataset_config.get("val_size", 5000),
#         split_seed=dataset_config.get("split_seed", 42),
#         pin_memory=dataset_config.get("pin_memory", True),
#         prefetch_factor=dataset_config.get("prefetch_factor", 4),
#     )

#     return MNISTDataset(config, mesh)
