from dataclasses import dataclass
from typing import Any, Iterable, SupportsIndex

import jax.numpy as jnp
import numpy as np
import torch.utils.data as data
from compoconf import RegistrableConfigInterface, register_interface
from flax.struct import dataclass as batch_dataclass

Dataset = data.Dataset | SupportsIndex
DataLoader = data.DataLoader | Iterable


@register_interface
@dataclass
class DatasetModule(RegistrableConfigInterface):
    """Data module class that holds the datasets and data loaders."""

    config: Any = None
    train: Dataset | None = None
    val: Dataset | None = None
    test: Dataset | None = None
    train_loader: DataLoader | None = None
    val_loader: DataLoader | None = None
    test_loader: DataLoader | None = None
    metadata: dict | None = None
    _short_name: str = ""


@dataclass
class DatasetConfig:
    local_batch_size: int = 128
    global_batch_size: int = 128
    num_workers: int = 4
    normalize: bool = True
    pin_memory: bool = True
    prefetch_factor: int = 4
    seed: int = 42


@batch_dataclass
class Batch:
    """Base class for batches.

    Attribute `size` is required and used, e.g. for logging.
    """

    size: int
    # Add any additional batch information here

    def __getitem__(self, key):
        vals = {}
        if isinstance(key, int):
            vals["size"] = 1
        for k, v in self.__dict__.items():
            if k == "size":
                continue
            if isinstance(v, (np.ndarray, jnp.ndarray)):
                vals[k] = v[key]
                if "size" not in vals:
                    vals["size"] = vals[k].shape[0]
            else:
                vals[k] = v
        return self.__class__(**vals)


@batch_dataclass
class SupervisedBatch(Batch):
    """Extension of the base batch class for supervised learning."""

    input: np.ndarray
    target: np.ndarray
