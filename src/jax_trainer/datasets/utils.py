from collections.abc import Iterable
from typing import Any, Callable, Sequence

import torch
import torch.utils.data as data
from jax.sharding import Mesh
from torch.utils.data.distributed import DistributedSampler

from .collate import numpy_collate
from .data_struct import DatasetConfig
from .multihost_dataloading import MultiHostDataLoadIterator


def build_data_loaders(
    *datasets: Sequence[data.Dataset],
    train: bool | Sequence[bool] = True,
    collate_fn: Callable | list[Callable] = numpy_collate,
    config: DatasetConfig,
    world_size: int = 1,
    rank: int = 0,
    mesh: Mesh | None = None,
):
    """Creates data loaders used in JAX for a set of datasets.

    Args:
        datasets: Datasets for which data loaders are created.
        train: Sequence indicating which datasets are used for
            training and which not. If single bool, the same value
            is used for all datasets.
        batch_size: Batch size to use in the data loaders.
        num_workers: Number of workers for each dataset.
        seed: Seed to initialize the workers and shuffling with.

    Returns:
        List of data loaders.
    """
    loaders = []
    if not isinstance(train, (list, tuple)):
        train = [train for _ in datasets]
    batch_size = config.local_batch_size
    num_workers = config.num_workers
    seed = config.seed

    if not isinstance(collate_fn, Iterable):
        collate_fn = [collate_fn] * len(datasets)

    for dataset, is_train, col_fn in zip(datasets, train, collate_fn):
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=train,
            drop_last=train,
            seed=seed,
        )

        # print("NEW SAMPLER WITH RANK: ", rank, "of", world_size)
        loader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=is_train,
            collate_fn=col_fn,
            num_workers=num_workers,
            pin_memory=config.pin_memory,
            persistent_workers=is_train and (num_workers > 0),
            prefetch_factor=config.prefetch_factor if num_workers > 0 else None,
            generator=torch.Generator().manual_seed(seed),
        )

        if mesh is not None:
            loader = MultiHostDataLoadIterator(
                loader,
                iterator_length=len(loader),
                global_mesh=mesh,
                reset_after_epoch=True,
            )

        loaders.append(loader)
    return loaders
