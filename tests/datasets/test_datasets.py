import optax
from absl.testing import absltest
from ml_collections import ConfigDict

from jax_trainer.datasets import DatasetModule
from jax_trainer.datasets.examples import CIFAR10Config, CIFAR10Dataset


def test_build_cifar10():
    config = CIFAR10Config(
        **{
            "local_batch_size": 128,
            "global_batch_size": 128,
            "num_workers": 4,
            "data_dir": "data/",
        }
    )
    dataset_module = config.instantiate(DatasetModule)
    assert isinstance(dataset_module, CIFAR10Dataset)

    for loaders in [
        dataset_module.train_loader,
        dataset_module.val_loader,
        dataset_module.test_loader,
    ]:
        batch = next(iter(loaders))
        assert batch.size == 128
        assert batch.input.shape == (128, 32, 32, 3)
        assert batch.target.shape == (128,)
        assert batch.input.dtype == "float32"
        assert batch.target.dtype == "int64"
