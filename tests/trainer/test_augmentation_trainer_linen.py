import os
import pathlib
from dataclasses import dataclass

import pytest
import yaml
from absl import logging

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

from compoconf import parse_config

from jax_trainer.datasets import DatasetModule
from jax_trainer.init_mesh import init_ddp_mesh
from jax_trainer.interfaces import BaseModelLinen, BaseModelNNX

# isort: off
from ..models import *  # noqa
from ..models.augmentations.gaussian_noise import *  # noqa


from jax_trainer.optimizer import OptimizerInterface
from jax_trainer.trainer.img_classifier import (
    AugmentedImgClassifierTrainer,
    AugmentedImgClassifierTrainerConfig,
)

# isort: on


@dataclass
class FullConfig:
    seed: int = 42
    num_gpus: int = 1
    trainer: AugmentedImgClassifierTrainerConfig | None = None
    model: BaseModelNNX.cfgtype | BaseModelLinen.cfgtype | None = None
    optimizer: OptimizerInterface.cfgtype | None = None
    dataset: DatasetModule.cfgtype | None = None


def load_and_update_config(tmpdir):
    """Load the config and update paths to use the temporary directory."""
    config = yaml.safe_load(pathlib.Path("tests/config/cifar10_classifier.yaml").read_text())

    # Update paths to use temporary directory
    log_dir = os.path.join(tmpdir, "checkpoints/AugmentationTrainerTest/")
    config["trainer"]["class_name"] = "AugmentedImgClassifierTrainer"
    config["trainer"]["model_mode"] = "linen"
    config["trainer"]["num_classes"] = 10
    config["model"]["class_name"] = "SimpleClassifierLinen"
    config["trainer"]["logger"]["log_dir"] = log_dir
    config["trainer"]["logger"]["tool_config"]["save_dir"] = log_dir

    # Add augmentation configuration
    config["trainer"]["augmentation"] = {
        "class_name": "GaussianNoiseAugmentationLinen",
        "scale": 0.1,
    }
    config["trainer"]["preprocessing"] = {
        "class_name": "TestPreprocessingLinen",
    }
    config["trainer"]["augmentation_seed"] = 43

    return parse_config(FullConfig, config)


@pytest.fixture
def cleanup_logging():
    """Fixture to clean up logging after test."""
    yield
    logging.get_absl_handler().flush()
    logging.get_absl_handler().close()


def test_augmentation_trainer_linen(tmpdir, cleanup_logging):
    """Test the AugmentationTrainer with Linen model."""
    config = load_and_update_config(tmpdir)

    mesh = init_ddp_mesh()
    dataset = config.dataset.instantiate(DatasetModule, mesh)
    exmp_input = next(iter(dataset.train_loader))

    trainer = AugmentedImgClassifierTrainer(
        config=config.trainer,
        model_config=config.model,
        optimizer_config=config.optimizer,
        exmp_input=exmp_input,
        mesh=mesh,
    )

    # Test initial accuracy (should be low)
    eval_metrics = trainer.test_model(test_loader=dataset.test_loader, epoch_idx=0)
    assert 0.25 > eval_metrics["test/acc"]

    # Train for a few epochs
    eval_metrics = trainer.train_model(
        train_loader=dataset.train_loader,
        val_loader=dataset.val_loader,
        test_loader=dataset.test_loader,
        num_epochs=trainer.config.train_epochs,
    )

    # Check that accuracy improved
    assert eval_metrics[5]["val/acc"] > 0.25
