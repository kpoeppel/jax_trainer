import json
import os
import pathlib
import shutil
from glob import glob

import numpy as np
import pytest
import yaml
from absl import logging
from ml_collections import ConfigDict

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

from dataclasses import dataclass

from compoconf import parse_config

from jax_trainer.datasets import DatasetModule
from jax_trainer.init_mesh import init_ddp_mesh
from jax_trainer.interfaces import BaseModelLinen, BaseModelNNX
from jax_trainer.optimizer import OptimizerInterface
from jax_trainer.trainer import ImgClassifierTrainer, ImgClassifierTrainerConfig

from ..models import *  # noqa


@dataclass
class FullConfig:
    seed: int = 42
    num_gpus: int = 1
    trainer: ImgClassifierTrainerConfig | None = None
    model: BaseModelNNX.cfgtype | BaseModelLinen.cfgtype | None = None
    optimizer: OptimizerInterface.cfgtype | None = None
    dataset: DatasetModule.cfgtype | None = None


def load_and_update_config(tmpdir):
    """Load the config and update paths to use the temporary directory."""
    config = yaml.safe_load(pathlib.Path("tests/config/cifar10_classifier.yaml").read_text())

    # Update paths to use temporary directory
    log_dir = os.path.join(tmpdir, "checkpoints/BuildTrainerTest/")
    config["trainer"]["model_mode"] = "linen"
    config["model"]["class_name"] = "SimpleClassifierLinen"
    config["trainer"]["logger"]["log_dir"] = log_dir
    config["trainer"]["logger"]["tool_config"]["save_dir"] = log_dir

    return parse_config(FullConfig, config)


@pytest.fixture
def cleanup_logging():
    """Fixture to clean up logging after test."""
    yield
    logging.get_absl_handler().flush()
    logging.get_absl_handler().close()


def test_build_trainer_ddp(tmpdir, cleanup_logging):
    config = load_and_update_config(tmpdir)

    mesh = init_ddp_mesh()
    dataset = config.dataset.instantiate(DatasetModule, mesh)
    exmp_input = next(iter(dataset.train_loader))
    trainer = ImgClassifierTrainer(
        config=config.trainer,
        model_config=config.model,
        optimizer_config=config.optimizer,
        exmp_input=exmp_input,
        mesh=mesh,
    )
    eval_metrics = trainer.test_model(test_loader=dataset.test_loader, epoch_idx=0)
    assert 0.25 > eval_metrics["test/acc"]
    eval_metrics = trainer.train_model(
        train_loader=dataset.train_loader,
        val_loader=dataset.val_loader,
        test_loader=dataset.test_loader,
        num_epochs=trainer.config.train_epochs,
    )
    assert eval_metrics[5]["val/acc"] > 0.25


def test_build_trainer_with_loading(tmpdir, cleanup_logging):
    """Test building a trainer and then loading it from a checkpoint."""
    # Step 1: Train the model and save checkpoints
    config = load_and_update_config(tmpdir)
    dataset = config.dataset.instantiate(DatasetModule, mesh=None)
    exmp_input = next(iter(dataset.train_loader))
    trainer = ImgClassifierTrainer(
        config=config.trainer,
        model_config=config.model,
        optimizer_config=config.optimizer,
        exmp_input=exmp_input,
    )

    # Train the model
    eval_metrics = trainer.train_model(
        train_loader=dataset.train_loader,
        val_loader=dataset.val_loader,
        test_loader=dataset.test_loader,
        num_epochs=trainer.config.train_epochs,
    )
    assert eval_metrics[5]["val/acc"] > 0.25

    # Test the model
    eval_metrics = trainer.test_model(test_loader=dataset.test_loader, epoch_idx=0)
    assert eval_metrics["test/acc"] > 0.25

    # Step 2: Load the trainer from checkpoint and verify it works
    log_dir = config.trainer.logger.log_dir
    ckpt_folder = sorted(glob(os.path.join(log_dir, "checkpoints/*")))[-1]
    exmp_input_file = os.path.join(log_dir, "exmp_input.pkl")

    # Load the trainer from checkpoint
    loaded_trainer = ImgClassifierTrainer.load_from_checkpoint(
        ckpt_folder, exmp_input_file=exmp_input_file
    )

    # Verify the loaded trainer has the correct configuration
    orig_config = ConfigDict(
        yaml.safe_load(pathlib.Path("tests/config/cifar10_classifier.yaml").read_text())
    )
    assert loaded_trainer.config.train_epochs == orig_config.trainer.train_epochs

    # Test the loaded model
    test_metrics = loaded_trainer.test_model(dataset.test_loader)
    assert test_metrics["test/acc"] > 0.25

    # Compare with the original metrics
    with open(os.path.join(log_dir, "metrics/test_epoch_0005.json"), "rb") as f:
        orig_test_metric = json.load(f)
    np.testing.assert_allclose(
        np.array(test_metrics["test/acc"]),
        np.array(orig_test_metric["test/acc"]),
        rtol=1e-3,
        atol=1e-3,
    )
