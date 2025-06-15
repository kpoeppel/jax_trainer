"""Tests for the backend module in jax_trainer.logger.backend."""

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

from jax_trainer.logger.backend import (
    TensorboardConfig,
    TensorboardToolLogger,
    ToolLoggerInterface,
    WandbConfig,
    WandbToolLogger,
)


@pytest.fixture(autouse=True)
def wandb_offline_mode(monkeypatch):
    """Set Weights & Biases to offline mode for all tests."""
    monkeypatch.setenv("WANDB_MODE", "offline")


def test_tensorboard_config():
    """Test TensorboardConfig initialization."""
    # Test with default values
    config = TensorboardConfig()
    assert config.save_dir == ""
    assert config.name == ""
    assert config.version != ""  # Should generate a timestamp version
    assert config.use_timestamp_version is True

    # Test with custom values
    config = TensorboardConfig(
        save_dir="logs", name="test_name", version="test_run", use_timestamp_version=False
    )
    assert config.save_dir == "logs"
    assert config.name == "test_name"
    assert config.version == "test_run"
    assert config.use_timestamp_version is False


def test_wandb_config():
    """Test WandbConfig initialization."""
    # Test with default values
    config = WandbConfig()
    assert config.save_dir == ""
    assert config.version != ""  # Should generate a timestamp version
    assert config.name is None
    assert config.project is None
    assert config.entity is None
    assert config.use_timestamp_version is True

    # Test with custom values
    config = WandbConfig(
        save_dir="logs",
        version="test_run",
        name="test_name",
        project="test_project",
        entity="test_entity",
        use_timestamp_version=False,
    )
    assert config.save_dir == "logs"
    assert config.version == "test_run"
    assert config.name == "test_name"
    assert config.project == "test_project"
    assert config.entity == "test_entity"
    assert config.use_timestamp_version is False


@pytest.fixture
def temp_log_dir():
    """Create a temporary directory for logs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Clean up after the test
    shutil.rmtree(temp_dir)


def test_wandb_offline_mode():
    """Test that WANDB_MODE environment variable is set to offline."""
    assert os.environ.get("WANDB_MODE") == "offline"


def test_tensorboard_tool_logger(temp_log_dir):
    """Test TensorboardToolLogger functionality."""
    # Create config
    config = TensorboardConfig(save_dir=temp_log_dir, name="test_logger", version="test_run")

    # Initialize logger
    full_config = {"model": {"name": "test_model"}, "data": {"batch_size": 32}}

    # Create logger with a mocked experiment
    logger = TensorboardToolLogger(config, full_config)
    # logger.logger.experiment = MagicMock()
    logger.logger.log_metrics = MagicMock()
    logger.logger.finalize = MagicMock()

    # Test log_metrics
    metrics = {"loss": 0.5, "accuracy": 0.8}
    logger.log_metrics(metrics, step=10)
    logger.logger.log_metrics.assert_called_once_with(metrics, step=10)

    # Untested for now
    # # Test log_figure
    # fig = plt.figure()
    # plt.plot([1, 2, 3], [4, 5, 6])
    # logger.log_figure(key="test_figure", figure=fig, step=10, log_postfix="_test", logging_mode="train")
    # logger.logger.experiment.add_figure.assert_called_once_with(
    #     tag="train/test_figure_test",
    #     figure=fig,
    #     global_step=10,
    # )

    # # Test log_image
    # image = np.zeros((32, 32, 3), dtype=np.uint8)
    # logger.log_image(key="test_image", image=image, step=10, log_postfix="_test", logging_mode="train")
    # logger.logger.experiment.add_image.assert_called_once_with(
    #     tag="train/test_image_test",
    #     img_tensor=image,
    #     global_step=10,
    #     dataformats="HWC",
    # )

    # Test finalize
    logger.finalize(status="success")
    logger.logger.finalize.assert_called_once_with("success")


def test_wandb_tool_logger(temp_log_dir):
    """Test WandbToolLogger functionality."""
    # Create config
    config = WandbConfig(
        save_dir=temp_log_dir,
        version="test_run",
        name="test_name",
        project="test_project",
        entity="test_entity",
    )

    # Initialize logger
    full_config = {"model": {"name": "test_model"}, "data": {"batch_size": 32}}

    # Create logger with mocked methods
    logger = WandbToolLogger(config, full_config)
    logger.logger.log_metrics = MagicMock()
    logger.logger.log_image = MagicMock()
    logger.logger.finalize = MagicMock()
    # logger.logger.experiment = MagicMock()

    # Test log_metrics
    metrics = {"loss": 0.5, "accuracy": 0.8}
    logger.log_metrics(metrics, step=10)
    logger.logger.log_metrics.assert_called_once_with(metrics, step=10)

    # Test log_figure
    # fig = plt.figure()
    # plt.plot([1, 2, 3], [4, 5, 6])
    # logger.log_figure(key="test_figure", figure=fig, step=10, log_postfix="_test", logging_mode="train")
    # logger.logger.experiment.log.assert_called_once()

    # Test log_image
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    logger.log_image(
        key="test_image", image=image, step=10, log_postfix="_test", logging_mode="train"
    )
    logger.logger.log_image.assert_called_once_with(
        key="train/test_image_test",
        images=[image],
        step=10,
    )

    # Test finalize
    logger.finalize(status="success")
    logger.logger.finalize.assert_called_once_with("success")

    # Test log_embedding (should log a warning)
    encodings = np.random.randn(10, 128)
    with patch("jax_trainer.logger.backend.LOGGER") as mock_logger:
        logger.log_embedding(
            key="test_embedding",
            encodings=encodings,
            metadata=None,
            label_img=None,
            step=10,
        )
        mock_logger.warning.assert_called_once()
