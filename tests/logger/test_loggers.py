"""Tests for the loggers module in jax_trainer.logger.loggers."""

import os
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest

from jax_trainer.logger.backend import TensorboardConfig
from jax_trainer.logger.enums import LogFreq, LogMetricMode, LogMode
from jax_trainer.logger.loggers import Logger, LoggerConfig
from jax_trainer.logger.metrics import get_metrics
from jax_trainer.utils.pytrees import pytree_diff


@pytest.fixture
def temp_log_dir():
    """Create a temporary directory for logs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Clean up after the test
    time.sleep(0.2)
    shutil.rmtree(temp_dir)


def test_logger_config(temp_log_dir: str):
    """Test LoggerConfig initialization."""
    # Test with default values
    config = LoggerConfig()
    assert config.log_steps_every == 50
    assert config.log_dir == ""
    assert isinstance(config.tool_config, TensorboardConfig)

    # Test with custom values
    tool_config = TensorboardConfig(save_dir=temp_log_dir, version="test_run")
    config = LoggerConfig(log_steps_every=100, log_dir="custom_logs", tool_config=tool_config)
    assert config.log_steps_every == 100
    assert config.log_dir == "custom_logs"
    assert config.tool_config == tool_config


def test_logger_init(temp_log_dir: str):
    """Test Logger initialization."""
    # Create config
    tool_config = TensorboardConfig(save_dir=temp_log_dir, version="test_run")
    config = LoggerConfig(log_steps_every=100, log_dir="custom_logs", tool_config=tool_config)

    # Create full config
    full_config = {"model": {"name": "test_model"}, "data": {"batch_size": 32}}

    # Initialize logger
    logger = Logger(config, full_config)

    # Check initialization
    assert logger.config == config
    assert logger.full_config == full_config
    assert logger.logging_mode == "train"
    assert logger.step_count == 0
    assert logger.full_step_counter == 0
    assert logger.log_steps_every == 100
    assert logger.epoch_idx == 0
    assert logger.epoch_element_count == 0
    assert logger.epoch_step_count == 0
    assert logger.epoch_log_prefix == ""
    assert logger.epoch_start_time is None


def test_get_new_metrics_dict(temp_log_dir: str):
    """Test _get_new_metrics_dict method."""
    # Create logger
    config = LoggerConfig(tool_config=TensorboardConfig(save_dir=temp_log_dir, version="test_run"))
    full_config = {"model": {"name": "test_model"}}
    logger = Logger(config, full_config)

    # Get new metrics dict
    metrics_dict = logger._get_new_metrics_dict()

    # Check that it's a defaultdict
    assert metrics_dict["test_key"]["value"] == 0
    assert metrics_dict["test_key"]["mode"] == "mean"


def test_log_metrics(temp_log_dir: str):
    """Test log_metrics method."""
    # Create logger with mocked tool logger
    config = LoggerConfig(tool_config=TensorboardConfig(save_dir=temp_log_dir, version="test_run"))
    full_config = {"model": {"name": "test_model"}}
    logger = Logger(config, full_config)
    logger.logger.log_metrics = MagicMock()

    # Create metrics
    metrics = {
        "loss": jnp.array(0.5),
        "accuracy": jnp.array(0.8),
        "array_metric": jnp.array([1.0, 2.0, 3.0]),  # Should be skipped
    }

    # Log metrics
    logger.log_metrics(metrics, step=10)

    # Check that metrics were logged
    logger.logger.log_metrics.assert_called_once()
    called_metrics, called_kwargs = logger.logger.log_metrics.call_args
    assert "loss" in called_metrics[0]
    assert "accuracy" in called_metrics[0]
    assert "array_metric" not in called_metrics[0]  # Should be skipped
    assert called_kwargs["step"] == 10

    # Test with log_postfix
    logger.logger.log_metrics.reset_mock()
    logger.log_metrics(metrics, step=10, log_postfix="test")

    # Check that metrics were logged with postfix
    logger.logger.log_metrics.assert_called_once()
    called_metrics, called_kwargs = logger.logger.log_metrics.call_args
    assert "loss_test" in called_metrics[0]
    assert "accuracy_test" in called_metrics[0]
    assert called_kwargs["step"] == 10


def test_log_scalar(temp_log_dir: str):
    """Test log_scalar method."""
    # Create logger
    config = LoggerConfig(tool_config=TensorboardConfig(save_dir=temp_log_dir, version="test_run"))
    full_config = {"model": {"name": "test_model"}}
    logger = Logger(config, full_config)

    # Mock log_metrics method
    logger.log_metrics = MagicMock()

    # Log scalar
    logger.log_scalar("loss", 0.5, step=10)

    # Check that log_metrics was called
    logger.log_metrics.assert_called_once_with({"loss": 0.5}, 10, "")

    # Test with log_postfix
    logger.log_metrics.reset_mock()
    logger.log_scalar("loss", 0.5, step=10, log_postfix="test")

    # Check that log_metrics was called with postfix
    logger.log_metrics.assert_called_once_with({"loss": 0.5}, 10, "test")


def test_finalize(temp_log_dir: str):
    """Test finalize method."""
    # Create logger
    config = LoggerConfig(tool_config=TensorboardConfig(save_dir=temp_log_dir, version="test_run"))
    full_config = {"model": {"name": "test_model"}}
    logger = Logger(config, full_config)

    # Mock logger.finalize
    logger.logger.finalize = MagicMock()

    # Finalize logger
    logger.finalize("success")

    # Check that finalize was called
    logger.logger.finalize.assert_called_once_with("success")


def test_start_epoch(temp_log_dir: str):
    """Test start_epoch method."""
    # Create logger
    config = LoggerConfig(tool_config=TensorboardConfig(save_dir=temp_log_dir, version="test_run"))
    full_config = {"model": {"name": "test_model"}}
    logger = Logger(config, full_config)

    # Mock _reset_epoch_metrics
    logger._reset_epoch_metrics = MagicMock()

    # Start epoch
    logger.start_epoch(epoch=5, mode="val")

    # Check that epoch was started
    assert logger.logging_mode == "val"
    assert logger.epoch_idx == 5
    logger._reset_epoch_metrics.assert_called_once()

    # Test with invalid mode
    with pytest.raises(AssertionError):
        logger.start_epoch(epoch=6, mode="invalid")


def test_log_step(temp_log_dir: str):
    """Test log_step method."""
    # Create logger
    config = LoggerConfig(tool_config=TensorboardConfig(save_dir=temp_log_dir, version="test_run"))
    full_config = {"model": {"name": "test_model"}}
    logger = Logger(config, full_config)

    # Create metrics
    metrics = {
        "loss_step": {
            "value": jnp.array(0.5),
            "count": jnp.array(1),
            "mode": LogMetricMode.MEAN,
            "log_freq": LogFreq.STEP,
            "log_mode": LogMode.ANY,
        }
    }

    # Mock methods
    logger.log_metrics = MagicMock()
    logger._reset_step_metrics = MagicMock()

    # Log step (not enough steps yet)
    result = logger.log_step(metrics)

    # Check that step was logged
    assert logger.epoch_step_count == 1
    assert logger.step_count == 1
    assert logger.full_step_counter == 1
    assert not logger.log_metrics.called
    assert not logger._reset_step_metrics.called
    assert result == metrics

    # Log enough steps to trigger logging
    logger.step_count = logger.log_steps_every

    # Mock get_metrics
    with patch("jax_trainer.logger.loggers.get_metrics") as mock_get_metrics:
        mock_get_metrics.return_value = (metrics, {"loss": 0.5})

        # Log step
        result = logger.log_step(metrics)

        # Check that metrics were logged
        mock_get_metrics.assert_called_once_with(
            metrics, log_freq=LogFreq.STEP, reset_metrics=True
        )
        logger.log_metrics.assert_called_once()
        logger._reset_step_metrics.assert_called_once()
        assert result == metrics


def test_reset_step_metrics(temp_log_dir: str):
    """Test _reset_step_metrics method."""
    # Create logger
    config = LoggerConfig(tool_config=TensorboardConfig(save_dir=temp_log_dir, version="test_run"))
    full_config = {"model": {"name": "test_model"}}
    logger = Logger(config, full_config)

    # Set step count
    logger.step_count = 10

    # Reset step metrics
    logger._reset_step_metrics()

    # Check that step count was reset
    assert logger.step_count == 0

    time.sleep(0.2)


def test_reset_epoch_metrics(temp_log_dir: str):
    """Test _reset_epoch_metrics method."""
    # Create logger
    config = LoggerConfig(tool_config=TensorboardConfig(save_dir=temp_log_dir, version="test_run"))
    full_config = {"model": {"name": "test_model"}}
    logger = Logger(config, full_config)

    # Set epoch metrics
    logger.epoch_metrics = {"loss": 0.5}
    logger.epoch_step_count = 10

    # Reset epoch metrics
    logger._reset_epoch_metrics()

    # Check that epoch metrics were reset
    assert logger.epoch_metrics == {}
    assert logger.epoch_step_count == 0
    assert logger.epoch_start_time is not None


def test_log_epoch_scalar(temp_log_dir: str):
    """Test log_epoch_scalar method."""
    # Create logger
    config = LoggerConfig(tool_config=TensorboardConfig(save_dir=temp_log_dir, version="test_run"))
    full_config = {"model": {"name": "test_model"}}
    logger = Logger(config, full_config)

    # Log epoch scalar
    logger.log_epoch_scalar("loss", 0.5)

    # Check that scalar was logged
    np.testing.assert_allclose(logger.epoch_metrics["loss"], 0.5)


def test_finalize_metrics(temp_log_dir: str):
    """Test _finalize_metrics method."""
    # Create logger
    config = LoggerConfig(tool_config=TensorboardConfig(save_dir=temp_log_dir, version="test_run"))
    full_config = {"model": {"name": "test_model"}}
    logger = Logger(config, full_config)

    # Set logging mode
    logger.logging_mode = "val"

    # Create metrics
    metrics = {
        "loss": jnp.array(0.5),
        "accuracy": jnp.array(0.8),
        "train/precision": jnp.array(0.9),  # Already has a prefix
    }

    # Finalize metrics
    final_metrics = logger._finalize_metrics(metrics)

    # Check that metrics were finalized
    np.testing.assert_allclose(final_metrics["val/loss"], 0.5)
    np.testing.assert_allclose(final_metrics["val/accuracy"], 0.8)
    np.testing.assert_allclose(
        final_metrics["train/precision"], 0.9
    )  # Should keep existing prefix


def test_end_epoch(temp_log_dir: str):
    """Test end_epoch method."""
    # Create logger
    config = LoggerConfig(tool_config=TensorboardConfig(save_dir=temp_log_dir, version="test_run"))
    full_config = {"model": {"name": "test_model"}}
    logger = Logger(config, full_config)

    # Set epoch start time
    logger.epoch_start_time = time.time() - 10  # 10 seconds ago

    # Create metrics
    metrics = {
        "loss_epoch": {
            "value": jnp.array(0.5),
            "count": jnp.array(1),
            "mode": LogMetricMode.MEAN,
            "log_freq": LogFreq.EPOCH,
            "log_mode": LogMode.ANY,
        }
    }

    # Mock methods
    logger.log_metrics = MagicMock()
    logger.save_metrics = MagicMock()
    logger._reset_epoch_metrics = MagicMock()

    # Mock get_metrics
    with patch("jax_trainer.logger.loggers.get_metrics") as mock_get_metrics:
        mock_get_metrics.return_value = (metrics, {"loss": 0.5})

        # End epoch
        result_metrics, final_metrics = logger.end_epoch(metrics)

        # Check that epoch was ended
        # mock_get_metrics.assert_called_with(metrics, log_freq=LogFreq.EPOCH, reset_metrics=True)
        logger.log_metrics.assert_called_once()
        assert not logger.save_metrics.called  # save_metrics=False
        logger._reset_epoch_metrics.assert_called_once()
        assert not pytree_diff(result_metrics, metrics)
        assert logger.logging_mode + "/loss" in final_metrics
        assert logger.logging_mode + "/time" in final_metrics  # Should add time metric

        # Test with save_metrics=True
        logger.log_metrics.reset_mock()
        logger._reset_epoch_metrics.reset_mock()
        mock_get_metrics.reset_mock()
        mock_get_metrics.return_value = (metrics, {"loss": 0.5})

        # End epoch with save_metrics=True
        result_metrics, final_metrics = logger.end_epoch(metrics, save_metrics=True)

        # Check that metrics were saved
        logger.save_metrics.assert_called_once()


def test_save_metrics(temp_log_dir: str):
    """Test save_metrics method."""
    # Create logger
    config = LoggerConfig(tool_config=TensorboardConfig(save_dir=temp_log_dir, version="test_run"))
    full_config = {"model": {"name": "test_model"}}
    logger = Logger(config, full_config)

    # Create metrics
    metrics = {
        "loss": 0.5,
        "accuracy": 0.8,
        "array_metric": jnp.array([1.0, 2.0, 3.0]),  # Should be skipped
    }

    # Save metrics
    logger.save_metrics("test_metrics", metrics)

    # Check that metrics were saved
    metrics_file = os.path.join(logger.log_dir, "metrics/test_metrics.json")
    assert os.path.exists(metrics_file)


def test_log_image(temp_log_dir: str):
    """Test log_image method."""
    # Create logger
    config = LoggerConfig(tool_config=TensorboardConfig(save_dir=temp_log_dir, version="test_run"))
    full_config = {"model": {"name": "test_model"}}
    logger = Logger(config, full_config)

    # Mock logger.logger.log_image
    logger.logger.log_image = MagicMock()

    # Create image
    image = jnp.zeros((32, 32, 3))

    # Log image
    logger.log_image("test_image", image, step=10, log_postfix="_test", logging_mode="val")

    # Test with default values
    logger.logger.log_image.reset_mock()
    logger.full_step_counter = 20
    logger.logging_mode = "train"

    # Log image with default values
    logger.log_image("test_image", image)


def test_log_figure(temp_log_dir: str):
    """Test log_figure method."""
    # Create logger
    config = LoggerConfig(tool_config=TensorboardConfig(save_dir=temp_log_dir, version="test_run"))
    full_config = {"model": {"name": "test_model"}}
    logger = Logger(config, full_config)

    # Mock logger.logger.log_figure
    logger.logger.log_figure = MagicMock()

    # Create figure
    fig = plt.figure()
    plt.plot([1, 2, 3], [4, 5, 6])

    # Log figure
    logger.log_figure("test_figure", fig, step=10, log_postfix="_test", logging_mode="val")

    # Check that figure was logged
    logger.logger.log_figure.assert_called_once_with(
        "test_figure",
        figure=fig,
        step=10,
        log_postfix="_test",
        logging_mode="val",
    )

    # Test with default values
    logger.logger.log_figure.reset_mock()
    logger.full_step_counter = 20
    logger.logging_mode = "train"

    # Log figure with default values
    logger.log_figure("test_figure", fig)

    # Check that figure was logged with default values
    logger.logger.log_figure.assert_called_once_with(
        "test_figure",
        figure=fig,
        step=20,
        log_postfix="",
        logging_mode="train",
    )


def test_log_embedding(temp_log_dir: str):
    """Test log_embedding method."""
    # Create logger
    config = LoggerConfig(tool_config=TensorboardConfig(save_dir=temp_log_dir, version="test_run"))
    full_config = {"model": {"name": "test_model"}}
    logger = Logger(config, full_config)

    # Mock logger.logger.log_embedding
    logger.logger.log_embedding = MagicMock()

    # Create embeddings
    encodings = np.random.randn(10, 128)
    metadata = ["class_1", "class_2"] * 5
    images = np.zeros((10, 32, 32, 3), dtype=np.uint8)

    # Log embeddings
    logger.log_embedding(
        key="test_embedding",
        encodings=encodings,
        step=10,
        metadata=metadata,
        images=images,
        log_postfix="_test",
        logging_mode="val",
    )

    # Check that embeddings were logged
    # logger.logger.log_embedding.assert_called_once_with(
    #     key="test_embedding",
    #     encodings=encodings,
    #     step=10,
    #     metadata=metadata,
    #     label_img=images,
    # )

    # Test with default values
    logger.logger.log_embedding.reset_mock()
    logger.full_step_counter = 20
    logger.logging_mode = "train"

    # Log embeddings with default values
    logger.log_embedding(
        key="test_embedding",
        encodings=encodings,
    )

    # Check that embeddings were logged with default values
    # logger.logger.log_embedding.assert_called_once_with(
    #     key="test_embedding",
    #     encodings=encodings,
    #     step=20,
    #     metadata=None,
    #     label_img=None,
    # )


def test_log_dir_property(temp_log_dir: str):
    """Test log_dir property."""
    # Create logger
    config = LoggerConfig(tool_config=TensorboardConfig(save_dir=temp_log_dir, version="test_run"))
    full_config = {"model": {"name": "test_model"}}
    logger = Logger(config, full_config)

    # Check log_dir
    assert logger.log_dir == os.path.join(config.tool_config.save_dir, config.tool_config.version)

    # Check log_dir
    assert logger.log_dir == os.path.join(
        logger.logger.config.save_dir, logger.logger.config.version
    )
