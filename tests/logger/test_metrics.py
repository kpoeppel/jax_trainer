"""Tests for the metrics module in jax_trainer.logger.metrics."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.core import FrozenDict, freeze, unfreeze

from jax_trainer.logger.enums import LogFreq, LogMetricMode, LogMode
from jax_trainer.logger.metrics import get_metrics, update_metrics


def test_update_metrics_new_metrics():
    """Test update_metrics with new metrics (global_metrics=None)"""
    # Create step metrics
    step_metrics = {
        "loss": 0.5,
        "accuracy": 0.8,
    }

    # Update metrics
    global_metrics = update_metrics(
        global_metrics=None,
        step_metrics=step_metrics,
        train=True,
        batch_size=4,
    )

    # Check that metrics were created
    assert isinstance(global_metrics, FrozenDict)
    assert "loss_step" in global_metrics
    assert "loss_epoch" in global_metrics
    assert "accuracy_step" in global_metrics
    assert "accuracy_epoch" in global_metrics

    # Check values (multiplied by batch_size for MEAN mode)
    assert global_metrics["loss_step"]["value"] == 2.0  # 0.5 * 4
    assert global_metrics["loss_step"]["count"] == 4
    assert global_metrics["loss_step"]["mode"] == LogMetricMode.MEAN

    assert global_metrics["accuracy_step"]["value"] == 3.2  # 0.8 * 4
    assert global_metrics["accuracy_step"]["count"] == 4
    assert global_metrics["accuracy_step"]["mode"] == LogMetricMode.MEAN


def test_update_metrics_existing_metrics():
    """Test update_metrics with existing metrics."""
    # Create initial global metrics
    global_metrics = {
        "loss_step": {
            "value": 2.0,
            "count": 4,
            "mode": LogMetricMode.MEAN,
            "log_freq": LogFreq.STEP,
            "log_mode": LogMode.ANY,
        },
        "loss_epoch": {
            "value": 2.0,
            "count": 4,
            "mode": LogMetricMode.MEAN,
            "log_freq": LogFreq.EPOCH,
            "log_mode": LogMode.ANY,
        },
        "accuracy_step": {
            "value": 3.2,
            "count": 4,
            "mode": LogMetricMode.MEAN,
            "log_freq": LogFreq.STEP,
            "log_mode": LogMode.ANY,
        },
        "accuracy_epoch": {
            "value": 3.2,
            "count": 4,
            "mode": LogMetricMode.MEAN,
            "log_freq": LogFreq.EPOCH,
            "log_mode": LogMode.ANY,
        },
    }
    global_metrics = freeze(global_metrics)

    # Create new step metrics
    step_metrics = {
        "loss": 0.3,
        "accuracy": 0.9,
    }

    # Update metrics
    updated_metrics = update_metrics(
        global_metrics=global_metrics,
        step_metrics=step_metrics,
        train=True,
        batch_size=2,
    )

    # Check that metrics were updated
    assert isinstance(updated_metrics, FrozenDict)

    # Check values (added to existing values)
    assert updated_metrics["loss_step"]["value"] == 2.6  # 2.0 + (0.3 * 2)
    assert updated_metrics["loss_step"]["count"] == 6  # 4 + 2

    assert updated_metrics["accuracy_step"]["value"] == 5.0  # 3.2 + (0.9 * 2)
    assert updated_metrics["accuracy_step"]["count"] == 6  # 4 + 2


def test_update_metrics_with_different_modes():
    """Test update_metrics with different metric modes."""
    # Create step metrics with different modes
    step_metrics = {
        "mean_metric": {"value": 0.5, "mode": LogMetricMode.MEAN},
        "sum_metric": {"value": 0.5, "mode": LogMetricMode.SUM},
        "single_metric": {"value": 0.5, "mode": LogMetricMode.SINGLE},
        "max_metric": {"value": 0.5, "mode": LogMetricMode.MAX},
        "min_metric": {"value": 0.5, "mode": LogMetricMode.MIN},
    }

    # Update metrics
    global_metrics = update_metrics(
        global_metrics=None,
        step_metrics=step_metrics,
        train=True,
        batch_size=2,
    )

    # Check that metrics were created with correct modes
    assert global_metrics["mean_metric_step"]["mode"] == LogMetricMode.MEAN
    assert global_metrics["sum_metric_step"]["mode"] == LogMetricMode.SUM
    assert global_metrics["single_metric_step"]["mode"] == LogMetricMode.SINGLE
    assert global_metrics["max_metric_step"]["mode"] == LogMetricMode.MAX
    assert global_metrics["min_metric_step"]["mode"] == LogMetricMode.MIN

    # Check values based on mode
    assert global_metrics["mean_metric_step"]["value"] == 1.0  # 0.5 * 2
    assert global_metrics["mean_metric_step"]["count"] == 2

    assert global_metrics["sum_metric_step"]["value"] == 0.5  # Direct sum
    assert global_metrics["sum_metric_step"]["count"] == 1

    assert global_metrics["single_metric_step"]["value"] == 0.5  # Direct value
    assert global_metrics["single_metric_step"]["count"] == 1

    assert global_metrics["max_metric_step"]["value"] == 0.5  # Direct value for first update
    assert global_metrics["max_metric_step"]["count"] == 1

    assert global_metrics["min_metric_step"]["value"] == 0.5  # Direct value for first update
    assert global_metrics["min_metric_step"]["count"] == 1


def test_update_metrics_with_log_freq():
    """Test update_metrics with specific log frequencies."""
    # Create step metrics with specific log frequencies
    step_metrics = {
        "step_only": {"value": 0.5, "log_freq": LogFreq.STEP},
        "epoch_only": {"value": 0.5, "log_freq": LogFreq.EPOCH},
        "any_freq": {"value": 0.5, "log_freq": LogFreq.ANY},
    }

    # Update metrics
    global_metrics = update_metrics(
        global_metrics=None,
        step_metrics=step_metrics,
        train=True,
        batch_size=1,
    )

    # Check that metrics were created with correct frequencies
    assert "step_only_step" in global_metrics
    assert "step_only_epoch" not in global_metrics

    assert "epoch_only_step" not in global_metrics
    assert "epoch_only_epoch" in global_metrics

    assert "any_freq_step" in global_metrics
    assert "any_freq_epoch" in global_metrics


def test_update_metrics_with_log_mode():
    """Test update_metrics with specific log modes."""
    # Create step metrics with specific log modes
    step_metrics = {
        "train_only": {"value": 0.5, "log_mode": LogMode.TRAIN},
        "val_only": {"value": 0.5, "log_mode": LogMode.VAL},
        "any_mode": {"value": 0.5, "log_mode": LogMode.ANY},
    }

    # Update metrics in train mode
    train_metrics = update_metrics(
        global_metrics=None,
        step_metrics=step_metrics,
        train=True,
        batch_size=1,
    )

    # Check that only train and any metrics were created
    assert "train_only_step" in train_metrics
    assert "val_only_step" not in train_metrics
    assert "any_mode_step" in train_metrics

    # Update metrics in val mode (train=False)
    val_metrics = update_metrics(
        global_metrics=None,
        step_metrics=step_metrics,
        train=False,
        batch_size=1,
    )

    # Check that only val and any metrics were created
    assert "train_only_epoch" not in val_metrics
    assert "val_only_epoch" in val_metrics
    assert "any_mode_epoch" in val_metrics


def test_get_metrics():
    """Test get_metrics function."""
    # Create global metrics
    global_metrics = {
        "loss_step": {
            "value": 10.0,
            "count": 5,
            "mode": LogMetricMode.MEAN,
            "log_freq": LogFreq.STEP,
            "log_mode": LogMode.ANY,
        },
        "loss_epoch": {
            "value": 20.0,
            "count": 10,
            "mode": LogMetricMode.MEAN,
            "log_freq": LogFreq.EPOCH,
            "log_mode": LogMode.ANY,
        },
        "accuracy_step": {
            "value": 8.0,
            "count": 4,
            "mode": LogMetricMode.MEAN,
            "log_freq": LogFreq.STEP,
            "log_mode": LogMode.ANY,
        },
        "sum_metric_step": {
            "value": 5.0,
            "count": 1,
            "mode": LogMetricMode.SUM,
            "log_freq": LogFreq.STEP,
            "log_mode": LogMode.ANY,
        },
    }
    global_metrics = freeze(global_metrics)

    # Get step metrics
    updated_metrics, host_metrics = get_metrics(
        global_metrics=global_metrics,
        log_freq=LogFreq.STEP,
        reset_metrics=True,
    )

    # Check that only step metrics were returned
    assert "loss" in host_metrics
    assert "accuracy" in host_metrics
    assert "sum_metric" in host_metrics

    # Check that metrics were calculated correctly
    assert host_metrics["loss"] == 2.0  # 10.0 / 5
    assert host_metrics["accuracy"] == 2.0  # 8.0 / 4
    assert host_metrics["sum_metric"] == 5.0  # Direct sum

    # Check that step metrics were reset
    assert updated_metrics["loss_step"]["value"] == 0.0
    assert updated_metrics["loss_step"]["count"] == 0
    assert updated_metrics["accuracy_step"]["value"] == 0.0
    assert updated_metrics["accuracy_step"]["count"] == 0
    assert updated_metrics["sum_metric_step"]["value"] == 0.0
    assert updated_metrics["sum_metric_step"]["count"] == 0

    # Check that epoch metrics were not reset
    assert updated_metrics["loss_epoch"]["value"] == 20.0
    assert updated_metrics["loss_epoch"]["count"] == 10


def test_get_metrics_without_reset():
    """Test get_metrics function without resetting metrics."""
    # Create global metrics
    global_metrics = {
        "loss_step": {
            "value": 10.0,
            "count": 5,
            "mode": LogMetricMode.MEAN,
            "log_freq": LogFreq.STEP,
            "log_mode": LogMode.ANY,
        },
        "accuracy_step": {
            "value": 8.0,
            "count": 4,
            "mode": LogMetricMode.MEAN,
            "log_freq": LogFreq.STEP,
            "log_mode": LogMode.ANY,
        },
    }
    global_metrics = freeze(global_metrics)

    # Get metrics without reset
    updated_metrics, host_metrics = get_metrics(
        global_metrics=global_metrics,
        log_freq=LogFreq.STEP,
        reset_metrics=False,
    )

    # Check that metrics were calculated correctly
    assert host_metrics["loss"] == 2.0  # 10.0 / 5
    assert host_metrics["accuracy"] == 2.0  # 8.0 / 4

    # Check that metrics were not reset
    assert updated_metrics["loss_step"]["value"] == 10.0
    assert updated_metrics["loss_step"]["count"] == 5
    assert updated_metrics["accuracy_step"]["value"] == 8.0
    assert updated_metrics["accuracy_step"]["count"] == 4


def test_get_metrics_with_std_mode():
    """Test get_metrics function with STD mode metrics."""
    # Create global metrics with STD mode
    global_metrics = {
        "std_metric_step": {
            "value": 10.0,  # sum of values
            "value2": 30.0,  # sum of squared values
            "count": 5,
            "mode": LogMetricMode.STD,
            "log_freq": LogFreq.STEP,
            "log_mode": LogMode.ANY,
        },
    }
    global_metrics = freeze(global_metrics)

    # Get metrics
    _, host_metrics = get_metrics(
        global_metrics=global_metrics,
        log_freq=LogFreq.STEP,
        reset_metrics=False,
    )

    # Check that STD was calculated correctly
    # mean = 10.0 / 5 = 2.0
    # mean_squared = 30.0 / 5 = 6.0
    # variance = 6.0 - 2.0^2 = 6.0 - 4.0 = 2.0
    # std = sqrt(2.0) = 1.414...
    assert "std_metric" in host_metrics
    assert np.isclose(host_metrics["std_metric"], np.sqrt(2.0))
