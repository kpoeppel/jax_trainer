"""Tests for the scheduler module in jax_trainer.optimizer.scheduler."""

from collections import OrderedDict

import numpy as np
import pytest

from jax_trainer.optimizer.scheduler import (
    BaseScheduleConfig,
    ConcatSchedule,
    ConcatScheduleConfig,
    ConstantSchedule,
    ConstantScheduleConfig,
    CosineSchedule,
    CosineScheduleConfig,
    ExponentialSchedule,
    ExponentialScheduleConfig,
    LinearSchedule,
    LinearScheduleConfig,
    ScheduleInterface,
    WarmupCosineDecaySchedule,
    WarmupCosineDecayScheduleConfig,
)

# Config Tests


def test_base_schedule_config():
    """Test BaseScheduleConfig validation."""
    # Valid config
    config = BaseScheduleConfig(init_value=0.1, end_value=0.01, steps=100)
    np.testing.assert_allclose(config.init_value, 0.1)
    np.testing.assert_allclose(config.end_value, 0.01)
    assert config.steps == 100

    # Invalid config - steps must be >= 0
    with pytest.raises(AssertionError):
        BaseScheduleConfig(init_value=0.1, end_value=0.01, steps=-1)


def test_constant_schedule_config():
    """Test ConstantScheduleConfig validation."""
    # Valid config with explicit end_value
    config = ConstantScheduleConfig(init_value=0.1, end_value=0.1, steps=100)
    np.testing.assert_allclose(config.init_value, 0.1)
    np.testing.assert_allclose(config.end_value, 0.1)
    assert config.steps == 100

    # Valid config with None end_value (should be set to init_value)
    config = ConstantScheduleConfig(init_value=0.1, end_value=None, steps=100)
    np.testing.assert_allclose(config.end_value, 0.1)

    # Invalid config - init_value must equal end_value
    with pytest.raises(AssertionError):
        ConstantScheduleConfig(init_value=0.1, end_value=0.2, steps=100)


def test_cosine_schedule_config():
    """Test CosineScheduleConfig validation."""
    # Valid config with explicit end_value
    config = CosineScheduleConfig(init_value=0.1, end_value=0.01, decay_factor=0.1, steps=100)
    np.testing.assert_allclose(config.init_value, 0.1)
    np.testing.assert_allclose(config.end_value, 0.01)
    np.testing.assert_allclose(config.decay_factor, 0.1)
    assert config.steps == 100

    # Valid config with None end_value (should be set to init_value * decay_factor)
    config = CosineScheduleConfig(init_value=0.1, end_value=None, decay_factor=0.1, steps=100)
    np.testing.assert_allclose(config.end_value, 0.01)

    # Invalid config - end_value must equal init_value * decay_factor
    with pytest.raises(AssertionError):
        CosineScheduleConfig(init_value=0.1, end_value=0.02, decay_factor=0.1, steps=100)


def test_linear_schedule_config():
    """Test LinearScheduleConfig validation."""
    # Valid config
    config = LinearScheduleConfig(init_value=0.1, end_value=0.01, steps=100)
    np.testing.assert_allclose(config.init_value, 0.1)
    np.testing.assert_allclose(config.end_value, 0.01)
    assert config.steps == 100


def test_exponential_schedule_config():
    """Test ExponentialScheduleConfig validation."""
    # Valid config with explicit end_value
    config = ExponentialScheduleConfig(init_value=0.1, end_value=0.01, decay_rate=0.1, steps=100)
    np.testing.assert_allclose(config.init_value, 0.1)
    np.testing.assert_allclose(config.end_value, 0.01)
    np.testing.assert_allclose(config.decay_rate, 0.1)
    assert config.steps == 100

    # Valid config with None end_value (should be set to init_value * decay_rate)
    config = ExponentialScheduleConfig(init_value=0.1, end_value=None, decay_rate=0.1, steps=100)
    np.testing.assert_allclose(config.end_value, 0.01)

    # Invalid config - end_value must equal init_value * decay_rate
    with pytest.raises(AssertionError):
        ExponentialScheduleConfig(init_value=0.1, end_value=0.02, decay_rate=0.1, steps=100)


def test_concat_schedule_config():
    """Test ConcatScheduleConfig validation."""
    # Create sub-schedules
    sched1 = ConstantScheduleConfig(init_value=0.1, end_value=0.1, steps=50)
    sched2 = LinearScheduleConfig(init_value=0.1, end_value=0.01, steps=50)

    # Valid config
    schedules = OrderedDict([("constant", sched1), ("linear", sched2)])
    config = ConcatScheduleConfig(init_value=0.1, end_value=0.01, steps=100, schedules=schedules)
    np.testing.assert_allclose(config.init_value, 0.1)
    np.testing.assert_allclose(config.end_value, 0.01)
    assert config.steps == 100
    assert len(config.schedules) == 2

    # Invalid config - total steps don't match
    with pytest.raises(AssertionError):
        ConcatScheduleConfig(init_value=0.1, end_value=0.01, steps=90, schedules=schedules)

    # Invalid config - init_value doesn't match first schedule
    sched1_bad = ConstantScheduleConfig(init_value=0.2, end_value=0.2, steps=50)
    schedules_bad = OrderedDict([("constant", sched1_bad), ("linear", sched2)])
    with pytest.raises(AssertionError):
        ConcatScheduleConfig(init_value=0.1, end_value=0.01, steps=100, schedules=schedules_bad)

    # Invalid config - end_value doesn't match last schedule
    sched2_bad = LinearScheduleConfig(init_value=0.1, end_value=0.02, steps=50)
    schedules_bad = OrderedDict([("constant", sched1), ("linear", sched2_bad)])
    with pytest.raises(AssertionError):
        ConcatScheduleConfig(init_value=0.1, end_value=0.01, steps=100, schedules=schedules_bad)


def test_warmup_cosine_decay_schedule_config():
    """Test WarmupCosineDecayScheduleConfig validation."""
    # Valid config
    config = WarmupCosineDecayScheduleConfig(
        init_value=0.0,
        peak_value=0.1,
        end_value=0.01,
        warmup_steps=20,
        decay_steps=80,
        steps=100,
    )
    np.testing.assert_allclose(config.init_value, 0.0)
    np.testing.assert_allclose(config.peak_value, 0.1)
    np.testing.assert_allclose(config.end_value, 0.01)
    assert config.warmup_steps == 20
    assert config.decay_steps == 80
    assert config.steps == 100

    # Invalid config - warmup_steps must be > 0
    with pytest.raises(AssertionError):
        WarmupCosineDecayScheduleConfig(
            init_value=0.0,
            peak_value=0.1,
            end_value=0.01,
            warmup_steps=0,
            decay_steps=80,
            steps=80,
        )

    # Invalid config - decay_steps must be > 0
    with pytest.raises(AssertionError):
        WarmupCosineDecayScheduleConfig(
            init_value=0.0,
            peak_value=0.1,
            end_value=0.01,
            warmup_steps=20,
            decay_steps=0,
            steps=20,
        )

    # Invalid config - warmup_steps + decay_steps must equal steps
    with pytest.raises(AssertionError):
        WarmupCosineDecayScheduleConfig(
            init_value=0.0,
            peak_value=0.1,
            end_value=0.01,
            warmup_steps=20,
            decay_steps=70,
            steps=100,
        )


# Scheduler Tests


def test_constant_schedule():
    """Test ConstantSchedule."""
    config = ConstantScheduleConfig(init_value=0.1, end_value=0.1, steps=100)
    scheduler = ConstantSchedule(config)

    # Value should remain constant at all steps
    np.testing.assert_allclose(scheduler(0), 0.1)
    np.testing.assert_allclose(scheduler(50), 0.1)
    np.testing.assert_allclose(scheduler(100), 0.1)
    np.testing.assert_allclose(scheduler(200), 0.1)  # Even beyond steps


def test_cosine_schedule():
    """Test CosineSchedule."""
    config = CosineScheduleConfig(init_value=0.1, end_value=0.01, decay_factor=0.1, steps=100)
    scheduler = CosineSchedule(config)

    # Value should start at init_value
    np.testing.assert_allclose(scheduler(0), 0.1)

    # Value should end at end_value
    np.testing.assert_allclose(scheduler(100), 0.01)

    # Value at middle step should be between init_value and end_value
    middle_value = scheduler(50)
    assert middle_value > 0.01
    assert middle_value < 0.1


def test_linear_schedule():
    """Test LinearSchedule."""
    config = LinearScheduleConfig(init_value=0.1, end_value=0.01, steps=100)
    scheduler = LinearSchedule(config)

    # Value should start at init_value
    np.testing.assert_allclose(scheduler(0), 0.1)

    # Value should end at end_value
    np.testing.assert_allclose(scheduler(100), 0.01)

    # Value at middle step should be the average of init_value and end_value
    np.testing.assert_allclose(scheduler(50), 0.055)


def test_exponential_schedule():
    """Test ExponentialSchedule."""
    config = ExponentialScheduleConfig(init_value=0.1, end_value=0.01, decay_rate=0.1, steps=100)
    scheduler = ExponentialSchedule(config)

    # Value should start at init_value
    np.testing.assert_allclose(scheduler(0), 0.1)

    # Value should approach end_value
    np.testing.assert_allclose(scheduler(100), 0.01)

    # Value at middle step should be between init_value and end_value
    middle_value = scheduler(50)
    assert middle_value > 0.01
    assert middle_value < 0.1


def test_concat_schedule():
    """Test ConcatSchedule."""
    # Create sub-schedules
    sched1 = ConstantScheduleConfig(init_value=0.1, end_value=0.1, steps=50)
    sched2 = LinearScheduleConfig(init_value=0.1, end_value=0.01, steps=50)

    schedules = OrderedDict([("constant", sched1), ("linear", sched2)])
    config = ConcatScheduleConfig(init_value=0.1, end_value=0.01, steps=100, schedules=schedules)
    scheduler = ConcatSchedule(config)

    # First half should be constant
    np.testing.assert_allclose(scheduler(0), 0.1)
    np.testing.assert_allclose(scheduler(25), 0.1)
    np.testing.assert_allclose(scheduler(49), 0.1)

    # Second half should be linear decay
    np.testing.assert_allclose(scheduler(50), 0.1)
    np.testing.assert_allclose(scheduler(75), 0.055)
    np.testing.assert_allclose(scheduler(100), 0.01)


def test_warmup_cosine_decay_schedule():
    """Test WarmupCosineDecaySchedule."""
    config = WarmupCosineDecayScheduleConfig(
        init_value=0.0,
        peak_value=0.1,
        end_value=0.01,
        warmup_steps=20,
        decay_steps=80,
        steps=100,
    )
    scheduler = WarmupCosineDecaySchedule(config)

    # Value should start at init_value
    np.testing.assert_allclose(scheduler(0), 0.0)

    # Value should reach peak_value at warmup_steps
    np.testing.assert_allclose(scheduler(20), 0.1)

    # Value should end at end_value
    np.testing.assert_allclose(scheduler(100), 0.01)

    # Value during warmup should increase linearly
    np.testing.assert_allclose(scheduler(10), 0.05)

    # Value during decay should follow cosine decay
    middle_decay = scheduler(60)
    assert middle_decay > 0.01
    assert middle_decay < 0.1
