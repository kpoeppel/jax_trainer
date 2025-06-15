"""Tests for the optimizer module in jax_trainer.optimizer.optimizer."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from jax_trainer.optimizer.optimizer import (
    SGD,
    AdamW,
    AdamWConfig,
    BaseOptimizerConfig,
    Lamb,
    LambConfig,
    OptimizerInterface,
    SGDConfig,
)
from jax_trainer.optimizer.scheduler import (
    ConstantSchedule,
    ConstantScheduleConfig,
    LinearSchedule,
    LinearScheduleConfig,
)
from jax_trainer.optimizer.transforms import WeightDecayConfig

# Config Tests


def test_base_optimizer_config():
    """Test BaseOptimizerConfig validation."""
    # Valid config with float learning rate
    config = BaseOptimizerConfig(learning_rate=0.01)
    np.testing.assert_allclose(config.learning_rate, 0.01)

    # Valid config with scheduler
    scheduler_config = ConstantScheduleConfig(init_value=0.01, end_value=0.01, steps=100)
    config = BaseOptimizerConfig(learning_rate=scheduler_config)
    assert isinstance(config.learning_rate, ConstantScheduleConfig)


def test_sgd_config():
    """Test SGDConfig validation."""
    # Valid config with default values
    config = SGDConfig(learning_rate=0.01)
    np.testing.assert_allclose(config.learning_rate, 0.01)
    assert config.momentum is None
    assert config.nesterov is False

    # Valid config with custom values
    config = SGDConfig(learning_rate=0.01, momentum=0.9, nesterov=True)
    np.testing.assert_allclose(config.learning_rate, 0.01)
    np.testing.assert_allclose(config.momentum, 0.9)
    assert config.nesterov is True

    # Valid config with scheduler
    scheduler_config = ConstantScheduleConfig(init_value=0.01, end_value=0.01, steps=100)
    config = SGDConfig(learning_rate=scheduler_config, momentum=0.9, nesterov=True)
    assert isinstance(config.learning_rate, ConstantScheduleConfig)
    np.testing.assert_allclose(config.momentum, 0.9)
    assert config.nesterov is True


def test_adamw_config():
    """Test AdamWConfig validation."""
    # Valid config with default values
    config = AdamWConfig(learning_rate=0.01)
    np.testing.assert_allclose(config.learning_rate, 0.01)
    np.testing.assert_allclose(config.b1, 0.9)
    np.testing.assert_allclose(config.b2, 0.999)
    np.testing.assert_allclose(config.eps, 1e-8)
    assert isinstance(config.weight_decay, WeightDecayConfig)
    np.testing.assert_allclose(config.weight_decay.value, 0.0)

    # Valid config with custom values
    config = AdamWConfig(
        learning_rate=0.01, b1=0.8, b2=0.99, eps=1e-6, weight_decay=WeightDecayConfig(value=0.01)
    )
    np.testing.assert_allclose(config.learning_rate, 0.01)
    np.testing.assert_allclose(config.b1, 0.8)
    np.testing.assert_allclose(config.b2, 0.99)
    np.testing.assert_allclose(config.eps, 1e-6)
    np.testing.assert_allclose(config.weight_decay.value, 0.01)

    # Valid config with scheduler
    scheduler_config = ConstantScheduleConfig(init_value=0.01, end_value=0.01, steps=100)
    config = AdamWConfig(
        learning_rate=scheduler_config,
        b1=0.8,
        b2=0.99,
        eps=1e-6,
        weight_decay=WeightDecayConfig(value=0.01),
    )
    assert isinstance(config.learning_rate, ConstantScheduleConfig)
    np.testing.assert_allclose(config.b1, 0.8)
    np.testing.assert_allclose(config.b2, 0.99)
    np.testing.assert_allclose(config.eps, 1e-6)
    np.testing.assert_allclose(config.weight_decay.value, 0.01)


# Optimizer Tests


def test_sgd_with_float_lr():
    """Test SGD optimizer with float learning rate."""
    config = SGDConfig(learning_rate=0.01, momentum=0.9, nesterov=True)
    optimizer = SGD(config)

    # Check that the scheduler is a float
    assert isinstance(optimizer.scheduler, float)
    np.testing.assert_allclose(optimizer.scheduler, 0.01)

    # Check that the optimizer is initialized correctly
    params = {"w": jnp.array([1.0, 2.0, 3.0])}
    opt_state = optimizer.init(params)

    # Check that the update function works
    grads = {"w": jnp.array([0.1, 0.2, 0.3])}
    updates, new_opt_state = optimizer.update(grads, opt_state, params)

    # The optimizer should return updates
    assert "w" in updates
    assert updates["w"].shape == params["w"].shape


def test_sgd_with_scheduler():
    """Test SGD optimizer with scheduler."""
    scheduler_config = LinearScheduleConfig(init_value=0.01, end_value=0.001, steps=100)
    config = SGDConfig(learning_rate=scheduler_config, momentum=0.9, nesterov=True)
    optimizer = SGD(config)

    # Check that the scheduler is a LinearSchedule
    assert isinstance(optimizer.scheduler, LinearSchedule)

    # Check learning rate at different steps
    np.testing.assert_allclose(optimizer.scheduler(0), 0.01)
    np.testing.assert_allclose(optimizer.scheduler(50), 0.0055)
    np.testing.assert_allclose(optimizer.scheduler(100), 0.001)

    # Check that the optimizer is initialized correctly
    params = {"w": jnp.array([1.0, 2.0, 3.0])}
    opt_state = optimizer.init(params)

    # Check that the update function works
    grads = {"w": jnp.array([0.1, 0.2, 0.3])}
    updates, new_opt_state = optimizer.update(grads, opt_state, params)

    # The optimizer should return updates
    assert "w" in updates
    assert updates["w"].shape == params["w"].shape


def test_adamw_with_float_lr():
    """Test AdamW optimizer with float learning rate."""
    config = AdamWConfig(
        learning_rate=0.01, b1=0.8, b2=0.99, eps=1e-6, weight_decay=WeightDecayConfig(value=0.01)
    )
    optimizer = AdamW(config)

    # Check that the scheduler is a float
    assert isinstance(optimizer.scheduler, float)
    np.testing.assert_allclose(optimizer.scheduler, 0.01)

    # Check that weight decay is set correctly
    np.testing.assert_allclose(optimizer.weight_decay, 0.01)

    # Check that the optimizer is initialized correctly
    params = {"w": jnp.array([1.0, 2.0, 3.0])}
    opt_state = optimizer.init(params)

    # Check that the update function works
    grads = {"w": jnp.array([0.1, 0.2, 0.3])}
    updates, new_opt_state = optimizer.update(grads, opt_state, params)

    # The optimizer should return updates
    assert "w" in updates
    assert updates["w"].shape == params["w"].shape


def test_adamw_with_scheduler():
    """Test AdamW optimizer with scheduler."""
    scheduler_config = LinearScheduleConfig(init_value=0.01, end_value=0.001, steps=100)
    config = AdamWConfig(
        learning_rate=scheduler_config,
        b1=0.8,
        b2=0.99,
        eps=1e-6,
        weight_decay=WeightDecayConfig(value=0.01),
    )
    optimizer = AdamW(config)

    # Check that the scheduler is a LinearSchedule
    assert isinstance(optimizer.scheduler, LinearSchedule)

    # Check learning rate at different steps
    np.testing.assert_allclose(optimizer.scheduler(0), 0.01)
    np.testing.assert_allclose(optimizer.scheduler(50), 0.0055)
    np.testing.assert_allclose(optimizer.scheduler(100), 0.001)

    # Check that weight decay is set correctly
    np.testing.assert_allclose(optimizer.weight_decay, 0.01)

    # Check that the optimizer is initialized correctly
    params = {"w": jnp.array([1.0, 2.0, 3.0])}
    opt_state = optimizer.init(params)

    # Check that the update function works
    grads = {"w": jnp.array([0.1, 0.2, 0.3])}
    updates, new_opt_state = optimizer.update(grads, opt_state, params)

    # The optimizer should return updates
    assert "w" in updates
    assert updates["w"].shape == params["w"].shape


def test_adamw_with_weight_decay_mask():
    """Test AdamW optimizer with weight decay mask."""
    config = AdamWConfig(
        learning_rate=0.01,
        b1=0.8,
        b2=0.99,
        eps=1e-6,
        weight_decay=WeightDecayConfig(
            value=0.01, mode="whitelist", parameter_regex_include=".*weight.*"
        ),
    )
    optimizer = AdamW(config)

    # Check that the weight decay mask function exists
    assert optimizer.weight_decay_mask is not None

    # Create params with both weight and bias parameters
    params = {
        "layer1": {"weight": jnp.array([1.0, 2.0, 3.0]), "bias": jnp.array([0.1, 0.2, 0.3])},
        "layer2": {"weight": jnp.array([4.0, 5.0, 6.0]), "bias": jnp.array([0.4, 0.5, 0.6])},
    }

    # Apply the mask
    mask = optimizer.weight_decay_mask(params)

    # Check that weights are included (True) and biases are excluded (False)
    assert mask["layer1"]["weight"]
    assert not mask["layer1"]["bias"]
    assert mask["layer2"]["weight"]
    assert not mask["layer2"]["bias"]

    # Check that the optimizer is initialized correctly
    opt_state = optimizer.init(params)

    # Check that the update function works
    grads = {
        "layer1": {"weight": jnp.array([0.1, 0.2, 0.3]), "bias": jnp.array([0.01, 0.02, 0.03])},
        "layer2": {"weight": jnp.array([0.4, 0.5, 0.6]), "bias": jnp.array([0.04, 0.05, 0.06])},
    }
    updates, new_opt_state = optimizer.update(grads, opt_state, params)

    # The optimizer should return updates for all parameters
    assert "layer1" in updates
    assert "weight" in updates["layer1"]
    assert "bias" in updates["layer1"]
    assert "layer2" in updates
    assert "weight" in updates["layer2"]
    assert "bias" in updates["layer2"]

    # Check that the update function works for zero gradient
    opt_state = optimizer.init(params)
    grads = {
        "layer1": {"weight": jnp.zeros([3]), "bias": jnp.zeros([3])},
        "layer2": {"weight": jnp.zeros([3]), "bias": jnp.zeros([3])},
    }
    updates, new_opt_state = optimizer.update(grads, opt_state, params)

    # The optimizer should return non-zero updates for all weight-decay parameters, zero otherwise
    assert "layer1" in updates
    assert "weight" in updates["layer1"]
    assert "bias" in updates["layer1"]
    assert jnp.all(updates["layer1"]["weight"] != 0.0)
    assert jnp.all(updates["layer1"]["bias"] == 0.0)
    assert "layer2" in updates
    assert "weight" in updates["layer2"]
    assert "bias" in updates["layer2"]
    assert jnp.all(updates["layer2"]["weight"] != 0.0)
    assert jnp.all(updates["layer2"]["bias"] == 0.0)


def test_lamb_with_weight_decay_mask():
    """Test Lamb optimizer with weight decay mask."""
    config = LambConfig(
        learning_rate=0.01,
        b1=0.8,
        b2=0.99,
        eps=1e-6,
        eps_root=0.0,
        weight_decay=WeightDecayConfig(
            value=0.01, mode="whitelist", parameter_regex_include=".*weight.*"
        ),
    )
    optimizer = Lamb(config)

    # Check that the weight decay mask function exists
    assert optimizer.weight_decay_mask is not None

    # Create params with both weight and bias parameters
    params = {
        "layer1": {"weight": jnp.array([1.0, 2.0, 3.0]), "bias": jnp.array([0.1, 0.2, 0.3])},
        "layer2": {"weight": jnp.array([4.0, 5.0, 6.0]), "bias": jnp.array([0.4, 0.5, 0.6])},
    }

    # Apply the mask
    mask = optimizer.weight_decay_mask(params)

    # Check that weights are included (True) and biases are excluded (False)
    assert mask["layer1"]["weight"]
    assert not mask["layer1"]["bias"]
    assert mask["layer2"]["weight"]
    assert not mask["layer2"]["bias"]

    # Check that the optimizer is initialized correctly
    opt_state = optimizer.init(params)

    # Check that the update function works
    grads = {
        "layer1": {"weight": jnp.array([0.1, 0.2, 0.3]), "bias": jnp.array([0.01, 0.02, 0.03])},
        "layer2": {"weight": jnp.array([0.4, 0.5, 0.6]), "bias": jnp.array([0.04, 0.05, 0.06])},
    }
    updates, new_opt_state = optimizer.update(grads, opt_state, params)

    # The optimizer should return updates for all parameters
    assert "layer1" in updates
    assert "weight" in updates["layer1"]
    assert "bias" in updates["layer1"]
    assert "layer2" in updates
    assert "weight" in updates["layer2"]
    assert "bias" in updates["layer2"]

    # Check that the update function works for zero gradient
    opt_state = optimizer.init(params)
    grads = {
        "layer1": {"weight": jnp.zeros([3]), "bias": jnp.zeros([3])},
        "layer2": {"weight": jnp.zeros([3]), "bias": jnp.zeros([3])},
    }
    updates, new_opt_state = optimizer.update(grads, opt_state, params)

    # The optimizer should return non-zero updates for all weight-decay parameters, zero otherwise
    assert "layer1" in updates
    assert "weight" in updates["layer1"]
    assert "bias" in updates["layer1"]
    assert jnp.all(updates["layer1"]["weight"] != 0.0)
    assert jnp.all(updates["layer1"]["bias"] == 0.0)
    assert "layer2" in updates
    assert "weight" in updates["layer2"]
    assert "bias" in updates["layer2"]
    assert jnp.all(updates["layer2"]["weight"] != 0.0)
    assert jnp.all(updates["layer2"]["bias"] == 0.0)
