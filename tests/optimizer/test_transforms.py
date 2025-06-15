"""Tests for the transforms module in jax_trainer.optimizer.transforms."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_trainer.optimizer.transforms import (
    GradClipNormConfig,
    GradClipValueConfig,
    GradientTransformInterface,
    WeightDecayConfig,
)

# Config Tests


def test_grad_clip_norm_config():
    """Test GradClipNormConfig validation."""
    # Valid config with default values
    config = GradClipNormConfig()
    np.testing.assert_allclose(config.max_norm, 1e8)
    assert config.before_optimizer is True

    # Valid config with custom values
    config = GradClipNormConfig(max_norm=1.0, before_optimizer=False)
    np.testing.assert_allclose(config.max_norm, 1.0)
    assert config.before_optimizer is False


def test_grad_clip_value_config():
    """Test GradClipValueConfig validation."""
    # Valid config with default values
    config = GradClipValueConfig()
    np.testing.assert_allclose(config.max_delta, 1e8)
    assert config.before_optimizer is True

    # Valid config with custom values
    config = GradClipValueConfig(max_delta=0.5, before_optimizer=False)
    np.testing.assert_allclose(config.max_delta, 0.5)
    assert config.before_optimizer is False


# Instantiation Tests


def test_grad_clip_norm_instantiation():
    """Test GradClipNorm instantiation from config."""
    # Create config
    config = GradClipNormConfig(max_norm=1.0)

    # Instantiate from config
    transform = config.instantiate(GradientTransformInterface)

    # Create simple params and gradients
    params = {"w": jnp.array([1.0, 2.0, 3.0])}
    grads = {"w": jnp.array([10.0, 20.0, 30.0])}

    # Initialize the transform
    state = transform.init(params)

    # Apply the transform
    clipped_grads, new_state = transform.update(grads, state, params)

    # Check that the gradients were clipped
    # The L2 norm of the original gradients is sqrt(10^2 + 20^2 + 30^2) = sqrt(1400) ≈ 37.42
    # After clipping to norm 1.0, each component should be scaled by 1.0/37.42 ≈ 0.0267
    expected_grads = {"w": jnp.array([10.0, 20.0, 30.0]) / jnp.sqrt(1400.0)}
    np.testing.assert_allclose(clipped_grads["w"], expected_grads["w"], rtol=1e-5)


def test_grad_clip_value_instantiation():
    """Test GradClipValue instantiation from config."""
    # Create config
    config = GradClipValueConfig(max_delta=0.5)

    # Instantiate from config
    transform = config.instantiate(GradientTransformInterface)

    # Create simple params and gradients
    params = {"w": jnp.array([1.0, 2.0, 3.0])}
    grads = {"w": jnp.array([0.3, 0.6, 0.9])}

    # Initialize the transform
    state = transform.init(params)

    # Apply the transform
    clipped_grads, new_state = transform.update(grads, state, params)

    # Check that the gradients were clipped
    # Values above 0.5 should be clipped to 0.5, values below -0.5 should be clipped to -0.5
    expected_grads = {"w": jnp.array([0.3, 0.5, 0.5])}
    np.testing.assert_allclose(clipped_grads["w"], expected_grads["w"])


def test_weight_decay_instantiation():
    """Test WeightDecay instantiation from config."""
    # Create config
    config = WeightDecayConfig(value=0.01)

    # Instantiate from config
    transform = config.instantiate(GradientTransformInterface)

    # Create simple params and gradients
    params = {
        "layer1": {"weight": jnp.array([1.0, 2.0, 3.0]), "bias": jnp.array([0.1, 0.2, 0.3])},
        "layer2": {"weight": jnp.array([4.0, 5.0, 6.0]), "bias": jnp.array([0.4, 0.5, 0.6])},
    }
    grads = {
        "layer1": {"weight": jnp.zeros([3]), "bias": jnp.zeros([3])},
        "layer2": {"weight": jnp.zeros([3]), "bias": jnp.zeros([3])},
    }

    # Initialize the transform
    state = transform.init(params)

    # Apply the transform
    updated_grads, new_state = transform.update(grads, state, params)

    # Check that weight decay was applied to weights but not biases
    assert jnp.all(updated_grads["layer1"]["weight"] != 0.0)
    assert jnp.all(updated_grads["layer1"]["bias"] == 0.0)
    assert jnp.all(updated_grads["layer2"]["weight"] != 0.0)
    assert jnp.all(updated_grads["layer2"]["bias"] == 0.0)

    # Check that weight decay is applied correctly
    # Weight decay adds -decay_rate * param to the gradient
    np.testing.assert_allclose(
        updated_grads["layer1"]["weight"], 0.01 * params["layer1"]["weight"]
    )
    np.testing.assert_allclose(
        updated_grads["layer2"]["weight"], 0.01 * params["layer2"]["weight"]
    )
