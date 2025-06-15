"""Tests for the array_storing module in jax_trainer.logger.array_storing."""

import os
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_trainer.logger.array_storing import (
    ArraySpec,
    array_to_spec,
    convert_from_array_spec,
    convert_to_array_spec,
    load_pytree,
    np_array_to_spec,
    save_pytree,
    spec_to_array,
)


def test_array_spec():
    """Test ArraySpec dataclass."""
    # Create an ArraySpec with default value
    spec = ArraySpec(shape=(2, 3), dtype=jnp.float32, device="cpu")
    assert spec.shape == (2, 3)
    assert spec.dtype == jnp.float32
    assert spec.device == "cpu"
    assert spec.value == 0

    # Create an ArraySpec with custom value
    spec = ArraySpec(shape=(2, 3), dtype=jnp.float32, device="cpu", value=1.5)
    assert spec.shape == (2, 3)
    assert spec.dtype == jnp.float32
    assert spec.device == "cpu"
    assert spec.value == 1.5


def test_array_to_spec():
    """Test array_to_spec function."""
    # Create a JAX array
    array = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    # Convert to spec
    spec = array_to_spec(array)

    # Check spec properties
    assert spec.shape == (2, 3)
    assert spec.dtype == jnp.float32
    assert spec.value == 1.0  # First element


def test_np_array_to_spec():
    """Test np_array_to_spec function."""
    # Create a NumPy array
    array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    # Convert to spec
    spec = np_array_to_spec(array)

    # Check spec properties
    assert spec.shape == (2, 3)
    assert spec.dtype == np.float64  # NumPy default float type
    assert spec.device == "numpy"
    assert spec.value == 1.0  # First element


def test_spec_to_array():
    """Test spec_to_array function."""
    # Test with numpy device
    spec = ArraySpec(shape=(2, 3), dtype=np.float32, device="numpy", value=7.0)
    array = spec_to_array(spec)
    assert isinstance(array, np.ndarray)
    assert array.shape == (2, 3)
    assert array.dtype == np.float32
    assert np.all(array == 7.0)

    # Test with JAX device
    spec = ArraySpec(shape=(2, 3), dtype=jnp.float32, device="cpu", value=5.0)
    array = spec_to_array(spec)
    assert isinstance(array, jnp.ndarray)
    assert array.shape == (2, 3)
    assert array.dtype == jnp.float32
    assert jnp.all(array == 5.0)


def test_convert_to_array_spec():
    """Test convert_to_array_spec function."""
    # Test with JAX array
    jax_array = jnp.array([1.0, 2.0, 3.0])
    result = convert_to_array_spec(jax_array)
    assert isinstance(result, ArraySpec)
    assert result.shape == (3,)

    # Test with NumPy array
    np_array = np.array([1.0, 2.0, 3.0])
    result = convert_to_array_spec(np_array)
    assert isinstance(result, ArraySpec)
    assert result.shape == (3,)

    # Test with non-array value
    value = 42
    result = convert_to_array_spec(value)
    assert result == 42  # Should return the original value


def test_convert_from_array_spec():
    """Test convert_from_array_spec function."""
    # Test with ArraySpec
    spec = ArraySpec(shape=(2, 2), dtype=jnp.float32, device="cpu", value=3.0)
    result = convert_from_array_spec(spec)
    assert isinstance(result, jnp.ndarray)
    assert result.shape == (2, 2)
    assert jnp.all(result == 3.0)

    # Test with non-ArraySpec value
    value = 42
    result = convert_from_array_spec(value)
    assert result == 42  # Should return the original value


def test_save_load_pytree():
    """Test save_pytree and load_pytree functions."""
    # Create a simple pytree with arrays
    pytree = {
        "a": jnp.array([1.0, 2.0, 3.0]),
        "b": {"c": jnp.array([[4.0, 5.0], [6.0, 7.0]]), "d": 42},
        "e": np.array([8.0, 9.0]),
    }

    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = temp_file.name

    try:
        # Save the pytree
        save_pytree(pytree, temp_path)

        # Load the pytree
        loaded_pytree = load_pytree(temp_path)

        # Check structure and values
        assert set(loaded_pytree.keys()) == set(pytree.keys())
        assert set(loaded_pytree["b"].keys()) == set(pytree["b"].keys())

        # Check array values
        assert loaded_pytree["a"].shape == pytree["a"].shape
        assert loaded_pytree["b"]["c"].shape == pytree["b"]["c"].shape
        assert loaded_pytree["b"]["d"] == pytree["b"]["d"]
        assert loaded_pytree["e"].shape == pytree["e"].shape

    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_save_load_pytree_with_path_object():
    """Test save_pytree and load_pytree functions with Path object."""
    # Create a simple pytree
    pytree = {"value": jnp.array([1.0, 2.0, 3.0])}

    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        # Save and load with Path object
        save_pytree(pytree, temp_path)
        loaded_pytree = load_pytree(temp_path)

        # Check values
        assert jnp.allclose(loaded_pytree["value"][0], pytree["value"][0])

    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)
