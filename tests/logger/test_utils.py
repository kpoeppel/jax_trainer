"""Tests for the utils module in jax_trainer.logger.utils."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_trainer.logger.utils import count_parameters, flatten_dict, module_named_params
from jax_trainer.nnx_dummy import _NNX_IS_DUMMY, nnx

has_nnx = not _NNX_IS_DUMMY


def test_flatten_dict():
    """Test flatten_dict function."""
    # Test with default separator
    config = {
        "a": 1,
        "b": {
            "c": 2,
            "d": {
                "e": 3,
            },
        },
        "f": {
            "g": 4,
        },
    }

    flattened = flatten_dict(config)
    assert flattened["a"] == 1
    assert flattened["b.c"] == 2
    assert flattened["b.d.e"] == 3
    assert flattened["f.g"] == 4
    assert len(flattened) == 4

    # Test with custom separator
    flattened = flatten_dict(config, separation_mark="/")
    assert flattened["a"] == 1
    assert flattened["b/c"] == 2
    assert flattened["b/d/e"] == 3
    assert flattened["f/g"] == 4
    assert len(flattened) == 4


@pytest.mark.skipif(not has_nnx, reason="NNX not available")
def test_module_named_params():
    """Test module_named_params function."""

    # Create a simple nnx module
    class SimpleModule(nnx.Module):
        def __init__(self):
            super().__init__()
            self.weight = nnx.Param(jnp.ones((2, 2)))
            self.bias = nnx.Param(jnp.zeros(2))

    # Create a nested module
    class NestedModule(nnx.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = SimpleModule()
            self.layer2 = SimpleModule()
            self.extra = nnx.Param(jnp.ones(3))

    # Test non-recursive
    simple_module = SimpleModule()
    params = list(module_named_params(simple_module, recursive=False))
    assert len(params) == 2
    param_names = [name for name, _ in params]
    assert "weight" in param_names
    assert "bias" in param_names

    # Test recursive
    nested_module = NestedModule()
    params = list(module_named_params(nested_module, recursive=True))
    assert len(params) == 5  # 2 from each SimpleModule + 1 from NestedModule
    param_names = [name for name, _ in params]
    assert "layer1.weight" in param_names
    assert "layer1.bias" in param_names
    assert "layer2.weight" in param_names
    assert "layer2.bias" in param_names
    assert "extra" in param_names


@pytest.mark.skipif(not has_nnx, reason="NNX not available")
def test_count_parameters():
    """Test count_parameters function."""

    # Create a simple module with known parameter count
    class TestModule(nnx.Module):
        def __init__(self):
            super().__init__()
            # 2x3 = 6 parameters
            self.weight1 = nnx.Param(jnp.ones((2, 3)))
            # 3 parameters
            self.bias1 = nnx.Param(jnp.zeros(3))
            # 3x4 = 12 parameters
            self.weight2 = nnx.Param(jnp.ones((3, 4)))
            # 4 parameters
            self.bias2 = nnx.Param(jnp.zeros(4))

    module = TestModule()
    param_count = count_parameters(module)
    assert param_count == 25  # 6 + 3 + 12 + 4 = 25
