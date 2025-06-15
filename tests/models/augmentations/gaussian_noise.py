from dataclasses import dataclass

import jax
import jax.numpy as jnp
from compoconf import register
from flax import linen as nn

from jax_trainer.interfaces import (
    BaseAugmentationLinen,
    BaseAugmentationLinenConfig,
    BaseAugmentationNNX,
    BaseAugmentationNNXConfig,
    BasePreprocessingLinen,
    BasePreprocessingLinenConfig,
    BasePreprocessingNNX,
    BasePreprocessingNNXConfig,
)
from jax_trainer.nnx_dummy import nnx


@dataclass
class GaussianNoiseAugmentationConfig(BaseAugmentationNNXConfig):
    """Configuration for Gaussian noise augmentation."""

    scale: float = 0.1


@register
class GaussianNoiseAugmentationNNX(BaseAugmentationNNX):
    """Adds Gaussian noise to the input for NNX models."""

    config: GaussianNoiseAugmentationConfig

    def __init__(
        self,
        config: GaussianNoiseAugmentationConfig,
        wrapped_model: nnx.Module,
        rngs: nnx.Rngs,
        **kwargs,
    ):
        super().__init__(config=config, wrapped_model=wrapped_model)
        self.config = config
        self.wrapped_model = wrapped_model
        self.augmentation = rngs.get("augmentation", None)

    def augment(self, inp, *args, **kwargs):
        """Add Gaussian noise to the input.

        Args:
            inp: Input data, can be a tuple of (images, labels) or just images

        Returns:
            Augmented input with the same structure as the input
        """
        if isinstance(inp, tuple):
            images, labels = inp
            noise = jax.random.normal(self.augmentation(), shape=images.shape) * self.config.scale
            return images + noise, labels
        else:
            noise = jax.random.normal(self.augmentation(), shape=inp.shape) * self.config.scale
            return inp + noise


@dataclass
class GaussianNoiseAugmentationLinenConfig(BaseAugmentationLinenConfig):
    """Configuration for Gaussian noise augmentation."""

    scale: float = 0.1


@register
class GaussianNoiseAugmentationLinen(BaseAugmentationLinen):
    """Adds Gaussian noise to the input for Linen models."""

    config: GaussianNoiseAugmentationLinenConfig
    wrapped_model: nn.Module

    def augment(self, inp, *, rng=None, **kwargs):
        """Add Gaussian noise to the input.

        Args:
            inp: Input data, can be a tuple of (images, labels) or just images
            rng: Random number generator key

        Returns:
            Augmented input with the same structure as the input
        """
        if isinstance(inp, tuple):
            images, labels = inp
            noise = jax.random.normal(rng["augmentation"], shape=images.shape) * self.config.scale
            return images + noise, labels
        else:
            noise = jax.random.normal(rng["augmentation"], shape=inp.shape) * self.config.scale
            return inp + noise

    def __call__(self, inp, *, train=True, rng=None, **kwargs):
        """Forward pass through the model with augmentation during training.

        Args:
            inp: Input data
            train: Whether in training mode
            rng: Random number generator key

        Returns:
            Model output
        """
        return self.wrapped_model_instance(inp, train=train, **kwargs)


@dataclass
class TestPreprocessingConfig(BasePreprocessingNNXConfig):
    pass


@register
class TestPreprocessingNNX(BasePreprocessingNNX):
    def __init__(self, config: TestPreprocessingConfig, **kwargs):
        pass

    def preprocess(self, x):
        return (x[0] + 0.01, x[1])


@dataclass
class TestPreprocessingLinenConfig(BasePreprocessingLinenConfig):
    pass


@register
class TestPreprocessingLinen(BasePreprocessingLinen):
    config: TestPreprocessingLinenConfig

    def preprocess(self, x):
        return (x[0] + 0.01, x[1])
