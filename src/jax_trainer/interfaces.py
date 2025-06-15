from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

import jax
from compoconf import (
    ConfigInterface,
    RegistrableConfigInterface,
    register,
    register_interface,
)
from flax import linen as nn
from jax.sharding import Mesh

from .nnx_dummy import nnx

PyTree = Any


@dataclass
class BaseModelConfig:
    pass


@register_interface
class BaseModelNNX(nnx.Module, RegistrableConfigInterface):
    def __init__(self, config: BaseModelConfig, **kwargs):
        nnx.Module.__init__(self)
        RegistrableConfigInterface.__init__(self, config)
        self.config = config

    @abstractmethod
    def __call__(self, x: jax.Array) -> jax.Array:
        raise NotImplementedError


@register_interface
class BaseModelLinen(nn.Module, RegistrableConfigInterface):
    pass


@dataclass
class BaseAugmentationNNXConfig(ConfigInterface):
    pass


@dataclass
class BaseAugmentationLinenConfig(BaseAugmentationNNXConfig):
    pass


@register
@register_interface
class BaseAugmentationNNX(nnx.Module, RegistrableConfigInterface):
    config: BaseAugmentationNNXConfig

    def __init__(self, config: BaseAugmentationNNXConfig, wrapped_model: nnx.Module, **kwargs):
        nnx.Module.__init__(self)
        self.config = config
        self.wrapped_model = wrapped_model

    def augment(self, inp: PyTree, *args, **kwargs) -> PyTree:
        # default is no augmentation
        return inp

    def __call__(self, inp: PyTree, *args, **kwargs) -> PyTree:
        return self.wrapped_model(inp, *args, **kwargs)


@register
@register_interface
class BaseAugmentationLinen(nn.Module, RegistrableConfigInterface):
    config: BaseAugmentationLinenConfig
    wrapped_model: nn.Module | None = None
    mesh: Mesh | None = None

    def augment(self, inp: PyTree, *args, **kwargs) -> PyTree:
        # default is no augmentation
        return inp

    def __call__(self, inp: PyTree, *args, **kwargs) -> PyTree:
        return self.wrapped_model(inp, *args, **kwargs)


@dataclass
class BasePreprocessingNNXConfig(ConfigInterface):
    pass


@dataclass
class BasePreprocessingLinenConfig(BasePreprocessingNNXConfig):
    pass


@register
@register_interface
class BasePreprocessingNNX(nnx.Module, RegistrableConfigInterface):
    config_class = BasePreprocessingNNXConfig

    def __init__(self, config: BasePreprocessingNNXConfig, **kwargs):
        nnx.Module.__init__(self)
        self.config = config

    def preprocess(self, inp: PyTree, *args, **kwargs) -> PyTree:
        # default is no augmentation
        return inp


@register
@register_interface
class BasePreprocessingLinen(nn.Module, RegistrableConfigInterface):
    config: BasePreprocessingLinenConfig
    mesh: Mesh | None = None

    def preprocess(self, inp: PyTree, *args, **kwargs) -> PyTree:
        # default is no preprocessing
        return inp

    def __call__(self, inp: PyTree, *args, **kwargs) -> PyTree:
        # default is no preprocessing
        return inp
