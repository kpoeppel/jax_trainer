from abc import abstractmethod
from dataclasses import dataclass, field
from functools import partial

import jax.numpy as jnp
import optax
from compoconf import (
    ConfigInterface,
    RegistrableConfigInterface,
    register,
    register_interface,
)

from .scheduler import ScheduleInterface
from .transforms import GradientTransformInterface, WeightDecay, WeightDecayConfig


@register_interface
class OptimizerInterface(RegistrableConfigInterface):
    @abstractmethod
    def init(*args, **kwargs):
        pass

    @abstractmethod
    def update(*args, **kwargs):
        pass


@dataclass
class BaseOptimizerConfig(ConfigInterface):
    learning_rate: float | ScheduleInterface.cfgtype = 1e-3
    transforms: dict[str, GradientTransformInterface.cfgtype] = field(default_factory=dict)
    log_learning_rate: bool = True


@dataclass
class SGDConfig(BaseOptimizerConfig):
    learning_rate: float | ScheduleInterface.cfgtype = 1e-3
    transforms: dict[str, GradientTransformInterface.cfgtype] = field(default_factory=dict)
    momentum: float | None = None
    nesterov: bool = False


@register
class SGD(OptimizerInterface):
    config: SGDConfig

    def __init__(self, config: SGDConfig):
        self.config = config
        if not isinstance(self.config.learning_rate, float):
            self.scheduler = self.config.learning_rate.instantiate(ScheduleInterface)
        else:
            self.scheduler = self.config.learning_rate

        self.transforms = {
            key: transform.instantiate(GradientTransformInterface)
            for key, transform in self.config.transforms.items()
        }
        pre = [transf for _, transf in self.transforms.items() if transf.config.before_optimizer]
        post = [
            transf for _, transf in self.transforms.items() if not transf.config.before_optimizer
        ]
        if self.config.log_learning_rate:

            @partial(optax.inject_hyperparams, hyperparam_dtype=jnp.float32)
            def optimizer_fn(learning_rate):
                return optax.chain(
                    *pre,
                    optax.sgd(
                        learning_rate=learning_rate,
                        momentum=config.momentum,
                        nesterov=config.nesterov,
                    ),
                    *post,
                )

            self.full_opt = optimizer_fn(self.scheduler)
        else:
            self.full_opt = optax.chain(
                *pre,
                optax.sgd(
                    learning_rate=self.scheduler,
                    momentum=config.momentum,
                    nesterov=config.nesterov,
                ),
                *post,
            )

    def init(self, *args, **kwargs):
        return self.full_opt.init(*args, **kwargs)

    def update(self, *args, **kwargs):
        return self.full_opt.update(*args, **kwargs)


@dataclass
class AdamWConfig(BaseOptimizerConfig):
    learning_rate: float | ScheduleInterface.cfgtype = 1e-3
    transforms: dict[str, GradientTransformInterface.cfgtype] = field(default_factory=dict)
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8
    weight_decay: WeightDecayConfig = field(default_factory=WeightDecayConfig)
    nesterov: bool = False


@register
class AdamW(OptimizerInterface):
    config: AdamWConfig

    def __init__(self, config: AdamWConfig):
        self.config = config

        if not isinstance(self.config.learning_rate, float):
            self.scheduler = self.config.learning_rate.instantiate(ScheduleInterface)
        else:
            self.scheduler = self.config.learning_rate
        if isinstance(config.weight_decay, float):
            self.weight_decay = config.weight_decay
            self.weight_decay_mask = None
        else:
            self.weight_decay_func = WeightDecay(self.config.weight_decay)
            self.weight_decay = self.config.weight_decay.value
            self.weight_decay_mask = self.weight_decay_func.mask

        self.transforms = {
            key: transform.instantiate(GradientTransformInterface)
            for key, transform in self.config.transforms.items()
        }

        pre = [transf for _, transf in self.transforms.items() if transf.config.before_optimizer]
        post = [
            transf for _, transf in self.transforms.items() if not transf.config.before_optimizer
        ]

        if self.config.log_learning_rate:

            @partial(optax.inject_hyperparams, hyperparam_dtype=jnp.float32)
            def optimizer_fn(learning_rate):
                return optax.chain(
                    *pre,
                    optax.adamw(
                        learning_rate,
                        b1=config.b1,
                        b2=config.b2,
                        eps=config.eps,
                        weight_decay=self.weight_decay,
                        mask=self.weight_decay_mask,
                        nesterov=config.nesterov,
                    ),
                    *post,
                )

            self.full_opt = optimizer_fn(self.scheduler)
        else:
            self.full_opt = optax.chain(
                *pre,
                optax.adamw(
                    learning_rate=self.scheduler,
                    b1=config.b1,
                    b2=config.b2,
                    eps=config.eps,
                    weight_decay=self.weight_decay,
                    mask=self.weight_decay_mask,
                    nesterov=config.nesterov,
                ),
                *post,
            )

    def init(self, *args, **kwargs):
        return self.full_opt.init(*args, **kwargs)

    def update(self, *args, **kwargs):
        return self.full_opt.update(*args, **kwargs)


@dataclass
class LambConfig(BaseOptimizerConfig):
    learning_rate: float | ScheduleInterface.cfgtype = 1e-3
    transforms: dict[str, GradientTransformInterface.cfgtype] = field(default_factory=dict)
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-6
    eps_root: float = 0.0
    weight_decay: WeightDecayConfig = field(default_factory=WeightDecayConfig)


@register
class Lamb(OptimizerInterface):
    config: LambConfig

    def __init__(self, config: LambConfig):
        self.config = config

        if not isinstance(self.config.learning_rate, float):
            self.scheduler = self.config.learning_rate.instantiate(ScheduleInterface)
        else:
            self.scheduler = self.config.learning_rate
        if isinstance(config.weight_decay, float):
            self.weight_decay = config.weight_decay
            self.weight_decay_mask = None
        else:
            self.weight_decay_func = WeightDecay(self.config.weight_decay)
            self.weight_decay = self.config.weight_decay.value
            self.weight_decay_mask = self.weight_decay_func.mask

        self.transforms = {
            key: transform.instantiate(GradientTransformInterface)
            for key, transform in self.config.transforms.items()
        }

        pre = [transf for _, transf in self.transforms.items() if transf.config.before_optimizer]
        post = [
            transf for _, transf in self.transforms.items() if not transf.config.before_optimizer
        ]

        if self.config.log_learning_rate:

            @partial(optax.inject_hyperparams, hyperparam_dtype=jnp.float32)
            def optimizer_fn(learning_rate):
                return optax.chain(
                    *pre,
                    optax.scale_by_adam(
                        b1=self.config.b1,
                        b2=self.config.b2,
                        eps=self.config.eps,
                        eps_root=self.config.eps_root,
                    ),
                    optax.add_decayed_weights(
                        weight_decay=self.weight_decay, mask=self.weight_decay_mask
                    ),
                    # Change to use trust ratio on weight decay parameters only.
                    optax.masked(optax.scale_by_trust_ratio(), mask=self.weight_decay_mask),
                    optax.scale_by_learning_rate(learning_rate),
                    *post,
                )

            self.full_opt = optimizer_fn(self.scheduler)
        else:
            self.full_opt = optax.chain(
                *pre,
                optax.scale_by_adam(
                    b1=self.config.b1,
                    b2=self.config.b2,
                    eps=self.config.eps,
                    eps_root=self.config.eps_root,
                ),
                optax.add_decayed_weights(
                    weight_decay=self.weight_decay, mask=self.weight_decay_mask
                ),
                # Change to use trust ratio on weight decay parameters only.
                optax.masked(optax.scale_by_trust_ratio(), mask=self.weight_decay_mask),
                optax.scale_by_learning_rate(self.scheduler),
                *post,
            )

    def init(self, *args, **kwargs):
        return self.full_opt.init(*args, **kwargs)

    def update(self, *args, **kwargs):
        return self.full_opt.update(*args, **kwargs)
