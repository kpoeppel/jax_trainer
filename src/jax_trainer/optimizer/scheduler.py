from abc import abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
import optax
from compoconf import (
    ConfigInterface,
    RegistrableConfigInterface,
    register,
    register_interface,
)


@register_interface
class ScheduleInterface(RegistrableConfigInterface):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


@dataclass
class BaseScheduleConfig(ConfigInterface):
    init_value: float = 1e-3
    end_value: float = 1e-3
    steps: int = -1

    def __post_init__(self):
        assert self.steps >= 0


@dataclass
class ConstantScheduleConfig(BaseScheduleConfig):
    end_value: None | float = None
    steps: int = -1

    def __post_init__(self):
        if self.end_value is None:
            self.end_value = self.init_value
        super().__post_init__()
        assert np.allclose(self.init_value, self.end_value)


@register
class ConstantSchedule(ScheduleInterface):
    config: ConstantScheduleConfig

    def __init__(self, config: ConstantScheduleConfig):
        self.config = config
        self._sched = optax.schedules.constant_schedule(value=self.config.init_value)

    def __call__(self, *args, **kwargs):
        return self._sched(*args, **kwargs)


@dataclass
class CosineScheduleConfig(BaseScheduleConfig):
    init_value: float = 1e-3
    end_value: float | None = None
    decay_factor: float = 0.1
    steps: int = -1
    exponent: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        if self.end_value is None:
            self.end_value = self.decay_factor * self.init_value
        assert np.allclose(self.end_value, self.decay_factor * self.init_value)


@register
class CosineSchedule(ScheduleInterface):
    config: CosineScheduleConfig

    def __init__(self, config: CosineScheduleConfig):
        self.config = config
        self._sched = optax.schedules.cosine_decay_schedule(
            init_value=config.init_value,
            decay_steps=config.steps,
            alpha=config.decay_factor,
            exponent=config.exponent,
        )

    def __call__(self, *args, **kwargs):
        return self._sched(*args, **kwargs)


@dataclass
class LinearScheduleConfig(BaseScheduleConfig):
    init_value: float = 1e-3
    end_value: float = 1e-3
    steps: int = -1

    def __post_init__(self):
        super().__post_init__()


@register
class LinearSchedule(ScheduleInterface):
    config: LinearScheduleConfig

    def __init__(self, config: LinearScheduleConfig):
        self.config = config
        self._sched = optax.schedules.linear_schedule(
            init_value=config.init_value, end_value=config.end_value, transition_steps=config.steps
        )

    def __call__(self, *args, **kwargs):
        return self._sched(*args, **kwargs)


@dataclass
class ExponentialScheduleConfig(BaseScheduleConfig):
    init_value: float = 1e-3
    end_value: float | None = None
    steps: int = -1
    decay_rate: float = 0.1

    def __post_init__(self):
        super().__post_init__()
        if self.end_value is None:
            self.end_value = self.decay_rate * self.init_value
        assert np.allclose(self.end_value, self.decay_rate * self.init_value)


@register
class ExponentialSchedule(ScheduleInterface):
    config: ExponentialScheduleConfig

    def __init__(self, config: ExponentialScheduleConfig):
        self.config = config
        self._sched = optax.schedules.exponential_decay(
            init_value=config.init_value,
            transition_steps=config.steps,
            decay_rate=config.decay_rate,
        )

    def __call__(self, *args, **kwargs):
        return self._sched(*args, **kwargs)


ElementaryScheduleConfig = (
    ConstantScheduleConfig
    | CosineScheduleConfig
    | LinearScheduleConfig
    | ExponentialScheduleConfig
)


@dataclass
class ConcatScheduleConfig(BaseScheduleConfig):
    init_value: float = 0.0
    end_value: float = 0.0
    steps: int = -1

    check_continuity: bool = True
    schedules: OrderedDict[str, ElementaryScheduleConfig] = field(
        default_factory=lambda: OrderedDict(const=ConstantScheduleConfig(value=1e-3))
    )

    def __post_init__(self):
        super().__post_init__()
        if self.check_continuity:
            assert self.steps == sum([sched.steps for _, sched in self.schedules.items()])

            prev_end_value = self.init_value
            for idx, (sname, sched) in enumerate(self.schedules.items()):
                assert np.allclose(sched.init_value, prev_end_value)
                prev_end_value = sched.end_value

            assert np.allclose(prev_end_value, self.end_value)


@register
class ConcatSchedule(ScheduleInterface):
    config: ConcatScheduleConfig

    def __init__(self, config: ConcatScheduleConfig):
        self.config = config
        sched_steps = [sched_config.steps for _, sched_config in config.schedules.items()]
        sched_transitions = [0]
        sched_transitions = [
            (sched_transitions[-1] + x) or sched_transitions[-1] for x in sched_steps
        ]
        self._sched = optax.schedules.join_schedules(
            [
                sched_config.instantiate(ScheduleInterface)
                for _, sched_config in config.schedules.items()
            ],
            boundaries=sched_transitions[:-1],
        )

    def __call__(self, *args, **kwargs):
        return self._sched(*args, **kwargs)


@dataclass
class WarmupCosineDecayScheduleConfig(BaseScheduleConfig):
    init_value: float = 0.0
    peak_value: float = 1e-3
    decay_factor: float = 1e-1
    warmup_steps: int = -1
    decay_steps: int = -1
    exponent: float = 1.0
    end_value: float | None = None

    def __post_init__(self):
        if self.steps < 0:
            self.steps = self.warmup_steps + self.decay_steps
        super().__post_init__()
        if self.end_value is None:
            self.end_value = self.decay_factor * self.peak_value
        assert np.allclose(self.peak_value * self.decay_factor, self.end_value)
        assert self.warmup_steps > 0
        assert self.decay_steps > 0
        assert self.warmup_steps + self.decay_steps == self.steps


@register
class WarmupCosineDecaySchedule(ScheduleInterface):
    config: WarmupCosineDecayScheduleConfig

    def __init__(self, config: WarmupCosineDecayScheduleConfig):
        self.config = config
        self._sched = optax.schedules.warmup_cosine_decay_schedule(
            init_value=config.init_value,
            peak_value=config.peak_value,
            warmup_steps=config.warmup_steps,
            decay_steps=config.decay_steps,
            end_value=config.end_value,
            exponent=config.exponent,
        )

    def __call__(self, *args, **kwargs):
        return self._sched(*args, **kwargs)
