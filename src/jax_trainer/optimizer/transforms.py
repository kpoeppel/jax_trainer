import re
from dataclasses import dataclass
from typing import Literal

import jax

try:
    from optax import transforms
    from optax.transforms import clip, clip_by_global_norm
except ImportError:
    from optax._src import transform as transforms
    from optax._src.clipping import clip_by_global_norm, clip

from compoconf import (
    ConfigInterface,
    RegistrableConfigInterface,
    assert_check_literals,
    from_annotations,
    register,
    register_interface,
)

from ..utils.pytrees import jax_path_to_str


@dataclass
class GradientTransformConfig(ConfigInterface):
    before_optimizer: bool = True


@register_interface
class GradientTransformInterface(RegistrableConfigInterface):
    def __init__(self, config):
        self.config


@dataclass
class WeightDecayConfig(GradientTransformConfig):
    value: float = 0.0
    mode: Literal["whitelist", "blacklist"] = "whitelist"
    parameter_regex_include: str | None = "((.*weight$)|(.*kernel$))"
    parameter_regex_exclude: str | None = ""

    def __post_init__(self):
        assert_check_literals(self)


@from_annotations(
    clip_by_global_norm,
    "GradClipNorm",
    GradientTransformInterface,
    default_as_pass_args=False,
    use_init=False,
)
@dataclass
class GradClipNormConfig(GradientTransformConfig):
    max_norm: float = 1e8


@from_annotations(
    clip,
    "GradClipValue",
    GradientTransformInterface,
    default_as_pass_args=False,
    use_init=False,
)
@dataclass
class GradClipValueConfig(GradientTransformConfig):
    max_delta: float = 1e8


@register
class WeightDecay(GradientTransformInterface):
    config: WeightDecayConfig

    def __init__(self, config: WeightDecayConfig):
        self.config = config
        self.transform = transforms.add_decayed_weights(weight_decay=config.value, mask=self.mask)

    def mask(self, params):
        def masked(path, leaf):
            path_str = jax_path_to_str(path)
            if self.config.mode == "whitelist":
                if re.match(self.config.parameter_regex_include, path_str):
                    if not self.config.parameter_regex_exclude or not re.match(
                        self.config.parameter_regex_exclude, path_str
                    ):
                        return True
                return False
            elif self.config.mode == "blacklist":
                if re.match(self.parameter_regex_exclude, path_str):
                    return False
                return True

        return jax.tree_util.tree_map_with_path(masked, params)

    def init(self, *args, **kwargs):
        return self.transform.init(*args, **kwargs)

    def update(self, *args, **kwargs):
        return self.transform.update(*args, **kwargs)
