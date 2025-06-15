from dataclasses import dataclass

import jax.numpy as jnp
from compoconf import ConfigInterface, register
from flax import linen as nn

from jax_trainer.interfaces import BaseModelNNX
from jax_trainer.nnx_dummy import nnx


@dataclass
class SimpleClassifierConfig(ConfigInterface):
    c_hid: int
    num_classes: int
    act_fn: str
    batch_norm: bool = True
    mh_norm: bool = True


@register
class SimpleClassifier(BaseModelNNX):
    config_class = SimpleClassifierConfig

    def __init__(self, config: SimpleClassifierConfig, rngs: nnx.Rngs):
        super().__init__(config)
        self.config = config
        self.act_fn = getattr(nnx, self.config.act_fn)
        self.conv1 = nnx.Conv(
            in_features=3, out_features=self.config.c_hid, kernel_size=(3, 3), strides=2, rngs=rngs
        )
        if self.config.batch_norm:
            self.norm1 = nnx.BatchNorm(num_features=self.config.c_hid, rngs=rngs)
        self.conv2 = nnx.Conv(
            in_features=self.config.c_hid,
            out_features=self.config.c_hid,
            kernel_size=(3, 3),
            rngs=rngs,
        )
        if self.config.mh_norm:
            norm = nn.vmap(
                target=nn.LayerNorm,
                variable_axes={"params": 0},
                in_axes=1,
                out_axes=1,
                axis_size=2,
                split_rngs={"params": True},
            )
            self.norm2 = nnx.bridge.ToNNX(norm(epsilon=1e-5, use_scale=True), rngs=rngs)
        self.head = nnx.Linear(in_features=8192, out_features=self.config.num_classes, rngs=rngs)

    def __call__(self, x, train=True, **kwargs):
        x = self.conv1(x)  # 32x32 => 16x16
        if self.config.batch_norm:
            x = self.norm1(x)
        x = self.act_fn(x)
        x = x
        if self.config.mh_norm:
            x_shape = x.shape
            x = self.norm2(x.reshape(x.shape[0], 2, *x_shape[1:-1], -1))
            x = x.reshape(x_shape)
        x = self.act_fn(x)
        x = x.reshape(x.shape[0], -1)  # Image grid to single feature vector
        x = self.head(x)
        x = nnx.log_softmax(x)
        return x
