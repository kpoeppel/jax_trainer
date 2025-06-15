from dataclasses import dataclass

import jax.numpy as jnp
from compoconf import ConfigInterface, register
from flax import linen as nn

from jax_trainer.interfaces import BaseModelLinen


@dataclass
class SimpleClassifierConfig(ConfigInterface):
    c_hid: int
    num_classes: int
    act_fn: str
    batch_norm: bool = True
    mh_norm: bool = True


@register
class SimpleClassifierLinen(BaseModelLinen):
    config: SimpleClassifierConfig

    def setup(self):
        self.act_fn = getattr(nn, self.config.act_fn)
        self.conv1 = nn.Conv(features=self.config.c_hid, kernel_size=(3, 3), strides=2)
        if self.config.batch_norm:
            self.norm1 = nn.BatchNorm()
        self.conv2 = nn.Conv(
            features=self.config.c_hid,
            kernel_size=(3, 3),
        )
        self.norm2 = nn.LayerNorm()
        self.head = nn.Dense(features=self.config.num_classes)

    def __call__(self, x, train=True, **kwargs):
        x = self.conv1(x)  # 32x32 => 16x16
        if self.config.batch_norm:
            x = self.norm1(x, use_running_average=not train)
        x = self.act_fn(x)
        x = x
        if self.config.mh_norm:
            x_shape = x.shape
            x = self.norm2(x.reshape(x.shape[0], 2, *x_shape[1:-1], -1))
            x = x.reshape(x_shape)
        x = self.act_fn(x)
        x = x.reshape(x.shape[0], -1)  # Image grid to single feature vector
        x = self.head(x)
        x = nn.log_softmax(x)
        return x
