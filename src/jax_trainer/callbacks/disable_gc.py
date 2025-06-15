import gc
import time
from dataclasses import dataclass

import jax
import optax
from absl import logging
from compoconf import register

from jax_trainer.callbacks.callback import Callback, CallbackConfig


@dataclass
class DisableGCConfig(CallbackConfig):
    pass


@register
class DisableGC(Callback):
    """Callback to profile model training steps."""

    config: DisableGCConfig

    def __init__(self, config, trainer, data_module=None):
        super().__init__(config, trainer, data_module)

    def on_training_epoch_start(self, *args, **kwargs):
        gc.disable()

    def on_training_epoch_end(self, train_metrics, epoch_idx):
        gc.enable()
