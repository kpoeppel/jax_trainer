import time
from dataclasses import dataclass

import jax
from absl import logging
from compoconf import register

from jax_trainer.callbacks.callback import Callback, CallbackConfig
from jax_trainer.nnx_dummy import nnx


@dataclass
class JAXProfilerConfig(CallbackConfig):
    every_n_minutes: float = 60
    first_step: int = 10
    profile_n_steps: int = 5


@register
class JAXProfiler(Callback):
    """Callback to profile model training steps."""

    config: JAXProfilerConfig

    def __init__(self, config, trainer, data_module=None):
        super().__init__(config, trainer, data_module)
        self.log_dir = self.trainer.log_dir
        self.profiler_active = False
        self.profiler_last_time = None

    def on_training_start(self):
        self.profiler_active = False
        self.profiler_last_time = time.time()

    def on_training_step(self, step_metrics, epoch_idx, step_idx):
        if self.profiler_active:
            if step_idx >= self.profile_start_step + self.config.profile_n_steps:
                self.stop_trace()
        else:
            if (step_idx == self.config.first_step) or (
                time.time() - self.profiler_last_time > self.config.every_n_minutes * 60
            ):
                self.start_trace(step_idx)

    def on_training_epoch_end(self, train_metrics, epoch_idx):
        self.stop_trace()

    def start_trace(self, step_idx):
        if not self.profiler_active:
            logging.info(f"Starting trace at step {step_idx}.")
            jax.profiler.start_trace(self.log_dir)
            self.profiler_active = True
            self.profile_start_step = step_idx
        else:
            logging.warning("Trace already active.")

    def stop_trace(self):
        if self.profiler_active:
            logging.info("Stopping trace")
            if self.trainer.config.model_mode == "nnx":
                jax.tree_util.tree_map(
                    lambda x: x.block_until_ready(), nnx.state(self.trainer.model)
                )
            else:
                jax.tree_util.tree_map(lambda x: x.block_until_ready(), self.trainer.state.params)
            jax.profiler.stop_trace()
            self.profiler_last_time = time.time()
            self.profiler_active = False
