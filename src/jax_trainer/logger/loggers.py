import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from compoconf import (
    ConfigInterface,
    RegistrableConfigInterface,
    dump_config,
    register,
    register_interface,
)

from .backend import TensorboardConfig, ToolLoggerInterface
from .enums import LogFreq, LogMetricMode, LogMode
from .metrics import HostMetrics, Metrics, get_metrics

LOGGER = logging.getLogger(__name__)


@dataclass
class LoggerConfig(ConfigInterface):
    log_steps_every: int = 50
    log_dir: str = ""
    log_file_verbosity: str = "warning"
    tool_config: ToolLoggerInterface.cfgtype = field(default_factory=TensorboardConfig)


@register
@register_interface
class Logger(RegistrableConfigInterface):
    """Logger class to log metrics, images, etc.

    to Tensorboard or Weights and Biases.
    """

    config: LoggerConfig

    def __init__(self, config: LoggerConfig, full_config: Any):
        """Logger class to log metrics, images, etc. to Tensorboard or Weights
        and Biases.

        Args:
            config (LoggerConfig): The logger config.
            full_config (dataclass | dict): The full config of the trainer, to be logged.
        """
        self.config = config
        self.full_config = full_config
        self.logger: ToolLoggerInterface = config.tool_config.instantiate(
            ToolLoggerInterface, full_config=dump_config(full_config)
        )
        self.logging_mode = "train"
        self.step_metrics = self._get_new_metrics_dict()
        self.step_count = 0
        self.full_step_counter = 0
        self.log_steps_every = config.log_steps_every
        self.epoch_metrics = self._get_new_metrics_dict()
        self.epoch_idx = 0
        self.epoch_element_count = 0
        self.epoch_step_count = 0
        self.epoch_log_prefix = ""
        self.epoch_start_time = None

    def _get_new_metrics_dict(self):
        """Returns a default dict for metrics, with the following structure: {
        'METRIC_KEY': { 'value': float | jnp.ndarray, 'mode': LogMetricMode } }

        Returns:
            Dict[str, Dict[str, Any]]: The default dict for metrics.
        """
        return defaultdict(lambda: {"value": 0, "mode": "mean"})

    def log_metrics(self, metrics: HostMetrics, step: int, log_postfix: str = ""):
        """Logs a dict of metrics to the tool of choice (e.g.
        Tensorboard/Wandb).

        Args:
            metrics (Dict[str, Any]): The metrics that should be logged.
            step (int): The step at which the metrics should be logged.
            log_postfix (str, optional): Postfix to append to the log key. Defaults to "".
        """
        metrics_to_log = {}
        for metric_key in metrics:
            metric_value = metrics[metric_key]
            if isinstance(metric_value, (jnp.ndarray, np.ndarray)):
                if metric_value.size == 1:
                    metric_value = metric_value.item()
                else:
                    continue
            save_key = metric_key
            if len(log_postfix) > 0:
                save_key += f"_{log_postfix}"
            metrics_to_log[save_key] = metric_value
        if len(metrics_to_log) > 0:
            self.logger.log_metrics(metrics_to_log, step=step)

    def log_scalar(
        self,
        metric_key: str,
        metric_value: float | jnp.ndarray,
        step: int,
        log_postfix: str = "",
    ):
        """Logs a single scalar metric to the tool of choice (e.g.
        Tensorboard/Wandb).

        Args:
            metric_key (str): The key of the metric to log.
            metric_value (Union[float, jnp.ndarray]): The value of the metric to log.
            step (int): The step at which the metric should be logged.
            log_postfix (str, optional): Postfix to append to the log key. Defaults to "".
        """
        self.log_metrics({metric_key: metric_value}, step, log_postfix)

    def finalize(self, status: str):
        """Closes the logger.

        Args:
            status (str): The status of the training run (e.g. success, failure).
        """
        self.logger.finalize(status)

    def start_epoch(self, epoch: int, mode: str = "train"):
        """Starts a new epoch.

        Args:
            epoch (int): The index of the epoch.
            mode (str, optional): The logging mode. Should be in ["train", "val", "test"]. Defaults to "train".
        """
        assert mode == "train" or mode.startswith("val") or mode.startswith("test")
        self.logging_mode = mode
        self.epoch_idx = epoch
        self._reset_epoch_metrics()

    def log_step(self, metrics: Metrics) -> Metrics:
        """Log metrics for a single step.

        Args:
            metrics: The metrics to log. Should follow the structure of the metrics in the metrics.py file.

        Returns:
            If the metrics are logged in this step, the metrics will be updated to reset all step-specific metrics.
            Other metrics, e.g. for epochs, will be kept. If the metrics are not logged in this step, the metrics
            will be returned unchanged.
        """
        self.epoch_step_count += 1
        # Log step metrics if applicable
        if self.logging_mode == "train" and self.log_steps_every > 0:
            self.step_count += 1
            self.full_step_counter += 1
            if self.step_count >= self.log_steps_every:
                if self.step_count > self.log_steps_every:
                    logging.warning(
                        f"Logging step count is {self.step_count} but should be {self.log_steps_every}."
                    )
                metrics, step_metrics = get_metrics(
                    metrics, log_freq=LogFreq.STEP, reset_metrics=True
                )
                final_step_metrics = self._finalize_metrics(metrics=step_metrics)
                self.log_metrics(
                    final_step_metrics, step=self.full_step_counter, log_postfix="step"
                )
                self._reset_step_metrics()
        return metrics

    def _reset_step_metrics(self):
        """Resets the step count for the current step."""
        self.step_count = 0

    def _reset_epoch_metrics(self):
        """Resets the step count for the current epoch."""
        self.epoch_metrics = {}
        self.epoch_step_count = 0
        self.epoch_start_time = time.time()

    def log_epoch_scalar(self, key: str, value: float | int | jnp.ndarray):
        """Logs a single scalar metric in the metric dict of the current epoch.

        Args:
            key (str): The key of the metric to log.
            value (Union[float, int, jnp.ndarray]): The value of the metric to log.
        """
        self.epoch_metrics[key] = value

    def _finalize_metrics(self, metrics: HostMetrics) -> HostMetrics:
        """Finalizes the metrics of the current epoch by aggregating them over
        the epoch, corresponding to their selected mode.

        Args:
            metrics (HostMetrics): The metrics to finalize.
        """
        final_metrics = {}
        for key in metrics:
            if "/" not in key:
                save_key = f"{self.logging_mode}/{key}"
            else:
                save_key = key
            final_metrics[save_key] = metrics[key]
        for key in final_metrics:
            val = final_metrics[key]
            if isinstance(val, (jnp.ndarray, np.ndarray)) and val.size == 1:
                val = val.item()
            final_metrics[key] = val
        return final_metrics

    def end_epoch(
        self, metrics: Metrics, save_metrics: bool = False
    ) -> Tuple[Metrics, HostMetrics]:
        """Ends the current epoch and logs the epoch metrics.

        Args:
            metrics (Metrics): The metrics that should be logged in this epoch.
            save_metrics (bool, optional): Whether to save the metrics to a file. Defaults to False.

        Returns:
            The originally passed metric dict will be updated by resetting all epoch-specific metrics.
            Other metrics, e.g. for steps, will be kept if the epoch size is not smaller than the
            step logging size. Otherwise, those will be reset as well. The metrics that are logged
            in this epoch will be returned as a separate dict.
        """
        self.log_epoch_scalar("time", time.time() - self.epoch_start_time)
        metrics, epoch_metrics = get_metrics(metrics, log_freq=LogFreq.EPOCH, reset_metrics=True)
        epoch_metrics.update(self.epoch_metrics)
        final_epoch_metrics = self._finalize_metrics(metrics=epoch_metrics)
        self.log_metrics(
            final_epoch_metrics,
            step=self.epoch_idx,
            log_postfix="epoch" if self.logging_mode == "train" else "",
        )
        if save_metrics:
            self.save_metrics(
                filename=f"{self.logging_mode}_epoch_{self.epoch_idx:04d}",
                metrics=final_epoch_metrics,
            )
        if (
            self.logging_mode == "train"
            and self.log_steps_every > 0
            and self.epoch_step_count < self.log_steps_every
        ):
            logging.info(
                "Training epoch has fewer steps than the logging frequency. Resetting step metrics."
            )
            metrics, _ = get_metrics(metrics, log_freq=LogFreq.STEP, reset_metrics=True)
            self._reset_step_metrics()
        self._reset_epoch_metrics()
        return metrics, final_epoch_metrics

    def save_metrics(self, filename: str, metrics: Dict[str, Any]):
        """Saves a dictionary of metrics to file. Can be used as a textual
        representation of the validation performance for checking in the
        terminal.

        Args:
          filename: Name of the metrics file without folders and postfix.
          metrics: A dictionary of metrics to save in the file.
        """
        metrics = {
            k: metrics[k] for k in metrics if isinstance(metrics[k], (int, float, str, bool))
        }
        os.makedirs(os.path.join(self.log_dir, "metrics"), exist_ok=True)
        with open(os.path.join(self.log_dir, f"metrics/{filename}.json"), "w") as f:
            json.dump(metrics, f, indent=4)

    def log_image(
        self,
        key: str,
        image: jnp.ndarray,
        step: int = None,
        log_postfix: str = "",
        logging_mode: str | None = None,
    ):
        """Logs an image to the tool of choice (e.g. Tensorboard/Wandb).

        Args:
          key: Name of the image.
          image: Image to log.
          step: Step to log the image at.
          log_postfix: Postfix to append to the log key.
        """
        if step is None:
            step = self.full_step_counter
        if logging_mode is None:
            logging_mode = self.logging_mode
        if isinstance(image, jnp.ndarray):
            image = jax.device_get(image)
        self.logger.log_image(
            key=key,
            image=image,
            log_postfix=log_postfix,
            logging_mode=logging_mode,
        )

    def log_figure(
        self,
        key: str,
        figure: plt.Figure,
        step: int = None,
        log_postfix: str = "",
        logging_mode: str | None = None,
    ):
        """Logs a matplotlib figure to the tool of choice (e.g.
        Tensorboard/Wandb).

        Args:
            key: Name of the figure.
            figure: Figure to log.
            step: Step to log the image at.
            log_postfix: Postfix to append to the log key.
        """
        if step is None:
            step = self.full_step_counter
        if logging_mode is None:
            logging_mode = self.logging_mode
        self.logger.log_figure(
            key, figure=figure, step=step, log_postfix=log_postfix, logging_mode=logging_mode
        )

    def log_embedding(
        self,
        key: str,
        encodings: np.ndarray,
        step: int | None = None,
        metadata: Any | None = None,
        images: np.ndarray | None = None,
        log_postfix: str = "",
        logging_mode: str | None = None,
    ):
        """Logs embeddings to the tool of choice (e.g. Tensorboard/Wandb).

        Args:
            key: Name of the figure.
            figure: Figure to log.
            step: Step to log the image at.
            log_postfix: Postfix to append to the log key.
        """
        if step is None:
            step = self.full_step_counter
        if logging_mode is None:
            logging_mode = self.logging_mode
        self.logger.log_embedding(
            key, encodings=encodings, step=step, metadata=metadata, label_img=images
        )

    @property
    def log_dir(self):
        """Returns the logging directory of the logger."""
        log_dir = self.logger.logger.log_dir
        if log_dir is None:
            log_dir = os.path.join(
                self.logger.config.save_dir,
                self.logger.config.version if self.logger.config.version is not None else "",
            )
        return log_dir
